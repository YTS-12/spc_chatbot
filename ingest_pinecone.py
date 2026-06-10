"""K-IFRS PDF -> Pinecone 적재(인제스트) 파이프라인 (정본 / canonical).

개선 사항(2026-06-07):
- 한컴(HWP) PDF 추출 텍스트 클린업: 추출 시 소실된 공백 복원 + 페이지번호/푸터 제거
- 정규식 기반 (기준서 번호, 한글명) 추출 -> 파일명 포맷 변화에 강건
- 결정적(deterministic) 문서 id -> 재적재해도 중복 누적 없이 '덮어쓰기'
- 슬림 메타데이터: 불필요한 PDF 메타(author/creator/producer 등) 제거, 인용용 핵심만
- 빈/너무 짧은 청크 제거
- 배치 업서트 + 진행 로그
- (선택) `--reset` 으로 재적재 전 namespace 비우기

주의:
- 이 스크립트를 '실행'하면 임베딩 호출(비용)과 Pinecone 쓰기가 발생한다.
- 위 클린업/메타데이터 개선은 '다시 적재할 때' 비로소 인덱스에 반영된다.
- 임베딩 모델/차원(EMBEDDING_MODEL/EMBEDDING_DIMENSION)을 바꿨다면 반드시
  새 namespace 또는 `--reset` 으로 다시 만들어야 한다(차원 불일치 방지).

실행 예:
    python ingest_pinecone.py            # 추가 적재(결정적 id라 동일 청크는 덮어씀)
    python ingest_pinecone.py --reset    # namespace 비우고 처음부터 다시 적재
"""

import hashlib
import json
import re
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from graph.config import (
    CHUNK_CACHE_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    MIN_CHUNK_CHARS,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
)
from graph.retrieval import get_vectorstore


# 12개 Proto 기준서의 한글명(인용 품질용). 그 외 번호는 이름 없이 번호만 사용.
STANDARD_NAMES = {
    "개념체계": "재무보고를 위한 개념체계",
    "제1001호": "재무제표 표시",
    "제1007호": "현금흐름표",
    "제1012호": "법인세",
    "제1027호": "별도재무제표",
    "제1032호": "금융상품 표시",
    "제1107호": "금융상품 공시",
    "제1109호": "금융상품",
    "제1110호": "연결재무제표",
    "제1112호": "타 기업에 대한 지분의 공시",
    "제1113호": "공정가치 측정",
    "제1116호": "리스",
}

# "- 659 -" / "660" 처럼 줄 전체가 페이지번호인 라인
_PAGE_NUM_LINE = re.compile(r"^[ \t]*-?\s*\d{1,4}\s*-?[ \t]*$", re.MULTILINE)


def clean_text(text: str) -> str:
    """한컴 PDF 추출 텍스트의 대표적 손상(공백 소실, 페이지번호 혼입)을 보정.

    도메인 토큰(예: 제1109호, 문단 5.4.5 등 '숫자.숫자')은 깨지지 않도록
    '한글 왼쪽 경계'에서만 공백을 삽입한다(소수점/문단번호는 왼쪽이 숫자라 영향 없음).
    """
    if not text:
        return ""

    # 1) 줄 전체가 페이지번호인 라인 제거("- 659 -", "660" 등)
    text = _PAGE_NUM_LINE.sub("", text)

    # 2) 추출 시 사라진 공백 복원(안전한 경계에서만)
    #    한글 다음의 마침표/쉼표 뒤 -> 공백 ("하였다.이자" -> "하였다. 이자")
    text = re.sub(r"(?<=[가-힣])([.,])(?=\S)", r"\1 ", text)
    #    한글<->영문 경계 ("은행간IBOR" -> "은행간 IBOR")
    text = re.sub(r"(?<=[가-힣])(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=[가-힣])", " ", text)
    #    영문 camelCase 분리 ("FinancialStabilityBoard" -> "Financial Stability Board")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # 3) 공백/개행 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # 4) (선택) 고품질 한국어 띄어쓰기 모델이 설치돼 있으면 추가 보정
    #    pip install pykospacing  -> 설치 시 영문 외 한글 붙음까지 크게 개선
    try:
        from pykospacing import Spacing  # type: ignore

        text = Spacing()(text)
    except Exception:
        pass

    return text


def extract_standard(filename: str):
    """파일명에서 (기준서 번호, 한글명) 추출. 포맷 변화에 강건한 정규식 기반."""
    m = re.search(r"제\s*(\d{3,4})\s*호", filename)
    if m:
        no = f"제{m.group(1)}호"
        return no, STANDARD_NAMES.get(no, "")
    if "개념체계" in filename:
        return "개념체계", STANDARD_NAMES["개념체계"]
    return "unknown", ""


def classify_section(text: str) -> str:
    """페이지/청크를 문서 구조 유형으로 분류(1층).

    반환 코드: body(본문) | basis(결론도출근거) | minority(소수의견)
              | example(적용사례) | toc(목차)
    핵심 규범인 body 를 보존하고, 분량 많은 BC/소수의견/목차를 가려낸다.
    """
    t = text or ""
    bc = len(re.findall(r"BC\s?\d", t))
    ie = len(re.findall(r"\bIE\s?\d", t))
    has_basis = "결론도출근거" in t
    has_minority = ("소수의견" in t) or ("반대의견" in t)
    has_example = bool(re.search(r"적용사례|실무적용지침|적용지침\s*사례|설례", t))

    scores = {
        "minority": 3 if has_minority else 0,
        "basis": (2 if has_basis else 0) + min(bc, 5),
        "example": (2 if has_example else 0) + min(ie, 5),
    }
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best

    if (
        len(re.findall(r"[·•]", t)) >= 8
        or re.search(r"\.{5,}\s*\d", t)
        or t.count("…") >= 4
    ):
        return "toc"
    return "body"


_PARA_LEAD = re.compile(r"^\s*(?:문단\s*)?(B?C?\d{1,4}[A-Z]?(?:\.\d{1,3}){0,3})\b")
_PARA_INLINE = re.compile(r"문단\s*(B?C?\d{1,4}[A-Z]?(?:\.\d{1,3}){0,3})")


def extract_paragraph_no(text: str):
    """청크에서 K-IFRS 문단 번호를 best-effort 추출(예: '9', '5.4.5', 'B7', 'BC12').

    청크 맨 앞의 문단 번호를 우선, 없으면 본문의 첫 '문단 N' 참조를 사용한다.
    추출 실패 시 None(메타데이터에 넣지 않음). 인용 입도 향상용 보조 필드.
    """
    if not text:
        return None
    m = _PARA_LEAD.match(text)
    if m:
        return m.group(1)
    m = _PARA_INLINE.search(text)
    if m:
        return m.group(1)
    return None


def ensure_index() -> None:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY를 .env 파일에 설정하세요.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = pc.list_indexes()
    existing_names = set(indexes.names()) if hasattr(indexes, "names") else set()

    if PINECONE_INDEX_NAME not in existing_names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def load_documents():
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

    documents = []
    for pdf_path in pdf_paths:
        standard_no, standard_name = extract_standard(pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))

        for doc in loader.load():
            cleaned = clean_text(doc.page_content)
            if len(cleaned) < MIN_CHUNK_CHARS:
                continue  # 표지/목차/빈 페이지 등 저가치 페이지 제거

            raw_page = doc.metadata.get("page")
            page_label = doc.metadata.get("page_label")
            if page_label is None and isinstance(raw_page, int):
                page_label = str(raw_page + 1)  # 0-base -> 사람이 보는 1-base

            # 슬림 메타데이터로 재구성(노이즈 메타 제거, 인용용 핵심 필드만 유지)
            doc.page_content = cleaned
            doc.metadata = {
                "source_type": "k_ifrs",
                "source_file": pdf_path.name,
                "standard_no": standard_no,
                "standard_name": standard_name,
                "section_type": classify_section(cleaned),
                "authority_rank": 1,
                "page": raw_page,
                "page_label": page_label,
            }
            documents.append(doc)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # (source_file, page)별 chunk_index 부여 -> 결정적 id와 인용 추적에 사용
    result = []
    per_page_counter = {}
    for chunk in chunks:
        if len(chunk.page_content.strip()) < MIN_CHUNK_CHARS:
            continue
        key = (chunk.metadata.get("source_file"), chunk.metadata.get("page"))
        idx = per_page_counter.get(key, 0)
        per_page_counter[key] = idx + 1
        chunk.metadata["chunk_index"] = idx
        para = extract_paragraph_no(chunk.page_content)
        if para:
            chunk.metadata["paragraph_no"] = para
        result.append(chunk)
    return result


def make_id(metadata) -> str:
    """source_file + page + chunk_index 기반 결정적 id.

    동일 입력은 항상 동일 id -> 재적재 시 새로 쌓이지 않고 덮어써진다.
    """
    raw = (
        f"{metadata.get('source_file')}|"
        f"{metadata.get('page')}|"
        f"{metadata.get('chunk_index')}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def reset_namespace() -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"[reset] namespace '{PINECONE_NAMESPACE}' 비움 완료.")
    except Exception as exc:
        print(f"[reset] 건너뜀(비어있거나 없음): {exc}")


def main() -> None:
    do_reset = "--reset" in sys.argv

    ensure_index()
    if do_reset:
        reset_namespace()

    documents = load_documents()
    chunks = split_documents(documents)
    ids = [make_id(c.metadata) for c in chunks]

    # 하이브리드(BM25) 검색용 로컬 코퍼스 캐시 저장(Pinecone와 동일 청크/ids)
    with open(CHUNK_CACHE_PATH, "w", encoding="utf-8") as f:
        for cid, c in zip(ids, chunks):
            f.write(
                json.dumps(
                    {"id": cid, "text": c.page_content, "metadata": c.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[cache] {len(chunks)} chunks -> {CHUNK_CACHE_PATH.name}")

    vectorstore = get_vectorstore()

    total = len(chunks)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        batch_ids = ids[i : i + EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch, ids=batch_ids)
        print(f"  upserted {min(i + EMBED_BATCH_SIZE, total)}/{total}")

    print(
        f"완료: {total} chunks -> index '{PINECONE_INDEX_NAME}', "
        f"namespace '{PINECONE_NAMESPACE}'."
    )


if __name__ == "__main__":
    main()
