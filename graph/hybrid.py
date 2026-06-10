"""하이브리드 검색(3순위): dense(Pinecone) + sparse(BM25) 결과를 RRF로 융합.

dense 검색은 '의미 유사도'에 강하지만 '문단 12', '제1113호 수준 2' 같은
정확한 번호/용어 매칭엔 약하다. BM25(키워드)를 더해 보완한다.

코퍼스는 적재 시 저장된 chunks_cache.jsonl(Pinecone와 동일 청크/ids)을 사용한다.
rank_bm25 미설치 또는 캐시 없음이면 자동으로 dense-only로 동작한다(안전 degrade).
"""

import json
import re

from langchain_core.documents import Document

from graph.config import CHUNK_CACHE_PATH, RRF_K

_corpus = None
_bm25 = None
_loaded = False

# 숫자/영문/한글/기준서번호를 토큰으로(조사 분리 없이 단순 분해 — 정확매칭 목적)
_TOKEN = re.compile(r"제\d+호|[0-9]+|[a-zA-Z]+|[가-힣]+")


def _tokenize(text):
    return _TOKEN.findall((text or "").lower())


def _load():
    global _corpus, _bm25, _loaded
    if _loaded:
        return
    _loaded = True

    if not CHUNK_CACHE_PATH.exists():
        print(f"[hybrid] 코퍼스 캐시 없음({CHUNK_CACHE_PATH.name}) -> dense-only (재적재 필요)")
        return
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        print("[hybrid] rank_bm25 미설치 -> dense-only")
        return

    docs = []
    with open(CHUNK_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            md = dict(o.get("metadata", {}))
            md["_id"] = o.get("id")
            docs.append(Document(page_content=o.get("text", ""), metadata=md))

    if not docs:
        return
    _corpus = docs
    _bm25 = BM25Okapi([_tokenize(d.page_content) for d in docs])
    print(f"[hybrid] BM25 코퍼스 {len(docs)} 로드")


def available():
    _load()
    return _bm25 is not None


def bm25_search(query, k, allowed_sections=None, standards=None):
    """BM25 상위 k개 Document. 섹션/기준서 필터는 메모리에서 적용."""
    _load()
    if _bm25 is None:
        return []

    scores = _bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        d = _corpus[i]
        if allowed_sections and d.metadata.get("section_type") not in allowed_sections:
            continue
        if standards and d.metadata.get("standard_no") not in standards:
            continue
        out.append(d)
        if len(out) >= k:
            break
    return out


def chunk_key(d):
    m = d.metadata
    return (m.get("source_file"), m.get("page"), m.get("chunk_index"))


def rrf_fuse(result_lists, k=None):
    """여러 결과 리스트를 Reciprocal Rank Fusion으로 융합.

    chunk 식별키(source_file, page, chunk_index)로 같은 청크 점수를 합산한다.
    먼저 등장한 문서(보통 dense)를 대표로 사용.
    """
    if k is None:
        k = RRF_K
    score, rep = {}, {}
    for lst in result_lists:
        for rank, d in enumerate(lst):
            key = chunk_key(d)
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
            rep.setdefault(key, d)
    ordered = sorted(score, key=score.get, reverse=True)
    return [rep[key] for key in ordered]
