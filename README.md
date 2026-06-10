# K-IFRS SPC 회계 질의응답 챗봇 (Proto)

K-IFRS(한국채택국제회계기준) 기준서 PDF를 검색 근거로 답변하는 RAG 챗봇.
**SPC(특수목적기업) 연결 판단**에 관련된 12개 기준서에 초점을 둔 프로토타입.

스택: `Streamlit + LangGraph + Pinecone + OpenAI(Embeddings/Chat)`

---

## 아키텍처

```
[인제스트] PDF 12종 -> 클린업/청크 -> OpenAI 임베딩 -> Pinecone(kifrs-proto-v1)

[질의] START
  -> rewrite        질문을 검색용 질의로 재작성
  -> retrieve       Pinecone MMR 검색(top-k)
  -> grade          검색결과 GOOD/BAD 채점
       GOOD                 -> generate(4단계 형식 답변) -> END
       BAD & retry<MAX      -> rewrite_retry(다른 각도로 재작성) -> retrieve
       BAD & retry>=MAX     -> fallback(정보부족 안내) -> END
```

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit 채팅 UI |
| `graph/config.py` | `.env` 로딩 + 런타임 상수(모델/청킹/검색/재시도) |
| `graph/state.py` | LangGraph 상태(TypedDict) |
| `graph/prompts.py` | rewrite / rewrite_retry / grade / generate / fallback 프롬프트 |
| `graph/retrieval.py` | Pinecone 연결 + retriever(MMR/필터) |
| `graph/workflow.py` | LangGraph 워크플로우 + 인용 포맷 |
| `ingest_pinecone.py` | **정본** 인제스트(클린업·결정적 id·슬림 메타데이터) |
| `ingest_pinecone.ipynb` | 최초 적재에 사용된 노트북(기록용 — 운영 기준은 `.py`) |

---

## 실행

```powershell
# 1) 패키지
& 'C:\Users\Admin\miniconda3\envs\langchain_rag_env\python.exe' -m pip install -r requirements.txt

# 2) 앱 실행(현재 인덱스 그대로 사용 — 재적재 불필요)
& 'C:\Users\Admin\miniconda3\envs\langchain_rag_env\python.exe' -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503
```

`.env`는 `.env.example`을 복사해 값을 채운다.

---

## 개선 적용 현황 (2026-06-07)

재임베딩 없이 **즉시 반영**되는 쿼리층 개선과, **다음 재적재 때 반영**되는 데이터층 개선을 분리해 적용함.

### 즉시 반영(현재 라이브에 적용됨)
- **MMR 검색**: 인접 중복 청크가 상위를 독점하던 문제 완화 (`SEARCH_TYPE=mmr`)
- **retry 실질화**: 재시도 시 직전 실패 질의를 참고해 *다른 각도*로 재작성(temperature 상향)
- **인용 페이지 보정**: 0-base `page` 대신 실제 표기 페이지 `page_label` 사용(±1 오차 해소)
- **grade 완화**: 부분적으로라도 근거가 있으면 GOOD — 불필요한 fallback 감소
- **메타데이터 필터 옵션**: `USE_METADATA_FILTER`(기본 off)

### 다음 재적재 시 반영(코드 완비, 미실행)
- **추출 텍스트 클린업**: 한컴 PDF 공백 소실 복원 + 페이지번호/푸터 제거
- **결정적 id**: 재적재해도 중복 누적 없이 덮어쓰기
- **슬림 메타데이터**: 불필요한 PDF 메타 제거, `standard_name`/`page_label`/`chunk_index` 추가
- **저가치 청크 제거**, 배치 업서트 + 진행 로그

> ⚠️ `EMBEDDING_MODEL`을 바꾸면(예: 3-large) 차원이 달라져 기존 인덱스와 충돌한다.
> 반드시 재임베딩 + 새 namespace(또는 `ingest_pinecone.py --reset`)로 다시 만들 것.

---

## 재적재(원할 때만)

```powershell
# 결정적 id라 동일 청크는 덮어씀(추가 적재)
& '...\python.exe' ingest_pinecone.py

# namespace 비우고 처음부터(모델/청킹/클린업 변경 후 권장)
& '...\python.exe' ingest_pinecone.py --reset
```

---

## 주요 튜닝 노브 (`.env`)

| 키 | 기본 | 설명 |
|----|------|------|
| `TOP_K` | 6 | 검색 결과 수 |
| `SEARCH_TYPE` | mmr | `mmr` / `similarity` / `similarity_score_threshold` |
| `MMR_FETCH_K` / `MMR_LAMBDA` | 20 / 0.5 | MMR 후보 수 / 다양성 가중(낮을수록 다양) |
| `MAX_RETRY` | 2 | 재검색 최대 횟수 |
| `RETRY_TEMPERATURE` | 0.7 | 재시도 재작성 다양성 |
| `MIN_CHUNK_CHARS` | 50 | 이보다 짧은 청크 제거(재적재 시) |
| `RERANK_ENABLED` | true | 검색 후보를 cross-encoder로 재정렬 |
| `RERANK_MODEL` | BAAI/bge-reranker-v2-m3 | 로컬 reranker 모델(다국어) |
| `RERANK_FETCH_K` | 20 | rerank 대상 후보 수 → TOP_K로 추림 |
| `SECTION_FILTER_ENABLED` | true | 본문 우선(BC/소수의견/목차 제외) |
| `ALLOWED_SECTION_TYPES` | body,example | 허용 섹션 코드 |
| `BASIS_RESERVE` | 2 | `include_basis` 개념의 BC 예약 슬롯 수 |
| `ONTOLOGY_FILTER_ENABLED` / `ONTOLOGY_GROUNDING` | true | 개념→기준서 라우팅 / 쿼리 그라운딩 |
| `ROUTING_STRICT` / `ROUTING_MAX_STANDARDS` | true / 4 | 라우팅 정밀화(흔한 단어 무시 + 기준서 상한) |
| `HYBRID_ENABLED` | true | dense(의미) + BM25(키워드) 융합 검색 |

### Rerank (정밀도 향상)

검색(MMR)으로 가져온 `RERANK_FETCH_K`개 후보를 cross-encoder로 질문-문서 관련도를 다시
매겨 상위 `TOP_K`만 남긴다. 로컬 `transformers`(torch) 백엔드를 사용하며 API 키가 필요 없다.

```powershell
& '...\python.exe' -m pip install transformers   # torch 필요(최초 1회 모델 ~2.3GB 다운로드)
# .env: RERANK_ENABLED=true
```

> 환경 참고: 이 PC에서는 `pyarrow 23.x`의 Windows DLL 로드 실패로 `sklearn`/`transformers`
> import가 깨져 있었음 → `pip install "pyarrow==17.0.0"` 로 고정해 해결. 끄려면 `RERANK_ENABLED=false`.

### 섹션 필터(1층) + 온톨로지(2층)

검색이 분량 많은 **결론도출근거(BC)·소수의견·목차**를 끌어오던 문제를 두 축으로 해결한다.

- **1층 — 섹션 타입 태깅**: 적재 시 각 페이지를 `body / basis / minority / example / toc`로 분류해
  `section_type` 메타데이터로 저장([ingest_pinecone.py](ingest_pinecone.py)의 `classify_section`).
  검색은 기본적으로 `body,example`만 본다 → BC 노이즈 제거.
- **2층 — 경량 온톨로지**([graph/ontology.py](graph/ontology.py)): 개념(동의어)→기준서 매핑.
  - **쿼리 그라운딩**: 질문에서 개념 탐지 → 정식 용어·기준서 번호를 검색 질의에 주입.
  - **메타데이터 라우팅**: 매칭된 `standard_no`로 검색 범위를 좁힘.
  - 예) "SPC 연결" → 개념 [SPC·연결·지배력], 기준서 [제1027·1110·1112호].

검색은 `(섹션 ∧ 기준서)` 필터로 시작해, 결과가 `TOP_K` 미만이면 **자동으로 필터를 완화**(섹션만 →
기준서만 → 무필터)하므로 답변이 비는 일이 없다. 새 개념은 `ONTOLOGY` 사전에 한 줄 추가하면 된다.

### 하이브리드 검색 (dense + BM25)

의미 검색(dense)은 "지배력≈control"은 잘 찾지만 **"문단 12", "제1113호 수준 2"** 같은
정확한 번호/용어엔 약하다. 키워드 검색(**BM25**)을 더해 **RRF**(Reciprocal Rank Fusion)로 융합한다
([graph/hybrid.py](graph/hybrid.py)).

- BM25 코퍼스는 적재 시 생성되는 `chunks_cache.jsonl`(Pinecone와 동일 청크/ids)을 사용.
- `pip install rank_bm25` 필요. 캐시·라이브러리가 없으면 **자동으로 dense-only**로 동작(안전).
- 인용은 페이지에 더해 **문단 번호**(`paragraph_no`, best-effort 추출)도 표시.

### 멀티쿼리 & 온톨로지 확장

- **멀티쿼리**: 실무 질문은 LLM이 스스로 이론·실무 하위쿼리(최대 `MAX_QUERIES`개)로 분해해
  각각 검색 → RRF 통합한다([graph/workflow.py](graph/workflow.py)의 `rewrite_node` + `PLAN_PROMPT`).
- **온톨로지 이론·실무**: 각 개념에 `theory`(정의·요건)·`practice`(절차·판단·사례) 관점을 둔다.
- **대량 확장**: `python build_ontology.py` → 12개 기준서 본문을 `ONTOLOGY_GEN_PROMPT`로 돌려
  `graph/ontology_generated.json` 생성 → `ontology.py` 가 import 시 자동 병합(큐레이션 우선).
- **BC 예약석**: 보통 `basis`(결론도출근거)는 제외하지만, `include_basis: True` 개념(예: SPC)은
  BC를 본문 풀과 **별도로 검색해 결과 중 `BASIS_RESERVE`칸(기본 2)을 BC에 보장**한다
  (SIC-12·구조화기업·대리관계 등 배경 포착). 인용에 `결론도출근거` 라벨로 표시된다.
