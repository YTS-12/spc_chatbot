from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

DATA_DIR = BASE_DIR / "data" / "K-IFRS" / "Proto"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "kifrs-proto-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "kifrs-proto-v1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))

# --- 검색(Retrieval) — MMR(다양성 확보) ---
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "20"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))

# --- 재시도(rewrite_retry) 설정 ---
# 재시도 재작성은 변형을 위해 temperature를 올린다(원본 grade/generate는 0 유지).
RETRY_TEMPERATURE = float(os.getenv("RETRY_TEMPERATURE", "0.7"))

# --- 인제스트 설정 (다음 재적재 시 반영) ---
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "50"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))

# --- Rerank 설정 (로컬 cross-encoder; 기본 비활성, RERANK_ENABLED=true로 활성화) ---
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_FETCH_K = int(os.getenv("RERANK_FETCH_K", "20"))  # rerank 대상 후보 수

# --- 섹션 타입 필터(1층) — 본문 우선, 결론도출근거(BC)/소수의견/목차 제외 ---
SECTION_FILTER_ENABLED = os.getenv("SECTION_FILTER_ENABLED", "true").lower() == "true"
# 허용 섹션 코드: body(본문) | example(적용사례) | basis(결론도출근거) | minority(소수의견) | toc(목차)
ALLOWED_SECTION_TYPES = [
    s.strip() for s in os.getenv("ALLOWED_SECTION_TYPES", "body,example").split(",") if s.strip()
]
# BC 예약석: include_basis 개념(예: SPC)일 때 결론도출근거(basis)에 보장할 슬롯 수.
BASIS_RESERVE = int(os.getenv("BASIS_RESERVE", "2"))

# --- 온톨로지(2층) — 개념->기준서 라우팅 및 쿼리 그라운딩 ---
ONTOLOGY_FILTER_ENABLED = os.getenv("ONTOLOGY_FILTER_ENABLED", "true").lower() == "true"
ONTOLOGY_GROUNDING = os.getenv("ONTOLOGY_GROUNDING", "true").lower() == "true"

# 라우팅 정밀도: 흔한 단어 무시 + 매칭 점수화로 약한 우연 매칭 제거(여유 캡).
ROUTING_MAX_CONCEPTS = int(os.getenv("ROUTING_MAX_CONCEPTS", "5"))
# 최종 라우팅 기준서 수 상한(점수 높은 개념부터 채움). 0이면 무제한.
ROUTING_MAX_STANDARDS = int(os.getenv("ROUTING_MAX_STANDARDS", "4"))

# --- 하이브리드 검색(3순위) — dense(Pinecone) + sparse(BM25) 융합(RRF) ---
# 번호/정확한 용어(예: "문단 12", "제1113호 수준 2") 매칭을 보완한다.
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() == "true"
RRF_K = int(os.getenv("RRF_K", "60"))
CHUNK_CACHE_PATH = BASE_DIR / "chunks_cache.jsonl"

# --- 멀티쿼리(실무 질문 분해) — LLM이 질문을 이론·실무 하위쿼리로 분해 ---
MULTIQUERY_ENABLED = os.getenv("MULTIQUERY_ENABLED", "true").lower() == "true"
MAX_QUERIES = int(os.getenv("MAX_QUERIES", "4"))
