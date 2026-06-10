"""검색 결과 재정렬(rerank).

검색 단계에서 후보를 넉넉히(RERANK_FETCH_K) 가져온 뒤, 질문-문서 관련도를
cross-encoder(로컬) 또는 Cohere(API)로 다시 매겨 상위 TOP_K개만 남긴다.
MMR(다양성) 다음에 적용되어, 정밀도(관련도 순위)를 끌어올린다.

활성화: .env 에서 RERANK_ENABLED=true 로 설정하고 백엔드를 준비한다.
  - local : pip install transformers  (torch 필요; sentence-transformers/datasets 불필요)
            예) RERANK_MODEL=BAAI/bge-reranker-v2-m3 (다국어, 한국어 양호; 최초 1회 모델 다운로드)
  - cohere: pip install cohere + .env 에 COHERE_API_KEY
            예) RERANK_MODEL=rerank-multilingual-v3.0

백엔드가 준비되지 않았으면 경고만 출력하고 rerank 없이 통과한다(앱은 정상 동작).
"""

from graph.config import (
    COHERE_API_KEY,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_PROVIDER,
)

_rerank_fn = None
_initialized = False


def _build_local(model_name):
    """로컬 cross-encoder(transformers) 기반 reranker. bge-reranker 계열 호환.

    sentence-transformers/datasets 의존 없이 transformers + torch만 사용한다.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    def _rerank(query, docs, top_n):
        if not docs:
            return docs
        pairs = [[query, d.page_content] for d in docs]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = model(**inputs, return_dict=True).logits.view(-1).float()
        order = scores.argsort(descending=True).tolist()
        return [docs[i] for i in order[:top_n]]

    return _rerank


def _build_cohere(model_name):
    """Cohere Rerank API 기반 reranker."""
    import cohere

    client = cohere.Client(COHERE_API_KEY)

    def _rerank(query, docs, top_n):
        if not docs:
            return docs
        texts = [d.page_content for d in docs]
        resp = client.rerank(
            query=query, documents=texts, top_n=top_n, model=model_name
        )
        return [docs[r.index] for r in resp.results]

    return _rerank


def get_reranker():
    """rerank 콜러블 (query, docs, top_n)->docs 를 반환. 비활성/백엔드 없음이면 None."""
    global _rerank_fn, _initialized
    if _initialized:
        return _rerank_fn
    _initialized = True

    if not RERANK_ENABLED:
        _rerank_fn = None
        return None

    try:
        if RERANK_PROVIDER == "cohere":
            if not COHERE_API_KEY:
                raise RuntimeError("COHERE_API_KEY가 .env에 없습니다.")
            _rerank_fn = _build_cohere(RERANK_MODEL)
        else:
            _rerank_fn = _build_local(RERANK_MODEL)
        print(f"[rerank] 활성화: provider={RERANK_PROVIDER}, model={RERANK_MODEL}")
    except Exception as exc:
        print(f"[rerank] 백엔드 준비 실패 -> rerank 없이 진행: {exc}")
        _rerank_fn = None

    return _rerank_fn
