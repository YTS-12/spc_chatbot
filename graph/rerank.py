"""검색 결과 재정렬(rerank) — 로컬 cross-encoder.

검색 단계에서 후보를 넉넉히(RERANK_FETCH_K) 가져온 뒤, 질문-문서 관련도를
cross-encoder로 다시 매겨 상위 TOP_K개만 남긴다. MMR(다양성) 다음에 적용되어
정밀도(관련도 순위)를 끌어올린다.

활성화: .env 에서 RERANK_ENABLED=true. pip install transformers (torch 필요).
        예) RERANK_MODEL=BAAI/bge-reranker-v2-m3 (다국어, 한국어 양호; 최초 1회 모델 다운로드)
미설치/실패 시 경고만 출력하고 rerank 없이 통과한다(앱은 정상 동작).
"""

from graph.config import RERANK_ENABLED, RERANK_MODEL

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
        _rerank_fn = _build_local(RERANK_MODEL)
        print(f"[rerank] 활성화: model={RERANK_MODEL}")
    except Exception as exc:
        print(f"[rerank] 백엔드 준비 실패 -> rerank 없이 진행: {exc}")
        _rerank_fn = None

    return _rerank_fn
