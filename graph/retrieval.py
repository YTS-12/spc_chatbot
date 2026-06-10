from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from graph.config import (
    EMBEDDING_MODEL,
    MMR_FETCH_K,
    MMR_LAMBDA,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RERANK_ENABLED,
    RERANK_FETCH_K,
    SCORE_THRESHOLD,
    SEARCH_TYPE,
    TOP_K,
)


def get_vectorstore():
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY를 불러오지 못했습니다. "
            ".env 파일 경로와 변수명을 확인하세요."
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )


def base_k():
    # rerank가 켜져 있으면 후보를 더 넉넉히(이후 rerank가 TOP_K로 추림).
    return RERANK_FETCH_K if RERANK_ENABLED else TOP_K


def search(vectorstore, query, search_filter=None):
    """SEARCH_TYPE에 맞춰 검색하고 문서 리스트 반환. per-query 메타데이터 필터 지원.

    워크플로에서 섹션/온톨로지 필터를 쿼리마다 다르게 적용하기 위해 사용한다.
    """
    k = base_k()
    if SEARCH_TYPE == "mmr":
        return vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=max(MMR_FETCH_K, k),
            lambda_mult=MMR_LAMBDA,
            filter=search_filter,
        )
    if SEARCH_TYPE == "similarity_score_threshold":
        pairs = vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=search_filter
        )
        return [doc for doc, score in pairs if score >= SCORE_THRESHOLD]
    return vectorstore.similarity_search(query, k=k, filter=search_filter)
