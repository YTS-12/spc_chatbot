import re
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from graph.config import (
    ALLOWED_SECTION_TYPES,
    BASIS_RESERVE,
    CHAT_MODEL,
    HYBRID_ENABLED,
    MAX_QUERIES,
    MAX_RETRY,
    MULTIQUERY_ENABLED,
    ONTOLOGY_FILTER_ENABLED,
    ONTOLOGY_GROUNDING,
    RETRY_TEMPERATURE,
    SECTION_FILTER_ENABLED,
    TOP_K,
)
from graph.hybrid import available as hybrid_available, bm25_search, rrf_fuse
from graph.ontology import match_concepts
from graph.prompts import (
    FALLBACK_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    PLAN_PROMPT,
    REWRITE_PROMPT,
    REWRITE_RETRY_PROMPT,
)
from graph.rerank import get_reranker
from graph.retrieval import base_k, get_vectorstore, search
from graph.state import GraphState


_SECTION_LABEL = {
    "basis": "결론도출근거",
    "minority": "소수의견",
    "example": "적용사례",
    "toc": "목차",
}


def format_docs(docs):
    chunks = []
    for doc in docs:
        meta = doc.metadata
        standard_no = meta.get("standard_no", "unknown")
        standard_name = meta.get("standard_name", "")
        source_file = meta.get("source_file", meta.get("source", "unknown"))

        # 실제 표기 페이지(1-base) 우선. 없으면 0-base page에 +1 보정.
        page = meta.get("page_label")
        if page is None:
            raw_page = meta.get("page")
            page = raw_page + 1 if isinstance(raw_page, int) else "unknown"

        paragraph_no = meta.get("paragraph_no")

        header = f"[기준서: {standard_no}"
        if standard_name:
            header += f"({standard_name})"
        header += f" | 파일: {source_file} | page: {page}"
        if paragraph_no:
            header += f" | 문단: {paragraph_no}"
        label = _SECTION_LABEL.get(meta.get("section_type"))
        if label:
            header += f" | {label}"
        header += "]"

        chunks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(chunks)


def format_sources(docs):
    """retrieved_docs 메타데이터로 '참고한 기준서' 출처 목록(markdown) 생성.

    LLM 답변과 무관하게, 실제 검색에 사용된 출처를 보장 표시한다(검증용).
    (기준서, 페이지, 문단) 기준으로 중복 제거.
    """
    lines, seen = [], set()
    for doc in docs:
        m = doc.metadata
        standard_no = m.get("standard_no", "unknown")
        standard_name = m.get("standard_name", "")

        page = m.get("page_label")
        if page is None:
            raw_page = m.get("page")
            page = raw_page + 1 if isinstance(raw_page, int) else "?"
        paragraph_no = m.get("paragraph_no")
        section = _SECTION_LABEL.get(m.get("section_type"), "본문")

        key = (standard_no, str(page), paragraph_no)
        if key in seen:
            continue
        seen.add(key)

        title = standard_no + (f"({standard_name})" if standard_name else "")
        loc = f"p.{page}" + (f" 문단 {paragraph_no}" if paragraph_no else "")
        lines.append(f"- **{title}** · {loc} · _{section}_")
    return "\n".join(lines)


def build_graph():
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    # 재시도 재작성 전용(변형을 위해 temperature 상향). grade/generate는 결정적 유지.
    llm_retry = ChatOpenAI(model=CHAT_MODEL, temperature=RETRY_TEMPERATURE)
    vectorstore = get_vectorstore()
    reranker = get_reranker()  # RERANK_ENABLED=false면 None(동작 변화 없음)

    def build_filters(standards, sections):
        """가장 구체적인 필터(섹션∧기준서)부터 점진 완화한 후보 목록(폴백용)."""
        sec = (
            {"section_type": {"$in": sections}}
            if (SECTION_FILTER_ENABLED and sections)
            else None
        )
        std = (
            {"standard_no": {"$in": standards}}
            if (ONTOLOGY_FILTER_ENABLED and standards)
            else None
        )
        candidates = []
        if sec and std:
            candidates.append({"$and": [sec, std]})
        if sec:
            candidates.append(sec)
        if std:
            candidates.append(std)
        candidates.append(None)
        out, seen = [], set()
        for c in candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def ground(queries, onto):
        # 각 쿼리에 정식 개념어 + 기준서 번호를 부착(검색 타겟 강화).
        if not ONTOLOGY_GROUNDING:
            return queries
        suffix = " ".join(onto["terms"] + onto["standards"]).strip()
        return [f"{q} {suffix}".strip() for q in queries] if suffix else queries

    def rewrite_node(state: GraphState) -> GraphState:
        query = state["original_query"]
        onto = match_concepts(query)

        queries = []
        if MULTIQUERY_ENABLED:
            # LLM이 스스로 판단: 단순 질문은 1개, 실무·복합 질문은 이론+실무로 분해.
            prompt = PLAN_PROMPT.format(
                query=query,
                max_q=MAX_QUERIES,
                theory="; ".join(onto["theory"]) or "(없음)",
                practice="; ".join(onto["practice"]) or "(없음)",
            )
            raw = llm.invoke(prompt).content.strip()
            for line in raw.splitlines():
                q = re.sub(r"^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?", "", line).strip()
                if len(q) > 3:
                    queries.append(q)
            queries = queries[:MAX_QUERIES]

        if not queries:
            queries = [query]

        queries = ground(queries, onto)
        return {
            "queries": queries,
            "current_query": queries[0],
            "retry_count": 0,
        }

    def rewrite_retry_node(state: GraphState) -> GraphState:
        original = state["original_query"]
        previous = state.get("current_query", original)
        retry_count = state.get("retry_count", 0) + 1

        # 직전 실패 질의를 참고해 '다른 각도'로 재작성(이전엔 동일 질의만 반복했음).
        prompt = REWRITE_RETRY_PROMPT.format(original=original, previous=previous)
        rewritten = llm_retry.invoke(prompt).content.strip()

        queries = ground([rewritten], match_concepts(original))
        return {
            "queries": queries,
            "current_query": queries[0],
            "retry_count": retry_count,
        }

    def retrieve_one(query, standards, sections):
        # dense(Pinecone): 섹션∧온톨로지 필터, 부족하면 점진 완화.
        dense = []
        for flt in build_filters(standards, sections):
            dense = search(vectorstore, query, flt)
            if len(dense) >= TOP_K:
                break
        # sparse(BM25): 번호/정확 용어 보완 후 RRF 융합(없으면 dense-only).
        if HYBRID_ENABLED and hybrid_available():
            sparse = bm25_search(query, base_k(), sections, standards or None)
            return rrf_fuse([dense, sparse]) if sparse else dense
        return dense

    def rank(cands, n, query):
        if reranker is not None and cands:
            return reranker(query, cands, n)
        return cands[:n]

    def retrieve_node(state: GraphState) -> GraphState:
        queries = state.get("queries") or [state["current_query"]]
        original = state["original_query"]
        onto = match_concepts(original)
        standards = onto["standards"]
        sections = list(ALLOWED_SECTION_TYPES) if SECTION_FILTER_ENABLED else None

        # BC 예약석: include_basis 개념(SPC 등)은 basis를 따로 검색해 일부 슬롯을 보장.
        basis_docs = []
        if SECTION_FILTER_ENABLED and onto.get("include_basis") and BASIS_RESERVE > 0:
            reserve = min(BASIS_RESERVE, TOP_K - 1)
            basis_fused = rrf_fuse([retrieve_one(q, standards, ["basis"]) for q in queries])
            basis_docs = rank(basis_fused[: max(base_k(), reserve)], reserve, original)

        # 본문/적용사례 풀에서 나머지 슬롯 채움(rerank는 '원 질문' 기준).
        main_fused = rrf_fuse([retrieve_one(q, standards, sections) for q in queries])
        main_docs = rank(main_fused[: max(base_k(), TOP_K)], TOP_K - len(basis_docs), original)

        docs = main_docs + basis_docs
        context = format_docs(docs)
        return {
            "retrieved_docs": docs,
            "retrieved_context": context,
        }

    def grade_node(state: GraphState) -> GraphState:
        query = state["original_query"]
        context = state.get("retrieved_context", "")

        prompt = GRADE_PROMPT.format(query=query, context=context)
        result = llm.invoke(prompt).content.strip().upper()

        grade = "good" if "GOOD" in result else "bad"
        return {"grade": grade}

    def generate_node(state: GraphState) -> GraphState:
        query = state["original_query"]
        context = state.get("retrieved_context", "")

        prompt = GENERATE_PROMPT.format(query=query, context=context)
        answer = llm.invoke(prompt).content.strip()

        return {"answer": answer}

    def fallback_node(state: GraphState) -> GraphState:
        query = state["original_query"]

        prompt = FALLBACK_PROMPT.format(query=query)
        answer = llm.invoke(prompt).content.strip()

        return {"answer": answer}

    def route_after_grade(
        state: GraphState,
    ) -> Literal["generate", "rewrite_retry", "fallback"]:
        grade = state.get("grade", "bad")
        retry_count = state.get("retry_count", 0)

        if grade == "good":
            return "generate"

        if grade == "bad" and retry_count < MAX_RETRY:
            return "rewrite_retry"

        return "fallback"

    graph = StateGraph(GraphState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("rewrite_retry", rewrite_retry_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("rewrite_retry", "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        ["generate", "rewrite_retry", "fallback"],
    )

    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()
