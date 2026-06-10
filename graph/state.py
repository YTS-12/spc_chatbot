from typing import TypedDict, List, Optional, Any


class GraphState(TypedDict, total=False):
    original_query: str
    current_query: str
    queries: List[str]
    retry_count: int
    retrieved_docs: List[Any]
    retrieved_context: str
    grade: str
    answer: str
