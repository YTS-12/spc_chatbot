import streamlit as st

from graph.config import OPENAI_API_KEY, PINECONE_API_KEY
from graph.workflow import build_graph, format_sources


st.set_page_config(
    page_title="K-IFRS SPC LangGraph Proto",
    page_icon="📚",
    layout="wide",
)

st.title("K-IFRS SPC LangGraph Proto")
st.caption("LangGraph + Pinecone + K-IFRS 기준서 기반 질의응답")


@st.cache_resource(show_spinner=False)
def get_app_graph():
    return build_graph()


if not OPENAI_API_KEY:
    st.warning(".env 파일에 OPENAI_API_KEY를 설정하세요.")

if not PINECONE_API_KEY:
    st.warning(".env 파일에 PINECONE_API_KEY를 설정하세요.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "질문을 입력하면 K-IFRS 기준서를 검색해 답변합니다.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 참고한 기준서"):
                st.markdown(msg["sources"])

user_input = st.chat_input("예: SPC를 연결해야 하는지 판단할 때 어떤 기준을 먼저 봐야 하나요?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("질문을 재작성하고 기준서를 검색하는 중입니다..."):
            sources = ""
            try:
                result = get_app_graph().invoke(
                    {
                        "original_query": user_input,
                        "retry_count": 0,
                    }
                )
                answer = result.get("answer", "답변을 생성하지 못했습니다.")
                sources = format_sources(result.get("retrieved_docs", []))
            except Exception as exc:
                answer = (
                    "앱 실행 중 오류가 발생했습니다.\n\n"
                    f"```text\n{type(exc).__name__}: {exc}\n```"
                )

            st.markdown(answer)
            if sources:
                with st.expander("📚 참고한 기준서"):
                    st.markdown(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
