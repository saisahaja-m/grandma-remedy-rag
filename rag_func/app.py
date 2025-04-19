import streamlit as st
from dotenv import load_dotenv

from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.utils.helpers import format_chat_history, format_context_from_docs, create_system_prompt
from rag_func.config.config import APP_CONFIG

load_dotenv()


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components"""
    docs = load_and_process_documents()
    retriever = get_retriever(docs)
    llm = get_llm_model()
    evaluator = get_evaluator()
    reranker = get_reranker()

    return {
        "docs": docs,
        "retriever": retriever,
        "llm": llm,
        "evaluator": evaluator,
        "reranker": reranker
    }

def main():
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon=APP_CONFIG["page_icon"],
        layout="wide"
    )

    st.title(APP_CONFIG["title"])

    rag_system = initialize_rag_system()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    user_input = st.chat_input("Tell Grandma your problem...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        relevant_docs = rag_system["retriever"].get_relevant_documents(user_input)
        reranked_docs = rag_system["reranker"].rerank(user_input, relevant_docs)
        context = format_context_from_docs(reranked_docs)
        chat_history = format_chat_history(st.session_state.messages[:-1])
        prompt = create_system_prompt(user_input, chat_history, context)
        response = rag_system["llm"].generate_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


        with st.spinner("Evaluating response quality..."):
            context_docs = [doc.page_content for doc in reranked_docs]
            result = rag_system["evaluator"].evaluate(user_input, response, context_docs)
            st.write("Evaluation result:", result)

        st.markdown("*Grandma's secrets, unlocked by Sahaja.*")

if __name__ == "__main__":
    main()