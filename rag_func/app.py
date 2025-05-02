import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.utils.helpers import format_chat_history, format_context_from_docs, create_system_prompt
from rag_func.constants.config import APP_CONFIG, user_greetings

load_dotenv()

def get_ground_truth_for_question(question: str, filepath: str = "ground_truths.csv") -> str:
    try:
        df = pd.read_csv(filepath)
        match = df[df['question'].str.strip().str.lower() == question.strip().lower()]
        if not match.empty:
            return match.iloc[0]['answer']
        else:
            return "Ground truth not found for this question."
    except Exception as e:
        return f"Error retrieving ground truth: {e}"

@st.cache_resource
def initialize_rag_system():
    docs = load_and_process_documents()
    retriever = get_retriever(docs)
    reranker = get_reranker()
    llm = get_llm_model()
    evaluator = get_evaluator()

    return {
        "docs": docs,
        "retriever": retriever,
        "reranker": reranker,
        "llm": llm,
        "evaluator": evaluator
    }

def process_query_with_rag(rag_system, user_input, chat_history=None):
    relevant_docs = rag_system["retriever"].get_relevant_documents(user_input)
    docs = [doc for doc in relevant_docs if doc.page_content.strip()]
    if not docs:
        return "I'm sorry, I don't have specific information about that. Is there something else I can help with?", []

    reranked_docs = rag_system["reranker"].rerank(user_input, docs)
    context = format_context_from_docs(reranked_docs)

    if chat_history is None:
        try:
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'messages'):
                chat_history = format_chat_history(st.session_state.messages[:-1])
            else:
                chat_history = ""
        except:
            chat_history = ""

    prompt = create_system_prompt(user_input, chat_history, context)
    response = rag_system["llm"].generate_response(prompt)

    return response, reranked_docs


def evaluate_response(rag_system, reranked_docs, user_input, response):
    context_docs = [doc.page_content for doc in reranked_docs]
    ground_truth = get_ground_truth_for_question(user_input)

    if "Ground truth not found" in ground_truth:
        return {"note": "Ground truth not found; evaluation skipped."}

    evaluation_result = rag_system["evaluator"].evaluate(
        user_input, response, context_docs, [ground_truth]
    )
    return evaluation_result


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    st.set_page_config(page_title=APP_CONFIG["title"], page_icon=APP_CONFIG["page_icon"], layout="wide")
    st.title(APP_CONFIG["title"])

    rag_system = initialize_rag_system()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    display_chat_history()

    user_input = st.chat_input("Tell Grandma your problem...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if user_input in user_greetings:
            response = (
                "Namaste, beta! How wonderful to hear from you. What can Grandma help you with today? "
                "I have so many ancient remedies passed down through generations, "
                "I'm sure we can find something to soothe your woes!"
            )
            reranked_docs = []
        else:
            response, reranked_docs = process_query_with_rag(rag_system, user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        if reranked_docs:
            with st.spinner("Evaluating response quality..."):
                evaluation_result = evaluate_response(
                    rag_system=rag_system,
                    reranked_docs=reranked_docs,
                    response=response,
                    user_input=user_input
                )
                st.write("Evaluation result:", evaluation_result)

        st.markdown("*Grandma's secrets, unlocked by Sahaja.*")

if __name__ == "__main__":
    main()
