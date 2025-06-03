import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.utils.helpers import format_chat_history
from rag_func.constants.config import APP_CONFIG, CLAUDE_API_KEY, OPENAI_API_KEY
from rag_func.core.function_calling import RAGAssistantWithFunctions

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

def get_chat_history(chat_history=None):
    if chat_history is None:
        try:
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'messages'):
                chat_history = format_chat_history(st.session_state.messages[:-1])
            else:
                chat_history = ""
        except:
            chat_history = ""
    return chat_history

def evaluate_response(rag_system, reranked_docs, user_input, response):
    context_docs = [doc.page_content for doc in reranked_docs]
    expected_output = "Triphala works by acting as a mild laxative, stimulating bowel movements, and supporting digestion with its anti-inflammatory and antioxidant properties."

    evaluation_result = rag_system["evaluator"].evaluate(
        question=user_input, answer=response, retrieved_context=context_docs, expected_output=expected_output
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

    assistant = RAGAssistantWithFunctions(openai_api_key=OPENAI_API_KEY, rag_system=rag_system)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memories" not in st.session_state:
        st.session_state.memories = []

    display_chat_history()

    user_input = st.chat_input("Tell Grandma your problem...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            response, reranked_docs = assistant.classify_and_handle_query(
                user_input=user_input,
                memories=st.session_state.memories,
                chat_history=st.session_state.messages
            )

        st.session_state.messages.append({"role": "assistant", "content": response})

        if reranked_docs:
            with st.spinner("Evaluating response quality..."):
                evaluation_result = evaluate_response(
                    rag_system=rag_system,
                    reranked_docs=reranked_docs,
                    response=response,
                    user_input=user_input
                )

                if "note" not in evaluation_result:
                    test_result = evaluation_result.test_results[0]
                    metrics_data = test_result.metrics_data

                    scores_dict = {}
                    for metric in metrics_data:
                        key = metric.name.lower().replace(' ', '_')
                        scores_dict[key] = metric.score

                    st.subheader("Evaluation Metrics")
                    st.write(
                        {
                            "Faithfulness": round(scores_dict.get("faithfulness", 0), 3),
                            "Answer Relevancy": round(scores_dict.get("answer_relevancy", 0), 3),
                            "Context Recall": round(scores_dict.get("contextual_recall", 0), 3),
                            "Context Relevance": round(scores_dict.get("contextual_relevancy", 0), 3),
                        }
                    )

        st.markdown("*Grandma's secrets, unlocked by Sahaja.*")


if __name__ == "__main__":
    main()
