import os
import streamlit as st
from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.constants.config import APP_CONFIG
from rag_func.core.function_calling import RAGAssistantWithFunctions
from rag_func.core.embedding import get_embedding_model
from rag_func.core.vector_stores import create_vector_store
from dotenv import load_dotenv
load_dotenv()

def initialize_data_processing():
    docs = load_and_process_documents()
    embeddings = get_embedding_model()
    vector_store = create_vector_store(docs, embeddings)

    return vector_store

def initialize_rag_components():
    embeddings = get_embedding_model()
    retriever = get_retriever()
    reranker = get_reranker()
    llm = get_llm_model()
    evaluator = get_evaluator()
    return {
        "embeddings": embeddings,
        "retriever": retriever,
        "reranker": reranker,
        "llm": llm,
        "evaluator": evaluator
    }

def evaluate_response(rag_components, reranked_docs, user_input, response):
    context_docs = [doc.page_content for doc in reranked_docs]
    expected_output = ""
    evaluation_result = rag_components["evaluator"].evaluate(
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

    initialize = True
    if initialize:
        initialize_data_processing()

    rag_system = initialize_rag_components()
    assistant = RAGAssistantWithFunctions(openai_api_key=os.getenv("OPENAI_API_KEY"), rag_system=rag_system)

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
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # if reranked_docs:
        #     with st.spinner("Evaluating response quality..."):
        #         evaluation_result = evaluate_response(
        #             rag_system=rag_system,
        #             reranked_docs=reranked_docs,
        #             response=response,
        #             user_input=user_input
        #         )
        #
        #         if "note" not in evaluation_result:
        #             test_result = evaluation_result.test_results[0]
        #             metrics_data = test_result.metrics_data
        #
        #             scores_dict = {}
        #             for metric in metrics_data:
        #                 key = metric.name.lower().replace(' ', '_')
        #                 scores_dict[key] = metric.score
        #
        #             st.subheader("Evaluation Metrics")
        #             st.write(
        #                 {
        #                     "Faithfulness": round(scores_dict.get("faithfulness", 0), 3),
        #                     "Answer Relevancy": round(scores_dict.get("answer_relevancy", 0), 3),
        #                     "Context Recall": round(scores_dict.get("contextual_recall", 0), 3),
        #                     "Context Relevance": round(scores_dict.get("contextual_relevancy", 0), 3),
        #                 }
        #             )

        st.markdown("*Grandma's secrets, unlocked by Sahaja.*")

if __name__ == "__main__":
    main()
