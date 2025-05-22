import streamlit as st
import requests
from dotenv import load_dotenv
import pandas as pd
from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.utils.helpers import format_chat_history, format_context_from_docs, create_system_prompt
from rag_func.constants.config import APP_CONFIG, user_greetings, CLAUDE_API_KEY

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
    # llm = get_llm_model()
    evaluator = get_evaluator()

    return {
        "docs": docs,
        "retriever": retriever,
        "reranker": reranker,
        # "llm": llm,
        "evaluator": evaluator
    }

def process_query_with_rag(rag_system, user_input, memories, chat_history):
    relevant_docs = rag_system["retriever"].get_relevant_documents(user_input)
    docs = [doc for doc in relevant_docs if doc.page_content.strip()]

    reranked_docs = rag_system["reranker"].rerank(user_input, docs)
    context = format_context_from_docs(reranked_docs)

    prompt = create_system_prompt(user_input, chat_history, context, memories)


    return prompt, reranked_docs


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
    ground_truth = get_ground_truth_for_question(user_input)

    if "Ground truth not found" in ground_truth:
        return {"note": "Ground truth not found; evaluation skipped."}

    evaluation_result = rag_system["evaluator"].evaluate(
        user_input, response, context_docs
    )
    return evaluation_result


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_retrieval_permission(user_input, chat_history):
    prompt = (
        "You are a strict classifier trained to identify whether a user input is related to health topics. "
        "Your task is to analyze the input and determine if it pertains to health, medicine, wellness, medical conditions, "
        "home remedies, any health-related concerns or something from chat history. "
        "Respond with ONLY one word: "
        "- Respond 'Yes' if the input is related to any health or wellness topic. "
        "- Respond 'No' if it is unrelated to health. "
        "Do NOT provide any explanations, context, punctuation, or additional words. "
        f"Input: {user_input}"
        f"Chat History: {chat_history}"
    )

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096
    }

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"Error from Claude API: {response.text}")

    return response.json()["content"][0]["text"].strip()

def generate_llm_response(prompt):
    from openai import OpenAI
    from openai.types.responses import ResponseTextDeltaEvent

    client = OpenAI()
    messages = [{"role": "user", "content": prompt}]

    response = client.responses.create(model="gpt-4.1", input=messages, stream=True)

    for event in response:
        if isinstance(event, ResponseTextDeltaEvent):
            yield event.delta

def main():
    st.set_page_config(page_title=APP_CONFIG["title"], page_icon=APP_CONFIG["page_icon"], layout="wide")
    st.title(APP_CONFIG["title"])
    rag_system = initialize_rag_system()

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

        chat_history = get_chat_history()
        retrieval = get_retrieval_permission(user_input, chat_history)

        if user_input.lower() in user_greetings:
            full_response = (
                "Namaste, beta! How wonderful to hear from you. What can Grandma help you with today? "
                "I have so many ancient remedies passed down through generations, "
                "I'm sure we can find something to soothe your woes!"
            )
            reranked_docs = []

        elif retrieval == "No":
            full_response = ("Oh dear, Grandma's always happy to help with your aches, sniffles, and remedies passed down "
                        "through the years. But when it comes to things outside of health—like rockets, robots, or"
                        " riddles—I'm afraid this old mind doesn't stretch quite that far! Now, if you've got a health "
                        "worry or a home remedy question, come sit beside me and ask away.")
            reranked_docs = []
        else:
            st.session_state.memories.append(user_input)

            prompt, reranked_docs = process_query_with_rag(
                rag_system,
                user_input,
                st.session_state.memories,
                chat_history
            )
            full_response = ""

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                for chunk in generate_llm_response(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        if reranked_docs:
            with st.spinner("Evaluating response quality..."):
                evaluation_result = evaluate_response(
                    rag_system=rag_system,
                    reranked_docs=reranked_docs,
                    response=full_response,
                    user_input=user_input
                )

                scores = evaluation_result.scores[0]

                st.subheader("Evaluation Metrics")
                st.write(
                    {
                        "Faithfulness": round(scores["faithfulness"], 3),
                        "Answer Relevancy": round(scores["answer_relevancy"], 3),
                        "Response Groundedness": round(scores["nv_response_groundedness"], 3),
                        "Context Relevance": round(scores["nv_context_relevance"], 3),
                    }
                )
        st.markdown("*Grandma's secrets, unlocked by Sahaja.*")

if __name__ == "__main__":
    main()
