import streamlit as st
import requests
import json
from dotenv import load_dotenv
import pandas as pd
from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.utils.helpers import format_chat_history, format_context_from_docs, create_system_prompt
from rag_func.constants.config import APP_CONFIG, OPENAI_API_KEY, GEMINI_API_KEY
from typing import Dict
import google.generativeai as genai

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

def process_query_with_rag(rag_system, user_input, memories, chat_history):
    relevant_docs = rag_system["retriever"].get_relevant_documents(user_input)
    docs = [doc for doc in relevant_docs if doc.page_content.strip()]

    reranked_docs = rag_system["reranker"].rerank(user_input, docs)
    context = format_context_from_docs(reranked_docs)

    prompt = create_system_prompt(user_input, chat_history, context, memories)
    response = rag_system["llm"].generate_response(prompt)

    return response, reranked_docs


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


def display_evaluation_results(evaluation_result):
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
            "Response Groundedness": round(scores_dict.get("nv_response_groundedness", 0), 3),
            "Context Relevance": round(scores_dict.get("nv_context_relevance", 0), 3),
        }
    )

def classify_query_with_openai_functions(user_input: str, chat_history: str) -> Dict:

    functions = [
        {
            "name": "classify_query",
            "description": "Classify the user query into greeting, health-related, or non-health categories",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["greeting", "health", "non_health"],
                        "description": "The type of query detected"
                    },
                    "greeting_text": {
                        "type": "string",
                        "description": "The greeting text identified in the user's message (for greeting type)"
                    },
                    "query_text": {
                        "type": "string",
                        "description": "The user's query text (for health and non-health types)"
                    },
                    "health_category": {
                        "type": "string",
                        "enum": ["remedy", "symptom", "herb", "wellness", "general_health", "other"],
                        "description": "The category of health query (for health type)"
                    },
                    "detected_category": {
                        "type": "string",
                        "enum": ["technology", "politics", "entertainment", "mathematics", "finance", "other"],
                        "description": "The detected category of the non-health query (for non-health type)"
                    }
                },
                "required": ["query_type"]
            }
        }
    ]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "system",
                "content": """
                You are an assistant that helps classify user queries for a health advice chatbot called "Grandma's Remedies".
                You must determine if the query is:
                1. A simple greeting
                2. A health-related question that should be answered
                3. A non-health question that should be politely declined

                For greetings, set query_type to "greeting" and include the greeting_text.

                For health queries, set query_type to "health", include the query_text, and specify the health_category.
                Health categories include: "remedy", "symptom", "herb", "wellness", "general_health", or "other"

                For non-health queries, set query_type to "non_health", include the query_text, and specify the detected_category.
                Non-health categories include: "technology", "politics", "entertainment", "mathematics", "finance", or "other"

                Examples of health queries:
                - "What can I do for a headache?"
                - "Is turmeric good for inflammation?"
                - "Best home remedy for sore throat?"

                Examples of non-health queries:
                - "How does a computer work?"
                - "What's the capital of France?"
                - "Can you solve this math problem?"

                Examples of greetings:
                - "Hello"
                - "Hi there"
                - "Namaste"
                """
            },
            {"role": "user", "content": f"Query: {user_input}\nChat History: {chat_history}"}
        ],
        "functions": functions,
        "function_call": {"name": "classify_query"},
        "temperature": 0.0,
        "max_tokens": 1024
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )

        response.raise_for_status()
        response_data = response.json()

        function_call = response_data["choices"][0]["message"].get("function_call")

        if not function_call:
            return {
                "name": "handle_health_query",
                "input": {
                    "query_text": user_input,
                    "health_category": "general_health"
                }
            }

        args = json.loads(function_call["arguments"])
        query_type = args.get("query_type")

        if query_type == "greeting":
            return {
                "name": "handle_greeting",
                "input": {
                    "greeting_text": args.get("greeting_text", "")
                }
            }
        elif query_type == "non_health":
            return {
                "name": "handle_non_health_query",
                "input": {
                    "query_text": args.get("query_text", user_input),
                    "detected_category": args.get("detected_category", "other")
                }
            }
        else:
            return {
                "name": "handle_health_query",
                "input": {
                    "query_text": args.get("query_text", user_input),
                    "health_category": args.get("health_category", "general_health")
                }
            }

    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return {
            "name": "handle_health_query",
            "input": {
                "query_text": user_input,
                "health_category": "general_health"
            }
        }

def handle_greeting(input_data):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    greeting_text = input_data.get("greeting_text", "")

    prompt = f"""
    You are Grandma, a warm and affectionate elder who shares traditional wisdom and natural remedies.
    A user has greeted you by saying: "{greeting_text}"
    Respond in your loving Grandma style, welcoming the user and offering help with traditional remedies. Keep it heartwarming.
    """

    response = model.generate_content(prompt)
    return response.text.strip(), []

def handle_non_health_query(input_data):
    query_text = input_data.get("query_text", "")

    response = (
        f"Oh dear, your question about matters outside of health and wellness is a bit outside my expertise. "
        "Grandma's always happy to help with your aches, sniffles, and remedies passed down "
        "through the years. But when it comes to things outside of health, "
        "I'm afraid this old mind doesn't stretch quite that far! Now, if you've got a health "
        "worry or a home remedy question, come sit beside me and ask away."
    )
    return response, []

def handle_health_query(rag_system, input_data, memories, chat_history):
    query_text = input_data.get("query_text", "")

    memories.append(query_text)

    return process_query_with_rag(
        rag_system,
        query_text,
        memories,
        chat_history
    )

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

        with st.spinner("Grandma is thinking..."):
            tool_call = classify_query_with_openai_functions(user_input, chat_history)

            function_name = tool_call.get("name")
            input_data = tool_call.get("input", {})

            if function_name == "handle_greeting":
                response, reranked_docs = handle_greeting(input_data)
                should_evaluate = False
            elif function_name == "handle_non_health_query":
                response, reranked_docs = handle_non_health_query(input_data)
                should_evaluate = False
            else:
                response, reranked_docs = handle_health_query(
                    rag_system,
                    input_data,
                    st.session_state.memories,
                    chat_history
                )
                should_evaluate = bool(reranked_docs)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        if should_evaluate and reranked_docs:
            with st.spinner("Evaluating response quality..."):
                evaluation_result = evaluate_response(
                    rag_system=rag_system,
                    reranked_docs=reranked_docs,
                    response=response,
                    user_input=user_input
                )
                display_evaluation_results(evaluation_result)

        st.markdown("*Grandma's secrets, unlocked by Sahaja.*")

if __name__ == "__main__":
    main()