import openai
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.utils.helpers import format_chat_history, format_context_from_docs, create_system_prompt
from rag_func.constants.config import APP_CONFIG, OPENAI_API_KEY, GEMINI_API_KEY
from typing import Dict, Any
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from google.genai import types
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

genai.configure(api_key=GEMINI_API_KEY)

# Function Declarations
classify_query_function = FunctionDeclaration(
    name="classify_query",
    description="Classify user queries for health chatbot routing",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query_type": {
                "type": "STRING",
                "enum": ["greeting", "health", "non_health"],
                "description": "Type of user query"
            },
            "greeting_text": {
                "type": "STRING",
                "description": "Identified greeting text (if greeting)"
            },
            "query_text": {
                "type": "STRING",
                "description": "User's actual question content"
            },
            "health_category": {
                "type": "STRING",
                "enum": ["remedy", "symptom", "herb", "wellness", "general_health", "other"],
                "description": "Health category classification"
            },
            "detected_category": {
                "type": "STRING",
                "enum": ["technology", "politics", "entertainment", "mathematics", "finance", "other"],
                "description": "Category for non-health queries"
            }
        },
        "required": ["query_type"]  # Added required field
    }
)

# Create Tool remains the same
query_classification_tool = Tool(
    function_declarations=[classify_query_function]
)


def classify_query_with_gemini_functions(user_input: str, chat_history: str) -> Dict[str, Any]:
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            tools=[query_classification_tool]
        )

        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=None, allowed_function_names=["get_current_temperature"]
            )
        )

        config = types.GenerateContentConfig(
            temperature=0,
            tools=[query_classification_tool],
            tool_config=tool_config,
        )

        # Generate content with proper configuration
        response = client.model.generate_content(
            f"Query: {user_input}\nChat History: {chat_history}",
            generation_config=genai.types.GenerationConfig(
                config=config
            )
        )

        # Process response
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call'):
                    args = part.function_call.args
                    return _process_function_args(args, user_input)

        # Fallback to health query
        return {
            "name": "handle_health_query",
            "input": {"query_text": user_input}
        }

    except Exception as e:
        st.error(f"Gemini Error: {str(e)}")
        return {
            "name": "handle_health_query",
            "input": {"query_text": user_input}
        }


# Update the helper function
def _process_function_args(args: dict, user_input: str) -> dict:
    """Process and validate function call arguments"""
    query_type = args.get("query_type", "health").lower()

    handlers = {
        "greeting": {
            "name": "handle_greeting",
            "input": {"greeting_text": args.get("greeting_text", user_input)}
        },
        "non_health": {
            "name": "handle_non_health_query",
            "input": {
                "query_text": args.get("query_text", user_input),
                "detected_category": args.get("detected_category", "other")
            }
        }
    }

    return handlers.get(query_type, {
        "name": "handle_health_query",
        "input": {
            "query_text": args.get("query_text", user_input),
            "health_category": args.get("health_category", "general_health")
        }
    })


def handle_greeting(input_data):
    openai.api_key = OPENAI_API_KEY

    prompt = f"""
    You are Grandma, a warm and affectionate elder who shares traditional wisdom and natural remedies.
    A user has greeted you like {input_data}
    Respond in your loving Grandma style, welcoming the user and offering help with traditional remedies. Keep it heartwarming.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip(), []

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
            tool_call = classify_query_with_gemini_functions(user_input, chat_history)

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