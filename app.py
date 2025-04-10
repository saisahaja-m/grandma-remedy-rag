import streamlit as st
import google.generativeai as genai
from process_data import load_and_process_documents
from data_retrieving import setup_retrieval_system

GOOGLE_API_KEY = "AIzaSyBm_X0w1tq3QD_yRuQ2jnV4LV77NsSTc5Y"
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

@st.cache_resource
def get_retrieval_system():
    docs = load_and_process_documents()
    return setup_retrieval_system(docs)

def format_chat_history(messages, max_tokens=1000):
    history = ""
    for i in range(0, len(messages), 2):
        user_msg = messages[i]["content"]
        assistant_msg = messages[i + 1]["content"] if i + 1 < len(messages) else ""
        history += f"User: {user_msg}\nGrandma: {assistant_msg}\n"
    # Trim history if needed
    return history[-max_tokens:]

def main():
    st.set_page_config(page_title="🌿 Grandma's Remedy RAG", page_icon="🌿", layout="wide")
    st.title("🌿 Grandma's Remedy RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": ""}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Tell Grandma your problem...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        retrieval_system = get_retrieval_system()
        retriever = retrieval_system["retriever"]
        relevant_docs = retriever.get_relevant_documents(user_input)

        context = "\n\n".join([
            f"SOURCE: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
            for doc in relevant_docs
        ])

        chat_history = format_chat_history(st.session_state.messages[:-1])

        grandma_prompt = f"""
        You are Grandma Remedy Bot, an expert in Indian home remedies and ayurvedic wisdom.

        USER QUERY: "{user_input}"

        CHAT HISTORY:
        {chat_history}

        RELEVANT REMEDIES:
        {context}

        INSTRUCTIONS:
        1. Respond in a warm, loving Indian grandma tone using words like beta or baccha.
        2. Recommend 1–3 home remedies from the RELEVANT REMEDIES section.
        3. Explain preparation and usage.
        4. End with a kind note or health wish.
        5. If you don't know, say so lovingly.
        """

        model = load_gemini_model()
        response = model.generate_content(grandma_prompt)

        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

main()
