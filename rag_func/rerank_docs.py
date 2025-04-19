import streamlit as st
import requests


@st.cache_resource
def rerank_with_groq(query, _documents, top_k=5):
    api_key = "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"
    reranked = []

    for doc in _documents:
        prompt = f"""You are a helpful assistant. Score the relevance of the following context to the user's query on a scale from 0 to 1.

Query: {query}

Context:
{doc.page_content}

Score (just a number between 0 and 1):"""

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            }

            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
            score_text = response.json()["choices"][0]["message"]["content"].strip()
            score = float(score_text.split()[0])  # Get the first number from the response
            reranked.append((doc, score))

        except Exception as e:
            print(f"Error scoring doc: {e}")
            reranked.append((doc, 0.0))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]
