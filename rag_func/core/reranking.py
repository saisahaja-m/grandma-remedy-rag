import requests
from langchain.schema import Document
from rag_func.config.config import RERANKING, GROQ_API_KEY, ACTIVE_CONFIG, COHERE_API_KEY
import cohere
from typing import List, Dict


def get_reranker():
    rerank_config = RERANKING["groq"]
    rerank_type = rerank_config["type"]

    if rerank_type == "groq":
        return GroqReranker(
            model=rerank_config["model"],
            top_k=rerank_config.get("top_k", 5)
        )
    elif rerank_type == "cohere":
        return CohereReranker(
            model=rerank_config["model"])


class GroqReranker():

    def __init__(self, model, top_k=5):
        self.model = model
        self.top_k = top_k
        self.api_key = GROQ_API_KEY

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:

        reranked = []

        for doc in documents:
            prompt = f"""
            You are a helpful assistant. Score the relevance of the following context to the user's query on a scale from 0 to 1.

            Query: {query}
            
            Context:
            {doc.page_content}
            
            Score (just a number between 0 and 1):"""

            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data
                )

                score_text = response.json()["choices"][0]["message"]["content"].strip()
                score = float(score_text.split()[0])  # Get first number from response
                reranked.append((doc, score))

            except Exception as e:
                print(f"Error scoring doc: {e}")
                reranked.append((doc, 0.0))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:self.top_k]]


class CohereReranker:
    def __init__(self, model: str):
        api_key = COHERE_API_KEY
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, float]]:
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_n
        )
        return [
        {
            "document": result.document["text"] if result.document else "",
            "score": result.relevance_score
        }
        for result in response.results
        if result.document and result.document.get("text")
    ]