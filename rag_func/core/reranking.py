import requests
import cohere
import re
from langchain.schema import Document
from rag_func.config.config import RERANKING, GROQ_API_KEY, ACTIVE_CONFIG, COHERE_API_KEY
from typing import List, Dict
from rag_func.config.enums import RerankingTypesEnum


def get_reranker():
    rerank_config = RERANKING[ACTIVE_CONFIG["reranking"]]
    rerank_type = rerank_config["type"]

    if rerank_type == RerankingTypesEnum.Groq.value:
        return GroqReranker(
            model=rerank_config["model"],
            top_k=rerank_config.get("top_k", 5)
        )
    elif rerank_type == RerankingTypesEnum.Cohere.value:
        return CohereReranker(
            model=rerank_config["model"])
    elif rerank_type == RerankingTypesEnum.Jina.value:
        return JinaReranker(
            model=rerank_config["model"],
            top_k=rerank_config["top_k"]
        )


class GroqReranker:
    def __init__(self, model: str, top_k: int = 5, api_key: str = None):
        self.model = model
        self.top_k = top_k
        self.api_key = api_key or GROQ_API_KEY  # Fallback if passed as None

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        reranked = []

        for doc in documents:
            prompt = (
                "You are a helpful assistant. Score the relevance of the following context "
                "to the user's query on a scale from 0.0 to 1.0.\n\n"
                f"Query: {query}\n"
                f"Context:\n{doc.page_content}\n\n"
                "Score (respond with only a float from 0.0 to 1.0):"
            )

            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
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
                    json=data,
                    timeout=10
                )

                response.raise_for_status()
                score_text = response.json()["choices"][0]["message"]["content"].strip()
                match = re.search(r"\d*\.?\d+", score_text)
                score = float(match.group()) if match else 0.0

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

class JinaReranker:
    def __init__(self, model, top_k):
        self.model = model
        self.top_k = top_k

    def rerank(self, query, documents):
        formatted_docs = [{"text": doc.page_content} for doc in documents]

        url = 'https://api.jina.ai/v1/rerank'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer jina_85bf0e24ca9646a7860a7095e4771a69x0FCKbVTF9JRNEb_bT88iQF4xiou'
        }
        data = {
            "model": self.model,
            "query": query,
            "top_n": self.top_k,
            "documents": formatted_docs,
            "return_documents": False
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = response.json()

        reranked_documents = response_data.get("reranked_documents", [])

        return reranked_documents