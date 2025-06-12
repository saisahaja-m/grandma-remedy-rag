import json
import os
import requests
import cohere
import re
from abc import ABC, abstractmethod
from langchain.schema import Document
from rag_func.constants.config import RERANKING, ACTIVE_CONFIG
from rag_func.constants.enums import RerankingTypesEnum
from groq import Groq
from dotenv import load_dotenv
from typing import List
from rag_func.prompt_providers.prompt_service.prompt_provider import RerankingPromptProvider

load_dotenv()

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        pass

class GroqReranker(BaseReranker):
    def __init__(self, model: str, top_k: int = 5):
        self.model = model
        self.top_k = top_k
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        reranked = []
        prompt_provider = RerankingPromptProvider()

        for doc in documents:
            prompt = prompt_provider.get_user_prompt(query=query, page_content=doc.page_content)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.choices[0].message.content.strip()
            match = re.search(r"\d*\.?\d+", score_text)
            score = float(match.group()) if match else 0.0
            reranked.append((doc, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:self.top_k]]


class CohereReranker(BaseReranker):
    def __init__(self, model: str, top_k: int = 5):
        api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(api_key)
        self.model = model
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        doc_texts = [doc.page_content for doc in documents]
        response = self.client.rerank(
            query=query,
            documents=doc_texts,
            model=self.model,
            top_n=self.top_k
        )
        reranked_docs = []
        for result in response.results:
            idx = result.index
            if 0 <= idx < len(documents):
                doc = documents[idx]
                if hasattr(doc, 'metadata'):
                    doc.metadata['relevance_score'] = result.relevance_score
                reranked_docs.append(doc)
        return reranked_docs


class JinaReranker(BaseReranker):
    def __init__(self, model: str, top_k: int):
        self.model = model
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        formatted_docs = [{"text": doc.page_content} for doc in documents]
        url = 'https://api.jina.ai/v1/rerank'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': os.getenv("JINA_API_KEY")
        }
        data = {
            "model": self.model,
            "query": query,
            "top_n": self.top_k,
            "documents": formatted_docs,
            "return_documents": True
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_data = response.json()

        reranked_docs = []
        if 'results' in response_data:
            sorted_results = sorted(response_data['results'], key=lambda x: x['index'])

            for result in sorted_results:
                idx = result['index']
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    if hasattr(doc, 'metadata'):
                        doc.metadata['relevance_score'] = result['relevance_score']
                    reranked_docs.append(doc)

        return reranked_docs

class RerankerFactory:
    @staticmethod
    def get_reranker() -> BaseReranker:
        rerank_config = RERANKING[ACTIVE_CONFIG["reranking"]]
        rerank_type = rerank_config["type"]
        model = rerank_config["model"]
        top_k = rerank_config.get("top_k", 5)

        reranker_classes = {
            RerankingTypesEnum.Groq.value: GroqReranker,
            RerankingTypesEnum.Cohere.value: CohereReranker,
            RerankingTypesEnum.Jina.value: JinaReranker
        }

        reranker_class = reranker_classes.get(rerank_type)
        if reranker_class is None:
            raise ValueError(f"Unsupported reranker type: {rerank_type}")

        return reranker_class(model=model, top_k=top_k)

def get_reranker() -> BaseReranker:
    return RerankerFactory.get_reranker()
