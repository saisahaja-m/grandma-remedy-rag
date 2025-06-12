from abc import ABC, abstractmethod
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from rag_func.constants.config import RETRIEVAL, ACTIVE_CONFIG
from rag_func.constants.enums import RetrievalTypesEnum
from langchain_core.documents import Document
from typing import List

class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        pass

class VectorRetriever(BaseRetriever):
    def __init__(self, vector_store, k: int):
        self.retriever = VectorStoreRetriever(vectorstore=vector_store, search_kwargs={"k": k})

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)

class BM25RetrieverWrapper(BaseRetriever):
    def __init__(self, docs: List[Document], k: int):
        self.retriever = BM25Retriever.from_documents(docs)
        self.retriever.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)

class EnsembleRetrieverWrapper(BaseRetriever):
    def __init__(self, docs: List[Document], vector_store, k: int, weights: List[float]):
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        vector_retriever = VectorStoreRetriever(vectorstore=vector_store, search_kwargs={"k": k})
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)

class SemanticRetriever(BaseRetriever):
    def __init__(self, vector_store, k: int, similarity_threshold: float, use_mmr: bool = True,
                 mmr_diversity_penalty: float = 0.5):
        self.vector_store = vector_store
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.use_mmr = use_mmr
        self.mmr_diversity_penalty = mmr_diversity_penalty

    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.use_mmr:
            docs = self.vector_store.max_marginal_relevance_search(
                query, k=self.k, lambda_mult=1 - self.mmr_diversity_penalty
            )
        else:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=self.k * 2
            )
            docs = [
                doc for doc, score in docs_with_scores
                if score >= self.similarity_threshold
            ][:self.k]
        return docs

class RetrieverFactory:
    @staticmethod
    def get_retriever(docs: List[Document], vector_store, retrieval_config: dict = None) -> BaseRetriever:
        if retrieval_config is None:
            retrieval_config = RETRIEVAL[ACTIVE_CONFIG["retrieval"]]
        retrieval_type = retrieval_config["type"]
        k = retrieval_config["k"]

        retriever_classes = {
            RetrievalTypesEnum.Vector.value: lambda: VectorRetriever(
                vector_store=vector_store, k=k
            ),
            RetrievalTypesEnum.bm25.value: lambda: BM25RetrieverWrapper(docs=docs, k=k),
            RetrievalTypesEnum.Ensemble.value: lambda: EnsembleRetrieverWrapper(
                docs=docs, vector_store=vector_store, k=k,
                weights=retrieval_config.get("weights", [0.5, 0.5])
            ),
            RetrievalTypesEnum.Semantic.value: lambda: SemanticRetriever(
                vector_store=vector_store,
                k=k,
                similarity_threshold=retrieval_config.get("similarity_threshold", 0.0)
            )
        }

        retriever_class = retriever_classes.get(retrieval_type)
        if retriever_class is None:
            raise ValueError(f"Unsupported retriever type: {retrieval_type}")

        return retriever_class()

def get_retriever(docs: List[Document], vector_store, retrieval_config: dict = None) -> BaseRetriever:
    return RetrieverFactory.get_retriever(docs, vector_store, retrieval_config)