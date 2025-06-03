import uuid
from langchain_community.vectorstores import Chroma, FAISS, Annoy
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from rag_func.core.embedding import get_embedding_model
from rag_func.constants.config import RETRIEVAL, VECTOR_STORES, ACTIVE_CONFIG
from rag_func.constants.enums import VectorStoresEnum, RetrievalTypesEnum


def create_vector_store(docs):
    embedding_model = get_embedding_model()
    vector_store_config = VECTOR_STORES[ACTIVE_CONFIG["vector_store"]]
    vector_store_type = vector_store_config["type"]

    if vector_store_type == VectorStoresEnum.Faiss.value:
        return FAISS.from_documents(docs, embedding_model)
    elif vector_store_type == VectorStoresEnum.Chroma.value:
        return Chroma.from_documents(docs, embedding_model)
    elif vector_store_type == VectorStoresEnum.Annoy.value:
        return Annoy.from_documents(docs, embedding_model)
    return None


def get_retriever(docs):
    retrieval_config = RETRIEVAL[ACTIVE_CONFIG["retrieval"]]
    retrieval_type = retrieval_config["type"]
    k = retrieval_config["k"]

    if retrieval_type == RetrievalTypesEnum.Vector.value:
        vector_store = create_vector_store(docs)
        return VectorStoreRetriever(vectorstore=vector_store, search_kwargs={"k": k})

    elif retrieval_type == RetrievalTypesEnum.bm25.value:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        return bm25_retriever

    elif retrieval_type == RetrievalTypesEnum.Ensemble.value:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k

        vector_store = create_vector_store(docs)
        vector_retriever = VectorStoreRetriever(vectorstore=vector_store, search_kwargs={"k": k})

        weights = retrieval_config.get("weights", [0.5, 0.5])
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )
    elif retrieval_type == RetrievalTypesEnum.Semantic.value:
        vector_store = create_vector_store(docs)
        similarity_threshold = retrieval_config.get("similarity_threshold", 0.0)
        return SemanticRetriever(vector_store=vector_store, k=k, similarity_threshold=similarity_threshold)

    return None


class SemanticRetriever:
    def __init__(self, vector_store, k: int, similarity_threshold: float, use_mmr: bool = True,
                 mmr_diversity_penalty: float = 0.5):
        self.vector_store = vector_store
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.use_mmr = use_mmr
        self.mmr_diversity_penalty = mmr_diversity_penalty

    def get_relevant_documents(self, query: str):
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
