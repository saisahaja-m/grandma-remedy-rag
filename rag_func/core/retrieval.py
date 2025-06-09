from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from rag_func.constants.config import RETRIEVAL, ACTIVE_CONFIG
from rag_func.constants.enums import RetrievalTypesEnum
from rag_func.core.vector_stores import create_vector_store


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


def get_retriever(docs, embedding_model):
    retrieval_config = RETRIEVAL[ACTIVE_CONFIG["retrieval"]]
    retrieval_type = retrieval_config["type"]
    k = retrieval_config["k"]

    if retrieval_type == RetrievalTypesEnum.Vector.value:
        vector_store = create_vector_store(docs, embedding_model)
        return VectorStoreRetriever(vectorstore=vector_store, search_kwargs={"k": k})

    elif retrieval_type == RetrievalTypesEnum.bm25.value:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        return bm25_retriever

    elif retrieval_type == RetrievalTypesEnum.Ensemble.value:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k

        vector_store = create_vector_store(docs, embedding_model)
        vector_retriever = VectorStoreRetriever(vectorstore=vector_store, search_kwargs={"k": k})

        weights = retrieval_config.get("weights", [0.5, 0.5])
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )
    elif retrieval_type == RetrievalTypesEnum.Semantic.value:
        vector_store = create_vector_store(docs, embedding_model)
        similarity_threshold = retrieval_config.get("similarity_threshold", 0.0)
        return SemanticRetriever(vector_store=vector_store, k=k, similarity_threshold=similarity_threshold)

    return None