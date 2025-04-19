from langchain.vectorstores import FAISS, Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from rag_func.core.embedding import get_embedding_model
from rag_func.config.config import RETRIEVAL, VECTOR_STORES, ACTIVE_CONFIG


def create_vector_store(docs):
    embedding_model = get_embedding_model()
    vector_store_config = VECTOR_STORES[ACTIVE_CONFIG["vector_store"]]
    vector_store_type = vector_store_config["type"]

    if vector_store_type == "faiss":
        return FAISS.from_documents(docs, embedding_model)

    elif vector_store_type == "chroma":
        return Chroma.from_documents(
            docs,
            embedding_model,
            persist_directory=vector_store_config.get("persist_directory")
        )


def get_retriever(docs):
    retrieval_config = RETRIEVAL[ACTIVE_CONFIG["retrieval"]]
    retrieval_type = retrieval_config["type"]
    k = retrieval_config.get("k", 5)

    if retrieval_type == "vector":
        vector_store = create_vector_store(docs)
        return vector_store.as_retriever(search_kwargs={"k": k})

    elif retrieval_type == "bm25":
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        return bm25_retriever

    elif retrieval_type == "ensemble":
        # Create individual retrievers
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k

        vector_store = create_vector_store(docs)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

        # Create ensemble retriever
        weights = retrieval_config.get("weights", [0.5, 0.5])
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )