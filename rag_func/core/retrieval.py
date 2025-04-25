from langchain.vectorstores import FAISS, Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from rag_func.core.embedding import get_embedding_model
from rag_func.config.config import RETRIEVAL, VECTOR_STORES, ACTIVE_CONFIG
from typing import Dict, List, Tuple
import uuid
import chromadb

def create_vector_store(docs):
    embedding_model = get_embedding_model()
    vector_store_config = VECTOR_STORES[ACTIVE_CONFIG["vector_store"]]
    vector_store_type = vector_store_config["type"]

    if vector_store_type == "faiss":
        return FAISS.from_documents(docs, embedding_model)

    # elif vector_store_type == "chroma":
    #     return Chroma.from_documents(
    #         docs,
    #         embedding_model,
    #         persist_directory=vector_store_config.get("persist_directory")
    #     )

    elif vector_store_type == "chroma":
        return ChromaVectorStore(
            docs,
            embedding_model
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


class ChromaVectorStore:
    def __init__(self, documents, embeddings):
        self._client = chromadb.Client()
        self._collection_name = "collection_" + uuid.uuid4().hex[:8]
        self._collection = self._client.create_collection(name=self._collection_name)
        self._id_to_doc: Dict[str, str] = {}
        self.documents = documents
        self.embeddings = embeddings

    def add_documents(self) -> None:
        if not self.documents or not self.embeddings:
            return

        ids = ["doc_" + str(i) for i in range(len(self.documents))]
        self._id_to_doc.update({doc_id: doc for doc_id, doc in zip(ids, self.documents)})

        self._collection.add(
            embeddings=self.embeddings,
            documents=self.documents,
            ids=ids
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._id_to_doc:
            return []

        available_docs = len(self._id_to_doc)
        limit = min(max(1, top_k), available_docs)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        return [(doc, float(dist)) for doc, dist in zip(documents, distances)]
