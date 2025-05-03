import uuid
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from rag_func.core.embedding import get_embedding_model
from rag_func.constants.config import RETRIEVAL, VECTOR_STORES, ACTIVE_CONFIG
from typing import Dict, List, Tuple
from rag_func.constants.enums import VectorStoresEnum, RetrievalTypesEnum
from langchain_community.vectorstores import Annoy


def create_vector_store(docs):
    embedding_model = get_embedding_model()
    vector_store_config = VECTOR_STORES[ACTIVE_CONFIG["vector_store"]]
    vector_store_type = vector_store_config["type"]

    if vector_store_type == VectorStoresEnum.Faiss.value:
        return FAISS.from_documents(docs, embedding_model)

    elif vector_store_type == VectorStoresEnum.Chroma.value:
        return Chroma.from_documents(docs, embedding_model)
    elif vector_store_type == VectorStoresEnum.Annoy.value:
        return AnnoyVectorStore(docs, embeddings=embedding_model)


def get_retriever(docs):
    retrieval_config = RETRIEVAL[ACTIVE_CONFIG["retrieval"]]
    retrieval_type = retrieval_config["type"]
    k = retrieval_config["k"]

    if retrieval_type == RetrievalTypesEnum.Vector.value:
        vector_store = create_vector_store(docs)
        return vector_store.as_retriever(search_kwargs={"k": k})

    elif retrieval_type == RetrievalTypesEnum.bm25.value:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        return bm25_retriever

    elif retrieval_type == RetrievalTypesEnum.Ensemble.value:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k

        vector_store = create_vector_store(docs)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

        weights = retrieval_config.get("weights", [0.5, 0.5])
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )


# class ChromaVectorStore:
#     def __init__(self, documents, embeddings):
#         self._client = Chroma.Client()
#         self._collection_name = "collection_" + uuid.uuid4().hex[:8]
#         self._collection = self._client.create_collection(name=self._collection_name)
#         self._id_to_doc: Dict[str, str] = {}
#         self.documents = documents
#         self.embeddings = embeddings
#
#     def add_documents(self) -> None:
#         if not self.documents or not self.embeddings:
#             return
#
#         ids = ["doc_" + str(i) for i in range(len(self.documents))]
#         self._id_to_doc.update({doc_id: doc for doc_id, doc in zip(ids, self.documents)})
#
#         self._collection.add(
#             embeddings=self.embeddings,
#             documents=self.documents,
#             ids=ids
#         )
#
#     def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
#         if not self._id_to_doc:
#             return []
#
#         available_docs = len(self._id_to_doc)
#         limit = min(max(1, top_k), available_docs)
#
#         results = self._collection.query(
#             query_embeddings=[query_embedding],
#             n_results=limit
#         )
#
#         documents = results.get('documents', [[]])[0]
#         distances = results.get('distances', [[]])[0]
#
#         return [(doc, float(dist)) for doc, dist in zip(documents, distances)]


# class QdrantVectorStore:
#     def __init__(self, documents, embeddings_model):
#         host = "localhost"
#         port = 6333
#         self.client = QdrantClient(host=host, port=port)
#
#         self._collection_name = "collection_" + uuid.uuid4().hex[:8]
#
#         self.documents = documents
#         self.embeddings_model = embeddings_model
#
#         self._id_to_doc = {}
#
#         if documents and embeddings_model:
#             texts = [doc.page_content for doc in documents]
#             self.embeddings = embeddings_model.embed_documents(texts)
#             vector_size = len(self.embeddings[0])
#             self.client.create_collection(
#                 collection_name=self._collection_name,
#                 vectors_config=VectorParams(
#                     size=vector_size,
#                     distance=Distance.COSINE
#                 )
#             )
#             self.add_documents()
#
#     def add_documents(self) -> None:
#         if not self.documents or not self.embeddings:
#             return
#
#         ids = [i for i in range(len(self.documents))]
#         self._id_to_doc.update({doc_id: doc for doc_id, doc in zip(ids, self.documents)})
#
#         points = []
#         for i, (embedding, document) in enumerate(zip(self.embeddings, self.documents)):
#             points.append(
#                 PointStruct(
#                     id=ids[i],
#                     vector=embedding,
#                     payload={"text": document}
#                 )
#             )
#
#         self.client.upsert(
#             collection_name=self._collection_name,
#             points=points
#         )
#
#     def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
#         if not self._id_to_doc:
#             return []
#
#         available_docs = len(self._id_to_doc)
#         limit = min(max(1, top_k), available_docs)
#
#         results = self.client.search(
#             collection_name=self._collection_name,
#             query_vector=query_embedding,
#             limit=limit
#         )
#
#         result_tuples = []
#         for scored_point in results:
#             doc_text = scored_point.payload.get("text", "")
#             distance = scored_point.score
#             result_tuples.append((doc_text, float(distance)))
#
#         return result_tuples
#
#

class AnnoyVectorStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def vector_store(self):
        vector_store = Annoy.from_documents(documents=self.documents, embedding=self.embeddings)

        return vector_store
