from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma, FAISS, Annoy
from rag_func.constants.config import VECTOR_STORES, ACTIVE_CONFIG
from rag_func.constants.enums import VectorStoresEnum
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List

class BaseVectorStore(VectorStore, ABC):
    @classmethod
    @abstractmethod
    def from_documents(cls, documents: List[Document], embedding, **kwargs) -> VectorStore:
        pass

class FaissVectorStore(FAISS, BaseVectorStore):
    @classmethod
    def from_documents(cls, documents: List[Document], embedding, **kwargs) -> VectorStore:
        return FAISS.from_documents(documents, embedding, **kwargs)

class ChromaVectorStore(Chroma, BaseVectorStore):
    @classmethod
    def from_documents(cls, documents: List[Document], embedding, **kwargs) -> VectorStore:
        return Chroma.from_documents(documents, embedding, **kwargs)

class AnnoyVectorStore(Annoy, BaseVectorStore):
    @classmethod
    def from_documents(cls, documents: List[Document], embedding, **kwargs) -> VectorStore:
        return Annoy.from_documents(documents, embedding, **kwargs)

class VectorStoreFactory:
    @staticmethod
    def create_vector_store(docs: List[Document], embedding_model, **kwargs) -> VectorStore:
        vector_store_config = VECTOR_STORES[ACTIVE_CONFIG["vector_store"]]
        vector_store_type = vector_store_config["type"]

        vector_store_classes = {
            VectorStoresEnum.Faiss.value: FaissVectorStore,
            VectorStoresEnum.Chroma.value: ChromaVectorStore,
            VectorStoresEnum.Annoy.value: AnnoyVectorStore
        }

        vector_store_class = vector_store_classes.get(vector_store_type)
        if vector_store_class is None:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")

        return vector_store_class.from_documents(docs, embedding_model, **kwargs)

def create_vector_store(docs: List[Document], embedding_model, **kwargs):
    return VectorStoreFactory.create_vector_store(docs, embedding_model, **kwargs)
