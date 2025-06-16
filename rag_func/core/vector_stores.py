import os
import faiss
from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma, FAISS, Annoy
from rag_func.constants.config import VECTOR_STORES, ACTIVE_CONFIG, PERSIST_DIRECTORY, COLLECTION_NAME, \
    ANNOY_VECTOR_STORE_FILE_PATH
from rag_func.constants.enums import VectorStoresEnum
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np
import chromadb

class BaseVectorStore(ABC):
    @classmethod
    @abstractmethod
    def from_documents(cls, documents: List[Document], embedding) -> VectorStore:
        pass

class FaissVectorStore(BaseVectorStore):
    @classmethod
    def from_documents(cls, documents: List[Document], embedding) -> VectorStore:
        texts = [doc.page_content for doc in documents]
        embeddings = embedding.embed_documents(texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))

        index.add(np.array(embeddings, dtype='float32'))

        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: i for i in range(len(documents))}

        vector_store = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        vector_store.save_local("faiss_index")
        return vector_store


class ChromaVectorStore(BaseVectorStore):
    @classmethod
    def from_documents(cls, documents: List[Document], embedding) -> VectorStore:
        documents = [doc.page_content for doc in documents]

        persistent_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = persistent_client.get_or_create_collection(name=COLLECTION_NAME)

        start_id = collection.count() + 1
        doc_ids = [str(i) for i in range(start_id, start_id + len(documents))]

        embeddings = embedding.embed_documents(documents)

        collection.add(
            ids=doc_ids,
            documents=documents,
            embeddings=embeddings
        )

        vector_store = Chroma(
            client=persistent_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding
        )

        return vector_store


class AnnoyVectorStore(BaseVectorStore):
    @classmethod
    def from_documents(cls, documents: List[Document], embedding) -> VectorStore:
        texts = [doc.page_content for doc in documents]

        vector_store = Annoy.from_texts(texts, embedding)
        save_path = ANNOY_VECTOR_STORE_FILE_PATH

        os.makedirs(save_path, exist_ok=True)
        vector_store.save_local(save_path)

        return vector_store


class VectorStoreFactory:
    @staticmethod
    def create_vector_store(docs: List[Document], embedding) -> VectorStore:
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

        return vector_store_class.from_documents(docs, embedding)

def create_vector_store(docs: List[Document], embedding):
    return VectorStoreFactory.create_vector_store(docs, embedding)
