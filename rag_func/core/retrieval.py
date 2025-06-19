import chromadb
from abc import ABC, abstractmethod
from rag_func.constants.config import RETRIEVAL, ACTIVE_CONFIG, PERSIST_DIRECTORY, COLLECTION_NAME
from rag_func.constants.enums import RetrievalTypesEnum
from langchain_core.documents import Document
from typing import List
from langchain_community.vectorstores import FAISS, Annoy, Chroma

class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents(self, query, embeddings) -> List[Document]:
        pass

class FaissRetriever(BaseRetriever):
    def get_relevant_documents(self, query, embeddings) -> List[Document]:
        k = RETRIEVAL[RetrievalTypesEnum.Faiss.value]["k"]
        new_vector_store = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_vector_store.similarity_search(query=query, k=k)

        return docs

class AnnoyRetriever(BaseRetriever):
    def get_relevant_documents(self, query, embeddings) -> List[Document]:
        k = RETRIEVAL[RetrievalTypesEnum.Annoy.value]["k"]
        loaded_vector_store = Annoy.load_local(
            folder_path="/home/ib-developer/Windsurf projects/grandma_remedy/annoy_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        docs = loaded_vector_store.similarity_search(query=query, k=k)
        return docs

class ChromaRetriever(BaseRetriever):
    def get_relevant_documents(self, query, embeddings) -> List[Document]:
        k = RETRIEVAL[RetrievalTypesEnum.Chroma.value]["k"]
        persistent_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        vector_store = Chroma(
            client=persistent_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        docs = vector_store.similarity_search(query, k=k)

        return docs


class RetrieverFactory:
    @staticmethod
    def get_retriever() -> BaseRetriever:
        retrieval_config = RETRIEVAL[ACTIVE_CONFIG["retrieval"]]
        retrieval_type = retrieval_config["type"]
        k = retrieval_config["k"]

        retriever_classes = {
            RetrievalTypesEnum.Faiss.value: lambda: FaissRetriever(),
            RetrievalTypesEnum.Annoy.value: lambda: AnnoyRetriever(),
            RetrievalTypesEnum.Chroma.value: lambda: ChromaRetriever()
        }

        retriever_class = retriever_classes.get(retrieval_type)
        if retriever_class is None:
            raise ValueError(f"Unsupported retriever type: {retrieval_type}")

        return retriever_class()

def get_retriever() -> BaseRetriever:
    return RetrieverFactory.get_retriever()
