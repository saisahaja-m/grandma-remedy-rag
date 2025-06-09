from langchain_community.vectorstores import Chroma, FAISS, Annoy
from rag_func.constants.config import VECTOR_STORES, ACTIVE_CONFIG
from rag_func.constants.enums import VectorStoresEnum


def create_vector_store(docs, embedding_model):
    vector_store_config = VECTOR_STORES[ACTIVE_CONFIG["vector_store"]]
    vector_store_type = vector_store_config["type"]

    if vector_store_type == VectorStoresEnum.Faiss.value:
        return FAISS.from_documents(docs, embedding_model)
    elif vector_store_type == VectorStoresEnum.Chroma.value:
        return Chroma.from_documents(docs, embedding_model)
    elif vector_store_type == VectorStoresEnum.Annoy.value:
        return Annoy.from_documents(docs, embedding_model)
    return None