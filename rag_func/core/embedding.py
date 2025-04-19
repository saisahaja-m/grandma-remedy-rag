from langchain.embeddings import HuggingFaceEmbeddings
from rag_func.config.config import EMBEDDING_MODELS, ACTIVE_CONFIG


def get_embedding_model():

    model_config = EMBEDDING_MODELS[ACTIVE_CONFIG["embedding"]]
    model_type = model_config["type"]

    if model_type == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_config["model_name"])
