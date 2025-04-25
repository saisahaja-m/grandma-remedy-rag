from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_func.config.config import EMBEDDING_MODELS, ACTIVE_CONFIG
import voyageai
from rag_func.config.config import VOYAGE_API_KEY
from langchain_core.embeddings import Embeddings


def get_embedding_model():

    model_config = EMBEDDING_MODELS["voyageai"]
    model_type = model_config["type"]

    if model_type == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_config["model_name"])
    elif model_type == "voyageai":
        return VoyageaiEmbeddings(model_name=model_config["model_name"])


class VoyageaiEmbeddings(Embeddings):
    def __init__(self, model_name):
        api_key = VOYAGE_API_KEY
        self.vo = voyageai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        result = self.vo.embed(texts, model=self.model_name, input_type="document")
        return result.embeddings

    def embed_query(self, text):
        result = self.vo.embed([text], model=self.model_name, input_type="query")
        return result.embeddings[0]
