import cohere
import numpy as np
import voyageai
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_func.config.config import EMBEDDING_MODELS
from rag_func.config.config import VOYAGE_API_KEY, COHERE_API_KEY
from langchain_core.embeddings import Embeddings


def get_embedding_model():

    model_config = EMBEDDING_MODELS["cohere"]
    model_type = model_config["type"]

    if model_type == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_config["model_name"])
    elif model_type == "voyageai":
        return VoyageaiEmbeddings(model_name=model_config["model_name"])
    elif model_type == "cohere":
        return CohereEmbeddings(model_name=model_config["model_name"])


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


class CohereEmbeddings:
    def __init__(self, model_name):
        api_key = COHERE_API_KEY
        self.co = cohere.ClientV2(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        batch_size = 96
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            res = self.co.embed(
                texts=batch,
                model=self.model_name,
                input_type="search_document",
                output_dimension=1024,
                embedding_types=["float"]
            )

            if hasattr(res.embeddings, 'float'):
                batch_embeddings = res.embeddings.float
            elif hasattr(res, 'float_embeddings'):
                batch_embeddings = res.float_embeddings
            else:
                batch_embeddings = list(res.embeddings)

            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, text):
        res = self.co.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query",
            output_dimension=1024,
            embedding_types=["float"]
        )

        if hasattr(res.embeddings, 'float'):
            embedding = res.embeddings.float[0]
        elif hasattr(res, 'float_embeddings'):
            embedding = res.float_embeddings[0]
        else:
            embedding = list(res.embeddings)[0]

        # Convert to numpy array and ensure it's the right shape
        embedding_array = np.array(embedding, dtype=np.float32)

        return embedding_array

    def __call__(self, text):
        return self.embed_query(text)