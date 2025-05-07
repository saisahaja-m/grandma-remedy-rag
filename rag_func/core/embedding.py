from langchain_cohere import CohereEmbeddings
import numpy as np
import voyageai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rag_func.constants.config import EMBEDDING_MODELS, ACTIVE_CONFIG
from rag_func.constants.config import VOYAGE_API_KEY, COHERE_API_KEY, MISTRAL_API_KEY
from langchain_core.embeddings import Embeddings
from rag_func.constants.enums import EmbeddingsTypeEnum, InputTypesEnum
from mistralai import Mistral
from typing import List

def get_embedding_model():

    model_config = EMBEDDING_MODELS[ACTIVE_CONFIG['embedding']]
    model_type = model_config["type"]

    if model_type == EmbeddingsTypeEnum.HuggingFace.value:
        return HuggingFaceEmbeddings(model_name=model_config["model_name"])
    elif model_type == EmbeddingsTypeEnum.Voyageai.value:
        return VoyageaiEmbeddings(model_name=model_config["model_name"])
    elif model_type == EmbeddingsTypeEnum.Cohere.value:
        return CohereEmbedding(model_name=model_config["model_name"])
    elif model_type == EmbeddingsTypeEnum.Mistral.value:
        return MistralEmbeddings(model_name=model_config["model_name"])
    return None


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


class CohereEmbedding(Embeddings):
    def __init__(self, model_name):
        api_key = COHERE_API_KEY
        self.model_name = model_name
        self._model = CohereEmbeddings(model=model_name, cohere_api_key=api_key)

    def embed_query(self, query: str) -> List[float]:
        return self._model.embed_query(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self._model.embed_documents(documents)


import time

class MistralEmbeddings(Embeddings):
    def __init__(self, model_name):
        api_key = MISTRAL_API_KEY
        self.model = model_name
        self.client = Mistral(api_key=api_key)
        self.sleep_seconds = 2
        self.batch_size = 8

    def embed_documents(self, documents):
        all_embeddings = []

        for i in range(0, len(documents), self.batch_size):
            if i > 0:
                time.sleep(self.sleep_seconds)

            batch = documents[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    inputs=batch
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Error embedding batch {i // self.batch_size + 1}: {e}")
                raise

        return all_embeddings

    def embed_query(self, text):
        time.sleep(self.sleep_seconds)
        response = self.client.embeddings.create(
            model=self.model,
            inputs=[text],
        )
        return response.data[0].embedding
