import os
import voyageai
import time
from abc import ABC, abstractmethod
from langchain_cohere import CohereEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rag_func.constants.config import EMBEDDING_MODELS, ACTIVE_CONFIG
from langchain_core.embeddings import Embeddings
from rag_func.constants.enums import EmbeddingsTypeEnum
from mistralai import Mistral
from typing import List
from dotenv import load_dotenv

load_dotenv()


class BaseEmbeddingModel(Embeddings, ABC):
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass


class VoyageaiEmbeddings(BaseEmbeddingModel):
    def __init__(self, model_name):
        api_key = os.getenv("VOYAGE_API_KEY")
        self.vo = voyageai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        result = self.vo.embed(texts, model=self.model_name, input_type="document")
        return result.embeddings

    def embed_query(self, text):
        result = self.vo.embed([text], model=self.model_name, input_type="query")
        return result.embeddings[0]


class CohereEmbedding(BaseEmbeddingModel):
    def __init__(self, model_name):
        api_key = os.getenv("COHERE_API_KEY")
        self.model_name = model_name
        self._model = CohereEmbeddings(model=model_name, cohere_api_key=api_key)

    def embed_query(self, query: str) -> List[float]:
        return self._model.embed_query(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self._model.embed_documents(documents)


class MistralEmbeddings(BaseEmbeddingModel):
    def __init__(self, model_name):
        api_key = os.getenv("MISTRAL_API_KEY")
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


class EmbeddingModelFactory:

    @staticmethod
    def get_embedding_model() -> BaseEmbeddingModel:
        model_config = EMBEDDING_MODELS[ACTIVE_CONFIG['embedding']]
        model_type = model_config["type"]
        model_name = model_config["model_name"]

        embedding_classes = {
            EmbeddingsTypeEnum.HuggingFace.value: HuggingFaceEmbeddings,
            EmbeddingsTypeEnum.Voyageai.value: VoyageaiEmbeddings,
            EmbeddingsTypeEnum.Cohere.value: CohereEmbedding,
            EmbeddingsTypeEnum.Mistral.value: MistralEmbeddings
        }

        embedding_class = embedding_classes.get(model_type)
        if embedding_class is None:
            raise ValueError(f"Unsupported embedding type: {model_type}")

        return embedding_class(model_name=model_name)


def get_embedding_model():
    return EmbeddingModelFactory.get_embedding_model()
