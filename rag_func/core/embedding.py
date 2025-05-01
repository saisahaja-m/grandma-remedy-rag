import cohere
import numpy as np
import voyageai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rag_func.constants.config import EMBEDDING_MODELS, ACTIVE_CONFIG
from rag_func.constants.config import VOYAGE_API_KEY, COHERE_API_KEY, MISTRAL_API_KEY
from langchain_core.embeddings import Embeddings
from rag_func.constants.enums import EmbeddingsTypeEnum, InputTypesEnum
from mistralai import Mistral

def get_embedding_model():

    model_config = EMBEDDING_MODELS[ACTIVE_CONFIG['embedding']]
    model_type = model_config["type"]

    if model_type == EmbeddingsTypeEnum.HuggingFace.value:
        return HuggingFaceEmbeddings(model_name=model_config["model_name"])
    elif model_type == EmbeddingsTypeEnum.Voyageai.value:
        return VoyageaiEmbeddings(model_name=model_config["model_name"])
    elif model_type == EmbeddingsTypeEnum.Cohere.value:
        return CohereEmbeddings(model_name=model_config["model_name"])
    elif model_type == EmbeddingsTypeEnum.Mistral.value:
        return MistralEmbeddings(model_name=model_config["model_name"])


class VoyageaiEmbeddings(Embeddings):
    def __init__(self, model_name):
        api_key = VOYAGE_API_KEY
        self.vo = voyageai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        result = self.vo.embed(texts, model=self.model_name, input_type=InputTypesEnum.SearchDocument.value)
        return result.embeddings

    def embed_query(self, text):
        result = self.vo.embed([text], model=self.model_name, input_type=InputTypesEnum.SearchQuery.value)
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
                input_type=InputTypesEnum.SearchDocument.value,
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
            input_type=InputTypesEnum.SearchQuery.value,
            output_dimension=1024,
            embedding_types=["float"]
        )

        if hasattr(res.embeddings, 'float'):
            embedding = res.embeddings.float[0]
        elif hasattr(res, 'float_embeddings'):
            embedding = res.float_embeddings[0]
        else:
            embedding = list(res.embeddings)[0]

        embedding_array = np.array(embedding, dtype=np.float32)

        return embedding_array

    def __call__(self, text):
        return self.embed_query(text)


class MistralEmbeddings(Embeddings):
    def __init__(self, model_name):
        api_key = MISTRAL_API_KEY
        self.model = model_name
        self.client = Mistral(api_key=api_key)

    def embed_documents(self, documents):

        embeddings_batch_response = self.client.embeddings.create(
            model=self.model,
            inputs=documents,
        )

        return [e.embedding for e in embeddings_batch_response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            inputs=[text],
        )
        return response.data[0].embedding
