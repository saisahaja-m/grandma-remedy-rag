import os
from dotenv import load_dotenv
from rag_func.constants.enums import EmbeddingsTypeEnum, ChunkingTypeEnum, LLMTypesEnum, EvaluatorTypesEnum, EvaluatingMetricsEnum, \
                    VectorStoresEnum, RetrievalTypesEnum, RerankingTypesEnum

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# URLs for data sources
URLS = [
    "https://www.healthline.com/health/constipation/instant-indian-home-remedy-for-constipation",
    "https://www.healthline.com/health/beauty-skin-care/indian-home-remedies-for-hair-growth",
    "https://www.healthline.com/health/home-remedies",
    "https://www.healthline.com/health/dental-and-oral-health/home-remedies-for-toothache",
    "https://www.healthline.com/health/pain-relief/knee-pain-home-remedies",
    "https://www.healthline.com/nutrition/how-long-does-it-take-to-lose-weight",
    "https://www.healthline.com/health/excessive-sleepiness#causes",
    "https://www.healthline.com/nutrition/vitamins-for-dry-skin",
    "https://www.healthline.com/nutrition/12-foods-for-healthy-skin",
    "https://www.healthline.com/nutrition/vitamins-for-nails?utm_source=ReadNext"
]

# Document processing config
DOC_PROCESSING = {
    "chunk_size": 700,
    "chunk_overlap": 100
}

# Embedding model configurations
EMBEDDING_MODELS = {
    EmbeddingsTypeEnum.HuggingFace.value: {
        "type": EmbeddingsTypeEnum.HuggingFace.value,
        "model_name": "all-MiniLM-L6-v2"
    },
    EmbeddingsTypeEnum.Voyageai.value: {
        "type": EmbeddingsTypeEnum.Voyageai.value,
        "model_name": "voyage-3"
    },
    EmbeddingsTypeEnum.Cohere.value: {
        "type": EmbeddingsTypeEnum.Cohere.value,
        "model_name": "embed-english-v3.0"
    },
    EmbeddingsTypeEnum.Mistral.value:{
        "type": EmbeddingsTypeEnum.Mistral.value,
        "model_name": "mistral-embed"
    }
}

# Vector store configurations
VECTOR_STORES = {
    VectorStoresEnum.Faiss.value: {
        "type": VectorStoresEnum.Faiss.value
    },
    VectorStoresEnum.Chroma.value: {
        "type": VectorStoresEnum.Chroma.value,
        "persist_directory": "./chroma_db"
    },
    VectorStoresEnum.Annoy.value: {
        "type": VectorStoresEnum.Annoy.value
    }
}

# Retrieval configurations
RETRIEVAL = {
    RetrievalTypesEnum.Ensemble.value: {
        "type": RetrievalTypesEnum.Ensemble.value,
        "retrievers": ["bm25", "vector"],
        "weights": [0.3, 0.7],
        "k": 5
    },
    RetrievalTypesEnum.Vector.value: {
        "type": RetrievalTypesEnum.Vector.value,
        "k": 5
    },
    RetrievalTypesEnum.bm25.value: {
        "type": RetrievalTypesEnum.bm25.value,
        "k": 20
    },
    RetrievalTypesEnum.Semantic.value: {
        "type": "semantic",
        "k": 20
    }
}

# LLM configurations
LLM_MODELS = {
    LLMTypesEnum.GeminiLLM.value: {
        "type": LLMTypesEnum.GeminiLLM.value,
        "model_name": "gemini-2.0-flash-exp",
        "temperature": 0.2
    },
    LLMTypesEnum.OpenAiLLM.value: {
        "type": LLMTypesEnum.OpenAiLLM.value,
        "model_name": "gpt-4-turbo",
        "temperature": 0.2
    },
    LLMTypesEnum.ClaudeLLM.value:{
        "type": LLMTypesEnum.ClaudeLLM.value,
        "model_name": "claude-3-5-sonnet-20240620",
        "temperature": 0.2
    }
}

RERANKING = {
    RerankingTypesEnum.Groq.value: {
        "type": RerankingTypesEnum.Groq.value,
        "model": "llama3-8b-8192",
        "top_k": 5
    },
    RerankingTypesEnum.Cohere.value: {
        "type": RerankingTypesEnum.Cohere.value,
        "model": "rerank-v3.5",
        "top_k": 5
    },
    RerankingTypesEnum.Jina.value: {
        "type": RerankingTypesEnum.Jina.value,
        "model": "jina-reranker-v2-base-multilingual",
        "top_k": 5
    }
}

CHUNKING = {
    ChunkingTypeEnum.Manual.value: {
        "type": ChunkingTypeEnum.Manual.value,
        "chunk_size": 700,
        "chunk_overlap": 100
    },
    ChunkingTypeEnum.SentenceWindow.value: {
        "type": ChunkingTypeEnum.SentenceWindow.value,
        "max_window_size": 5,
        "stride": 2
    },
    ChunkingTypeEnum.Recursive.value: {
        "type": ChunkingTypeEnum.Recursive.value,
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "markdown": {
        "type": "markdown"
    },
    ChunkingTypeEnum.Semantic.value: {
        "type": ChunkingTypeEnum.Semantic.value,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}

EVALUATION = {
    EvaluatorTypesEnum.RagasEvaluator.value: {
        "type": EvaluatorTypesEnum.RagasEvaluator.value,
        "metrics": [
            EvaluatingMetricsEnum.Faithfulness.value,
            EvaluatingMetricsEnum.AnswerRelevancy.value,
            EvaluatingMetricsEnum.Groundedness.value,
            EvaluatingMetricsEnum.ContextRelevance.value
        ]
    },
    EvaluatorTypesEnum.TrulensEvaluator.value: {
        "type": EvaluatorTypesEnum.TrulensEvaluator.value,
        "metrics": [],
        "model_name": "gpt-4.1-mini"
    },
    EvaluatorTypesEnum.DeepEvalEvaluator.value: {
        "type": EvaluatorTypesEnum.DeepEvalEvaluator.value,
        "model_name": "gpt-4o"
    }
}

# App configuration
APP_CONFIG = {
    "title": "🌿 Grandma's Remedy RAG",
    "page_icon": "🌿",
    "prompt_template": """
    You are Grandma Remedy Bot, an expert in Indian home remedies and ayurvedic wisdom.

    USER QUERY: "{query}"

    CHAT HISTORY:
    {chat_history}

    RELEVANT REMEDIES:
    {context}

    MEMORIES:
    {memories}

    INSTRUCTIONS:
        1. Answer **only** questions related to health, wellness, or the human body.
        2. **CRITICAL: NEVER suggest ingredients that the user has expressed dislike for in USER PREFERENCES**. If they've said they don't like an ingredient, do not recommend it in any form.
        3. Always be warm, loving, and nurturing—use affectionate terms like *beta* or *baccha* when appropriate.
        4. Never make up remedies. Your advice should be based on trustworthy sources such as Ayurvedic books, recognized wellness websites, or traditionally known practices.
        5. Back up your advice with friendly references when possible, e.g., "This is mentioned in the Charaka Samhita, beta."
        6. Avoid making exaggerated claims—let the natural power of the remedies speak for themselves.
    """
}

# Active configuration using enums
ACTIVE_CONFIG = {
    "embedding": EmbeddingsTypeEnum.Mistral.value,
    "vector_store": VectorStoresEnum.Faiss.value,
    "retrieval": RetrievalTypesEnum.Ensemble.value,
    "llm": LLMTypesEnum.OpenAiLLM.value,
    "evaluation": EvaluatorTypesEnum.DeepEvalEvaluator.value,
    "reranking": RerankingTypesEnum.Groq.value,
    "chunking": ChunkingTypeEnum.Semantic.value
}

user_greetings = [
    "hi", "hello", "hey", "hi there", "good morning", "good afternoon", "good evening",
    "hey grandma", "hello grandma", "hi grandma", "hey there", "yo", "what's up?",
    "hi, i need help", "hello, can you help me?", "hi, i’m not feeling well",
    "good day", "is anyone there?", "hi, i have a question", "hello, i need a remedy",
    "hi grandma, i need your advice", "hello grandma, can you help me?", "hey grandma, i feel sick",
    "hi grandma, i need a remedy", "hello, feeling unwell today", "hey grandma, not feeling great",
    "thank you", "thanks grandma", "you are the best", "thank you grandma", "thanks a lot"
]
