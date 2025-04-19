import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# URLs for data sources
URLS = [
    "https://www.healthline.com/nutrition/12-foods-that-help-digestion",
    "https://www.lybrate.com/topic/home-remedies-for-digestion",
    "https://www.stylecraze.com/articles/home-remedies-for-hair-fall/",
    "https://www.healthline.com/health/home-remedies#takeaway",
    "https://www.hopkinsmedicine.org/health/wellness-and-prevention/natural-sleep-aids-home-remedies-to-help-you-sleep",
    "https://www.healthline.com/health/dental-and-oral-health/home-remedies-for-toothache",
    "https://www.dhconcepts.com/7-home-remedies-for-dental-issues/",
    "https://www.healthline.com/health/pain-relief/knee-pain-home-remedies",
    "https://www.healthline.com/recipes/cozy-dinner-ideas"
]

# Document processing config
DOC_PROCESSING = {
    "chunk_size": 700,
    "chunk_overlap": 100
}

# Embedding model configurations
EMBEDDING_MODELS = {
    "default": {
        "type": "huggingface",
        "model_name": "all-MiniLM-L6-v2"
    },
    "openai": {
        "type": "openai",
        "model_name": "text-embedding-ada-002"
    },
    "cohere": {
        "type": "cohere",
        "model_name": "embed-english-v3.0"
    }
}

# Vector store configurations
VECTOR_STORES = {
    "default": {
        "type": "faiss"
    },
    "chroma": {
        "type": "chroma",
        "persist_directory": "./chroma_db"
    }
}

# Retrieval configurations
RETRIEVAL = {
    "default": {
        "type": "ensemble",
        "retrievers": ["bm25", "vector"],
        "weights": [0.3, 0.7],
        "k": 5
    },
    "vector_only": {
        "type": "vector",
        "k": 5
    },
    "bm25_only": {
        "type": "bm25",
        "k": 5
    }
}

# LLM configurations
LLM_MODELS = {
    "default": {
        "type": "gemini",
        "model_name": "gemini-2.0-flash-exp",
        "temperature": 0.2
    },
    "openai": {
        "type": "openai",
        "model_name": "gpt-4-turbo",
        "temperature": 0.2
    },
    "groq": {
        "type": "groq",
        "model_name": "llama3-8b-8192",
        "temperature": 0.0
    },
    "huggingface": {
        "type": "huggingface",
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": 0.2
    }
}

# Evaluation configurations
EVALUATION = {
    "default": {
        "type": "ragas",
        "metrics": ["faithfulness", "answer_relevancy", "groundedness", "context_relevance"]
    },
    "custom": {
        "type": "custom",
        "metrics": ["accuracy", "coherence"]
    }
}

# Reranking configurations
RERANKING = {
    "default": {
        "type": "groq",
        "model": "llama3-8b-8192",
        "top_k": 5
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

    INSTRUCTIONS:
        1. Only answer to the questions related to health and body
        2. Be warm and loving, use terms like beta or baccha, but never make things up.
        3. Use reliable sources, like books or websites, when mentioning remedies.
        4. Always back up your advice with references when possible.
        5. Don't make exaggerated claims—let the remedies speak for themselves.
        6. Provide references when you mention them, but in a friendly manner.
    """
}

# Active configuration - change these to switch components
ACTIVE_CONFIG = {
    "embedding": "default",
    "vector_store": "default",
    "retrieval": "default",
    "llm": "default",
    "evaluation": "default",
    "reranking": "default"
}