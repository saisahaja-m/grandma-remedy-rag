from enum import Enum

class EmbeddingsTypeEnum(Enum):
    HuggingFace = "huggingface"
    Voyageai = "voyageai"
    Cohere = "cohere"
    Mistral = "mistral"


class InputTypesEnum(Enum):
    SearchDocument = "search_document"
    SearchQuery = "search_query"


class ChunkingTypeEnum(Enum):
    Manual = "manual"
    SentenceWindow = "sentence_window"
    Recursive = "recursive"
    Semantic = "semantic"
    Agentic = "agentic"


class DocProcessingEnum(Enum):
    ChunkSize = "chunk_size"
    ChunkOverlap = "chunk_overlap"


class LLMTypesEnum(Enum):
    GeminiLLM = "gemini"
    OpenAiLLM = "openai"
    ClaudeLLM = "claude"
    GroqLLM = "groq"


class EvaluatorTypesEnum(Enum):
    RagasEvaluator = "ragas"
    TrulensEvaluator = "trulens"
    DeepEvalEvaluator = "deepeval"
    Custom = "custom"


class EvaluatingMetricsEnum(Enum):
    Faithfulness = "faithfulness"
    AnswerRelevancy = "answer_relevancy"
    Groundedness = "groundedness"
    ContextRelevance = "context_relevance"
    Correctness = "correctness"
    ContextRecall = "context_recall"


class VectorStoresEnum(Enum):
    Faiss = "faiss"
    Chroma = "chroma"
    Annoy = "annoy"


class RetrievalTypesEnum(Enum):
    Faiss = "faiss"
    Annoy = "annoy"
    Chroma = "chroma"


class RerankingTypesEnum(Enum):
    Groq = "groq"
    Cohere = "cohere"
    Jina = "jina"
    Voyageai = "voyageai"


class VoyageaiEmbeddingModels(Enum):
    Voyage3 = "voyage-3-large"
    Voyage35 = "voyage-3.5"


class CohereEmbeddingModels(Enum):
    Embed4 = "embed-v4.0"
    EmbedEnglish3 = "embed-english-v3.0"


class CohereRerankingModels(Enum):
    Rerank35 = "rerank-v3.5"
    RerankEnglish3 = "rerank-english-v3.0"


