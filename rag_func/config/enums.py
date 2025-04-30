from enum import Enum

class EmbeddingsTypeEnum(Enum):
    HuggingFace = "huggingface"
    Voyageai = "voyageai"
    Cohere = "cohere"


class InputTypesEnum(Enum):
    SearchDocument = "search_document"
    SearchQuery = "search_query"


class ChunkingTypeEnum(Enum):
    Manual = "manual"
    SentenceWindow = "sentence_window"
    Recursive = "recursive"
    Semantic = "semantic"


class DocProcessingEnum(Enum):
    ChunkSize = "chunk_size"
    ChunkOverlap = "chunk_overlap"


class LLMTypesEnum(Enum):
    GeminiLLM = "gemini"
    OpenAiLLM = "openai"
    GroqLLM = "groq"


class EvaluatorTypesEnum(Enum):
    RagasEvaluator = "ragas"
    TrulensEvaluator = "trulens"


class EvaluatingMetricsEnum(Enum):
    Faithfulness = "faithfulness"
    AnswerRelevancy = "answer_relevancy"
    Groundedness = "groundedness"
    ContextRelevance = "context_relevance"
    Correctness = "correctness"


class VectorStoresEnum(Enum):
    Faiss = "faiss"
    Chroma = "chroma"
    Annoy = "annoy"


class RetrievalTypesEnum(Enum):
    Vector = "vector"
    bm25 = "bm25"
    Ensemble = "ensemble"

