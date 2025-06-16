import tiktoken
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from rag_func.constants.config import CHUNKING, ACTIVE_CONFIG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from rag_func.constants.enums import ChunkingTypeEnum, DocProcessingEnum


class BaseChunker(ABC):
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        pass

class ManualChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            start += self.chunk_size - self.chunk_overlap

        return chunks

class RecursiveTextChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

class SentenceWindowChunker(BaseChunker):
    def __init__(self, window_size: int, window_overlap: int):
        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size
        )
        self.window_overlap = window_overlap

    def chunk_text(self, text: str) -> List[str]:
        document = Document(page_content=text)
        nodes = self.parser.get_nodes_from_documents([document])
        return [node.text for node in nodes]

class SemanticChunker(BaseChunker):
    def __init__(self):
        pass
    def chunk_text(self, text: str) -> List[str]:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai.embeddings import OpenAIEmbeddings

        text_splitter = SemanticChunker(OpenAIEmbeddings())
        docs = text_splitter.create_documents([text])
        docs = [doc.page_content for doc in docs]

        return docs


class ChunkingFactory:
    @staticmethod
    def get_chunking_strategy() -> BaseChunker:
        chunking_config = CHUNKING[ACTIVE_CONFIG["chunking"]]
        chunking_type = chunking_config["type"]
        chunk_size = chunking_config.get(DocProcessingEnum.ChunkSize.value)
        chunk_overlap = chunking_config.get(DocProcessingEnum.ChunkOverlap.value)

        chunker_classes = {
            ChunkingTypeEnum.Manual.value: lambda: ManualChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
            ChunkingTypeEnum.SentenceWindow.value: lambda: SentenceWindowChunker(
                window_size=chunking_config["max_window_size"],
                window_overlap=chunking_config["stride"]
            ),
            ChunkingTypeEnum.Recursive.value: lambda: RecursiveTextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
            ChunkingTypeEnum.Semantic.value: lambda: SemanticChunker()
        }

        chunker_class = chunker_classes.get(chunking_type)
        if chunker_class is None:
            raise ValueError(f"Unsupported chunking type: {chunking_type}")

        return chunker_class()

def get_chunking_strategy() -> BaseChunker:
    return ChunkingFactory.get_chunking_strategy()