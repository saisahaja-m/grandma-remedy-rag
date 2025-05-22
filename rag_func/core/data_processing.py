import os
import json
import re
import tiktoken
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from rag_func.constants.config import URLS, CHUNKING, ACTIVE_CONFIG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from rag_func.constants.enums import ChunkingTypeEnum, DocProcessingEnum


def get_chunking_strategy():
    chunking_config = CHUNKING[ACTIVE_CONFIG["chunking"]]
    chunking_type = chunking_config["type"]

    chunk_size = DocProcessingEnum.ChunkSize.value
    chunk_overlap = DocProcessingEnum.ChunkOverlap.value

    if chunking_type == ChunkingTypeEnum.Manual.value:
        return ManualChunker(
            chunk_size=chunking_config[chunk_size],
            chunk_overlap=chunking_config[chunk_overlap]
        )
    elif chunking_type == ChunkingTypeEnum.SentenceWindow.value:
        return SentenceWindowChunker(
            window_size=chunking_config["max_window_size"],
            window_overlap=chunking_config["stride"]
        )
    elif chunking_type == ChunkingTypeEnum.Recursive.value:
        return RecursiveTextChunker(
            chunk_size=chunking_config[chunk_size],
            chunk_overlap=chunking_config[chunk_overlap]
        )
    elif chunking_type == ChunkingTypeEnum.Semantic.value:
        return SemanticChunker(
            chunk_size=chunking_config[chunk_size],
            chunk_overlap=chunking_config[chunk_overlap]
        )
    return None


def load_and_process_documents() -> List[Document]:
    chunking_strategy = get_chunking_strategy()

    loader = WebBaseLoader(URLS)
    web_docs = loader.load()
    clean_web_docs = []

    for doc in web_docs:
        content = doc.page_content
        content = re.sub(r'(cookie|privacy|terms|copyright|navigation).*?\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'advertisement.*?\n', '', content, flags=re.IGNORECASE)

        clean_doc = Document(
            page_content=content,
            metadata={
                "source": doc.metadata.get("source"),
                "title": extract_title(content),
                "verified": True
            }
        )
        clean_web_docs.append(clean_doc)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    json_path = os.path.join(parent_dir, "data", "remidies.json")

    with open(json_path, "r") as file:
        remedy_data = json.load(file)

    remedy_docs = [
        Document(
            page_content=entry["content"],
            metadata={
                "source": "custom_remedies",
                "title": entry["title"],
                "verified": entry.get("verified", False)
            }
        ) for entry in remedy_data
    ]

    all_docs = []

    for doc in clean_web_docs + remedy_docs:
        chunks = chunking_strategy.chunk_text(doc.page_content)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    return all_docs


class ManualChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> List[str]:
        # Tokenize the text
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


class RecursiveTextChunker:

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)


class SentenceWindowChunker:

    def __init__(self, window_size: int = 2, window_overlap: int = 1):
        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size
        )

    def chunk_text(self, text: str) -> List[str]:
        document = Document(text=text, page_content="")
        nodes = self.parser.get_nodes_from_documents([document])
        return [node.text for node in nodes]


class SemanticChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _token_length(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        from flair.splitter import SegtokSentenceSplitter

        text = text.strip()
        if not text:
            return []

        splitter = SegtokSentenceSplitter()
        sentences = [s for s in splitter.split(text) if s.to_plain_string().strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence_str = sentence.to_plain_string()
            if self._token_length(current_chunk) + self._token_length(sentence_str) <= self.chunk_size:
                current_chunk += " " + sentence_str
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_str

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

def extract_title(content: str) -> str:
    lines = content.split('\n')
    for line in lines[:5]:
        if len(line) > 10 and len(line) < 100 and not line.startswith('http'):
            return line.strip()
    return "Remedy Information"