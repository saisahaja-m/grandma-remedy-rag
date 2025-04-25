import os
import json
import re
from typing import List
from langchain.schema import Document
from langchain.document_loaders import WebBaseLoader
from rag_func.config.config import URLS, CHUNKING, ACTIVE_CONFIG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser


def get_chunking_strategy():
    active_chunking = ACTIVE_CONFIG.get("chunking", "default")
    chunking_config = CHUNKING.get(active_chunking, CHUNKING["semantic"])

    chunking_type = chunking_config.get("type", "semantic")

    if chunking_type == "manual":
        return ManualChunker(
            chunk_size=chunking_config.get("chunk_size", 700),
            chunk_overlap=chunking_config.get("chunk_overlap", 100)
        )
    elif chunking_type == "sentence_window":
        return SentenceWindowChunker(
            window_size=chunking_config.get("max_window_size", 5),
            window_overlap=chunking_config.get("stride", 2)
        )
    elif chunking_type == "recursive":
        return RecursiveTextChunker(
            chunk_size=chunking_config.get("chunk_size", 700),
            chunk_overlap=chunking_config.get("chunk_overlap", 150)
        )
    elif chunking_type == "semantic":
        return SemanticChunker(
            chunk_size=chunking_config.get("chunk_size"),
            chunk_overlap=chunking_config.get("chunk_overlap")
        )


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

    # Combine and chunk all documents
    all_docs = []

    for doc in clean_web_docs + remedy_docs:
        chunks = chunking_strategy.chunk_text(doc.page_content)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    return all_docs

class ManualChunker():
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap

        return chunks


class RecursiveTextChunker():

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
            window_size=window_size,
            window_overlap=window_overlap
        )

    def chunk_text(self, text: str) -> List[str]:
        document = Document(text=text)
        nodes = self.parser.get_nodes_from_documents([document])
        return [node.text for node in nodes]


class SemanticChunker():
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        from flair.splitter import SegtokSentenceSplitter

        splitter = SegtokSentenceSplitter()

        sentences = splitter.split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence.to_plain_string()) <= self.chunk_size:
                current_chunk += " " + sentence.to_plain_string()
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence.to_plain_string()

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

def extract_title(content: str) -> str:
    lines = content.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        if len(line) > 10 and len(line) < 100 and not line.startswith('http'):
            return line.strip()
    return "Remedy Information"