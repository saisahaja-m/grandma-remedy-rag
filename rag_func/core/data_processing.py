import os
import json
import re
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.document_loaders import WebBaseLoader
from rag_func.config.config import URLS, DOC_PROCESSING


def load_and_process_documents() -> List[Document]:
    # Load from web URLs
    loader = WebBaseLoader(URLS)
    web_docs = loader.load()
    clean_web_docs = []

    for doc in web_docs:
        content = doc.page_content
        # Clean up content
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

    # Load from local JSON
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
    chunk_size = DOC_PROCESSING["chunk_size"]
    chunk_overlap = DOC_PROCESSING["chunk_overlap"]

    for doc in clean_web_docs + remedy_docs:
        chunks = chunk_text(doc.page_content, chunk_size, chunk_overlap)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    return all_docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks


def extract_title(content: str) -> str:
    lines = content.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        if len(line) > 10 and len(line) < 100 and not line.startswith('http'):
            return line.strip()
    return "Remedy Information"