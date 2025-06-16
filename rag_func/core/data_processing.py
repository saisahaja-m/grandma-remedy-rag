import os
import json
from typing import List
from langchain.schema import Document
from rag_func.utils.helpers import extract_title_from_text
from rag_func.core.chunking import get_chunking_strategy


def load_and_process_documents() -> List[Document]:
    chunking_strategy = get_chunking_strategy()
    all_prepared_documents = []

    web_data_path = os.path.join("rag_func", "data", "web_data")
    if os.path.exists(web_data_path):
        with open(web_data_path, "r", encoding="utf-8") as file:
            web_data_content = file.read().strip()

        if web_data_content and len(web_data_content.split()) >= 15:
            title = extract_title_from_text(web_data_content, fallback_title="Ayurvedic Remedies")
            all_prepared_documents.append(Document(
                page_content=web_data_content,
                metadata={
                    "source": "web_data_file",
                    "title": title,
                    "verified": False
                }
            ))

    json_path = os.path.join("rag_func", "data", "remidies.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            remedy_data = json.load(file)

        for entry_idx, entry in enumerate(remedy_data):
            json_content = entry.get("content", "").strip()
            if not json_content or len(json_content.split()) < 5:
                continue

            json_title = entry.get("title", "").strip()
            if json_title and len(json_title) > 5 and \
                    not any(kw in json_title.lower() for kw in ['untitled', 'error', 'just a moment', 'loading']):
                title = json_title
            else:
                title = extract_title_from_text(json_content, fallback_title=f"Custom Remedy {entry_idx + 1}")

            if json_content:
                all_prepared_documents.append(Document(
                    page_content=json_content,
                    metadata={
                        "source": "custom_remedies_json",
                        "title": title,
                        "verified": entry.get("verified", False)
                    }
                ))

    final_document_chunks = []
    if chunking_strategy:
        for doc in all_prepared_documents:
            if not doc.page_content.strip():
                continue

            chunks = chunking_strategy.chunk_text(doc.page_content)
            for chunk_idx, chunk_content in enumerate(chunks):
                if chunk_content.strip():
                    chunk_metadata = doc.metadata.copy()
                    final_document_chunks.append(Document(page_content=chunk_content, metadata=chunk_metadata))

    return final_document_chunks
