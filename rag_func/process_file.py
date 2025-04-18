import os
import json
import re
from typing import List
import streamlit as st

from langchain.schema import Document
from langchain.document_loaders import WebBaseLoader

urls = [
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
@st.cache_resource
def load_and_process_documents() -> List[Document]:
    """Load and process documents with caching"""
    loader = WebBaseLoader(urls)
    web_docs = loader.load()
    clean_web_docs = []

    for doc in web_docs:
        content = doc.page_content
        content = re.sub(r'(cookie|privacy|terms|copyright|navigation).*?\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'advertisement.*?\n', '', content, flags=re.IGNORECASE)

        clean_doc = Document(
            page_content=content,
            metadata={
                "source": doc.metadata.get("source"), "title": extract_title(content), "verified": True
            })
        clean_web_docs.append(clean_doc)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "remidies.json")

    with open(json_path, "r") as file:
        remedy_data = json.load(file)

    remedy_docs = [
        Document(
            page_content=entry["content"],
            metadata={
                "source": "custom_remedies", "title": entry["title"], "verified": entry.get("verified", False)
            }
        ) for entry in remedy_data]

    all_docs = []

    for doc in clean_web_docs + remedy_docs:
        chunks = custom_chunk_text(doc.page_content, chunk_size=700, chunk_overlap=100)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    return all_docs

def custom_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Manually split text into chunks"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks

def extract_title(content: str) -> str:
    """Extract a title from content"""
    lines = content.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        if len(line) > 10 and len(line) < 100 and not line.startswith('http'):
            return line.strip()
    return "Remedy Information"
