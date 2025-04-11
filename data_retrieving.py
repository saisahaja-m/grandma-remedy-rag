from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.retrievers import SKLearnRetriever
import google.generativeai as genai
import streamlit as st
import numpy as np


@st.cache_resource
def setup_retrieval_system(_docs):
    """Setup retrieval system with scikit-learn instead of Chroma"""
    # Extract text from documents
    texts = [doc.page_content for doc in _docs]

    # Create SKLearnRetriever with TF-IDF
    sklearn_retriever = SKLearnRetriever.from_texts(
        texts=texts,
        embeddings=None,  # Will use default TF-IDF
        metadata=[doc.metadata for doc in _docs],
        k=5
    )

    # Setup BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(_docs)
    bm25_retriever.k = 5

    # Create ensemble of the two retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, sklearn_retriever],
        weights=[0.3, 0.7]
    )

    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    return {"model": model, "retriever": ensemble_retriever}