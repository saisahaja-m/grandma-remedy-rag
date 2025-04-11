from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import BaseRetriever, Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import streamlit as st
import numpy as np
from typing import List, Dict, Any


# Create a custom scikit-learn based retriever
class CustomSKLearnRetriever(BaseRetriever):
    documents: List[Document]  # Define as class field
    k: int  # Define as class field

    def __init__(self, documents: List[Document], k: int = 5):
        """Initialize with documents and retrieval count."""
        self.documents = documents
        self.k = k
        self.texts = [doc.page_content for doc in documents]

        # Create and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query."""
        # Vectorize the query and calculate similarity
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()

        # Get the indices of the top k most similar documents
        top_k_indices = similarities.argsort()[-self.k:][::-1]

        # Return the top k documents
        return [self.documents[i] for i in top_k_indices]


@st.cache_resource
def setup_retrieval_system(_docs):
    """Setup retrieval system with scikit-learn instead of Chroma"""
    # Create our custom sklearn-based retriever
    sklearn_retriever = CustomSKLearnRetriever(documents=_docs, k=5)

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