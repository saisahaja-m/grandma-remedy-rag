from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import google.generativeai as genai
import streamlit as st

@st.cache_resource
def setup_retrieval_system(_docs, embedding_model_name="all-MiniLM-L6-v2"):
    """Setup retrieval system with caching"""
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_documents(_docs, embedding_model)

    bm25_retriever = BM25Retriever.from_documents(_docs)
    bm25_retriever.k = 5

    vector_retriever = db.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    return {"model": model, "retriever": ensemble_retriever, "vector_store": db}