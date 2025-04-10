from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import google.generativeai as genai
import streamlit as st

class SklearnRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def get_relevant_documents(self, query, k=5):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_indices]

@st.cache_resource
def setup_retrieval_system(_docs):
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(_docs)
    bm25_retriever.k = 5

    # Scikit-learn vector retriever
    vector_retriever = SklearnRetriever(_docs)

    # Wrap sklearn retriever into LangChain-style retriever
    class WrapperRetriever:
        def __init__(self, retriever):
            self.retriever = retriever
        def get_relevant_documents(self, query):
            return self.retriever.get_relevant_documents(query)

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, WrapperRetriever(vector_retriever)],
        weights=[0.3, 0.7]
    )

    # Gemini model
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    return {"model": model, "retriever": ensemble_retriever}
