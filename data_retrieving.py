from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
import google.generativeai as genai
import streamlit as st


class SklearnRetriever(BaseRetriever, Runnable):
    def __init__(self, docs, k=5):
        self.docs = docs
        self.k = k
        self.texts = [doc.page_content for doc in docs]
        self.vectorizer = TfidfVectorizer().fit(self.texts)
        self.tfidf_matrix = self.vectorizer.transform(self.texts)

    def _retrieve(self, query: str):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k = sims.argsort()[-self.k:][::-1]
        return [self.docs[i] for i in top_k]

    def invoke(self, input: str, config=None):
        return self._retrieve(input)


@st.cache_resource
def setup_retrieval_system(_docs):
    bm25 = BM25Retriever.from_documents(_docs)
    bm25.k = 5

    sklearn_retriever = SklearnRetriever(_docs)

    ensemble = EnsembleRetriever(
        retrievers=[bm25, sklearn_retriever],
        weights=[0.3, 0.7]
    )

    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    return {"model": model, "retriever": ensemble}
