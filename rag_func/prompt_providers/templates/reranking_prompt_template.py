RERANKING_PROMPT_TEMPLATE = \
"""
 You are a helpful assistant. Score the relevance of the following context 
 to the user's query on a scale from 0.0 to 1.0.\n\n
 Query: {query}\n
 Context:\n{page_content}\n\n
 Score (respond with only a float from 0.0 to 1.0):
"""