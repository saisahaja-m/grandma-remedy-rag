def rerank_with_gemini(query, documents, model, top_k=5):
    reranked = []
    for doc in documents:
        prompt = f"""You are a helpful assistant. Score the relevance of the following context to the user's query on a scale from 0 to 1.
        
        Query: {query}

        Context:
        {doc.page_content}

        Score:"""

        try:
            score_response = model.generate_content(prompt)
            score_text = score_response.text.strip()
            score = float(score_text.split()[0])  # get the first number
            reranked.append((doc, score))
        except Exception as e:
            print(f"Error scoring doc: {e}")
            reranked.append((doc, 0.0))

    # Sort by score descending and return top_k documents
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]
