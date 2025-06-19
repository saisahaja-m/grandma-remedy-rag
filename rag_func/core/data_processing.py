import os
import json
from typing import List
from langchain.schema import Document
from rag_func.core.chunking import get_chunking_strategy
from rag_func.core.generation import GroqLLM

def generate_title_with_llm(content: str) -> str:
    llm = GroqLLM(model_name="llama3-8b-8192")
    prompt = (
        "Generate a concise and meaningful title (5-10 words) for the following text content. "
        "The title should summarize the key topic or purpose of the content"
        "Just give the title without any other text:\n\n"
        f"{content[:1000]}"
    )

    title = llm.generate_response(prompt)
    return title.strip() if title.strip() else "Untitled Chunk"


def load_and_process_documents() -> List[Document]:
    chunking_strategy = get_chunking_strategy()
    all_entries = []

    web_data_path = os.path.join("rag_func", "data", "web_data")
    if os.path.exists(web_data_path):
        with open(web_data_path, "r", encoding="utf-8") as file:
            web_data_content = file.read().strip()
            if web_data_content and len(web_data_content.split()) >= 15:
                all_entries.append(web_data_content)

    json_path = os.path.join("rag_func", "data", "remidies.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            remedy_data = json.load(file)
            for entry in remedy_data:
                json_content = entry.get("content", "").strip()
                if json_content and len(json_content.split()) >= 5:
                    all_entries.append(json_content)

    combined_text = "\n\n".join(all_entries)

    final_document_chunks = []
    if chunking_strategy and combined_text.strip():
        chunks = chunking_strategy.chunk_text(combined_text)
        for chunk_idx, chunk_content in enumerate(chunks):
            if chunk_content.strip():
                title = generate_title_with_llm(chunk_content)
                final_document_chunks.append(Document(
                    page_content=chunk_content,
                    metadata={
                        "source": "combined_data",
                        "title": title,
                        "verified": False
                    }
                ))

    return final_document_chunks