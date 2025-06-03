import os
import json
import re
import tiktoken
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from rag_func.constants.config import URLS, CHUNKING, ACTIVE_CONFIG, OPENAI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from rag_func.constants.enums import ChunkingTypeEnum, DocProcessingEnum


def get_chunking_strategy():
    chunking_config = CHUNKING[ACTIVE_CONFIG["chunking"]]
    chunking_type = chunking_config["type"]

    chunk_size = DocProcessingEnum.ChunkSize.value
    chunk_overlap = DocProcessingEnum.ChunkOverlap.value

    if chunking_type == ChunkingTypeEnum.Manual.value:
        return ManualChunker(
            chunk_size=chunking_config[chunk_size],
            chunk_overlap=chunking_config[chunk_overlap]
        )
    elif chunking_type == ChunkingTypeEnum.SentenceWindow.value:
        return SentenceWindowChunker(
            window_size=chunking_config["max_window_size"],
            window_overlap=chunking_config["stride"]
        )
    elif chunking_type == ChunkingTypeEnum.Recursive.value:
        return RecursiveTextChunker(
            chunk_size=chunking_config[chunk_size],
            chunk_overlap=chunking_config[chunk_overlap]
        )
    elif chunking_type == ChunkingTypeEnum.Semantic.value:
        return SemanticChunker(
            chunk_size=chunking_config[chunk_size],
            chunk_overlap=chunking_config[chunk_overlap]
        )
    elif chunking_type == ChunkingTypeEnum.Agentic.value:
        return AgenticChunker()
    return None


def extract_title_from_text(content: str, fallback_title: str = "Remedy Information") -> str:
    lines = content.split('\n')
    potential_titles = []
    for line in lines[:5]:
        cleaned_line = line.strip()
        if 10 < len(cleaned_line) < 150 and \
                not cleaned_line.lower().startswith(('http:', 'https:')) and \
                not any(kw in cleaned_line.lower() for kw in [
                    'cookie', 'privacy', 'terms', 'copyright', 'navigation', 'advertisement',
                    'subscribe', 'follow us', 'home', 'about', 'contact', 'skip to content',
                    'search', 'login', 'register', '©', 'rights reserved', 'menu', '|', '•', '»'
                ]) and \
                sum(c.isalpha() for c in cleaned_line) / (len(cleaned_line) + 1e-5) > 0.6:
            potential_titles.append(cleaned_line)

    if potential_titles:
        potential_titles.sort(key=len)
        return potential_titles[0]
    return fallback_title


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
        for i, doc in enumerate(all_prepared_documents):
            if not doc.page_content.strip():
                continue

            chunks = chunking_strategy.chunk_text(doc.page_content)
            for chunk_idx, chunk_content in enumerate(chunks):
                if chunk_content.strip():
                    chunk_metadata = doc.metadata.copy()
                    final_document_chunks.append(Document(page_content=chunk_content, metadata=chunk_metadata))

    return final_document_chunks


class ManualChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> List[str]:
        # Tokenize the text
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            start += self.chunk_size - self.chunk_overlap

        return chunks


class RecursiveTextChunker:

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)


class SentenceWindowChunker:

    def __init__(self, window_size: int = 2, window_overlap: int = 1):
        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size
        )

    def chunk_text(self, text: str) -> List[str]:
        document = Document(text=text, page_content="")
        nodes = self.parser.get_nodes_from_documents([document])
        return [node.text for node in nodes]


class SemanticChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _token_length(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        from flair.splitter import SegtokSentenceSplitter

        text = text.strip()
        if not text:
            return []

        splitter = SegtokSentenceSplitter()
        sentences = [s for s in splitter.split(text) if s.to_plain_string().strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence_str = sentence.to_plain_string()
            if self._token_length(current_chunk) + self._token_length(sentence_str) <= self.chunk_size:
                current_chunk += " " + sentence_str
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_str

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

class AgenticChunker:
    def __init__(self):
        pass
    def chunk_text(self, text):
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate
        llm = ChatOpenAI(model="gpt-4o",
                         api_key=OPENAI_API_KEY,
                         verbose=True,
                         temperature=1)
        prompt = """I am providing a document below. 
        Please split the document into chunks that maintain semantic coherence and ensure that each chunk represents a complete and meaningful unit of information. 
        Each chunk should stand alone, preserving the context and meaning without splitting key ideas across chunks. 
        Use your understanding of the content's structure, topics, and flow to identify natural breakpoints in the text. 
        Ensure that no chunk exceeds 1000 characters length, and prioritize keeping related concepts or sections together.

        Do not modify the document, just split to chunks and return them as an array of strings, where each string is one chunk of the document.
        Return the entire book not dont stop in betweek some sentences.

        Document:
        {document}
        """

        prompt_template = PromptTemplate.from_template(prompt)

        chain = prompt_template | llm

        result = chain.invoke({"document": text})

        print(result)
        return result
