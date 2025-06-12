import google.generativeai as genai
import os
from abc import ABC, abstractmethod
from rag_func.constants.config import LLM_MODELS, ACTIVE_CONFIG
from rag_func.constants.enums import LLMTypesEnum
from anthropic import Anthropic
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.2):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.temperature = temperature

    def generate_response(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

class GroqLLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

class ClaudeLLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=4096
        )
        return response.content[0].text

class LLMFactory:
    @staticmethod
    def get_llm_model() -> BaseLLM:
        model_config = LLM_MODELS[ACTIVE_CONFIG["llm"]]
        model_type = model_config["type"]
        model_name = model_config["model_name"]
        temperature = model_config.get("temperature", 0.2)

        llm_classes = {
            LLMTypesEnum.GeminiLLM.value: GeminiLLM,
            LLMTypesEnum.OpenAiLLM.value: OpenAILLM,
            LLMTypesEnum.ClaudeLLM.value: ClaudeLLM,
            LLMTypesEnum.GroqLLM.value: GroqLLM
        }

        llm_class = llm_classes.get(model_type)
        if llm_class is None:
            raise ValueError(f"Unsupported LLM type: {model_type}")

        return llm_class(model_name=model_name, temperature=temperature)

def get_llm_model() -> BaseLLM:
    return LLMFactory.get_llm_model()
