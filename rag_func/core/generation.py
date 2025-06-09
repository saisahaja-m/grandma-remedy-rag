import google.generativeai as genai
import os
from rag_func.constants.config import LLM_MODELS, ACTIVE_CONFIG
from rag_func.constants.enums import LLMTypesEnum
from anthropic import Anthropic
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_llm_model():
    model_config = LLM_MODELS[ACTIVE_CONFIG["llm"]]
    model_type = model_config["type"]

    if model_type == LLMTypesEnum.GeminiLLM.value:
        return GeminiLLM(
            model_name=model_config["model_name"],
            temperature=model_config.get("temperature", 0.2)
        )

    elif model_type == LLMTypesEnum.OpenAiLLM.value:
        return OpenAILLM(
            model_name=model_config["model_name"],
            temperature=model_config.get("temperature", 0.2)
        )
    elif model_type == LLMTypesEnum.ClaudeLLM.value:
        return ClaudeLLM(
            model_name=model_config["model_name"],
            temperature=model_config.get("temperature", 0.0)
        )
    return None


class GeminiLLM:
    def __init__(self, model_name, temperature=0.2):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.temperature = temperature

    def generate_response(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text


class OpenAILLM:
    def __init__(self, model_name, temperature=0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content


class GroqLLM:

    def __init__(self, model_name, temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_response(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content


class ClaudeLLM:

    def __init__(self, model_name, temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def generate_response(self, prompt):
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=4096
        )
        return response.content[0].text
