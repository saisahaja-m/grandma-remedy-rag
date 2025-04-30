import google.generativeai as genai
import requests
from rag_func.config.config import LLM_MODELS, GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, ACTIVE_CONFIG
from rag_func.config.enums import LLMTypesEnum

genai.configure(api_key=GEMINI_API_KEY)


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

    elif model_type == LLMTypesEnum.GroqLLM.value:
        return GroqLLM(
            model_name=model_config["model_name"],
            temperature=model_config.get("temperature", 0.0)
        )

class GeminiLLM:
    """Gemini LLM implementation"""

    def __init__(self, model_name, temperature=0.2):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.temperature = temperature

    def generate_response(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text


class OpenAILLM:
    """OpenAI LLM implementation"""

    def __init__(self, model_name, temperature=0.2):
        self.model_name = model_name
        self.temperature = temperature

    def generate_response(self, prompt):
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )

        return response.json()["choices"][0]["message"]["content"]


class GroqLLM:

    def __init__(self, model_name, temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature

    def generate_response(self, prompt):
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )

        return response.json()["choices"][0]["message"]["content"]
