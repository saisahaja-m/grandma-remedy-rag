from google import genai
from google.genai import types
import os
import httpx
from task_tracking.constants import CLAUDE_API_KEY
from task_tracking.paragraph import apollo_17_summary, legal_document


import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def main():
    client = genai.Client(api_key=GEMINI_API_KEY)

    temp_file_path = os.path.join(os.path.dirname(__file__), "apollo_info.txt")

    with open(temp_file_path, "w") as f:
        f.write(apollo_17_summary)

    document = client.files.upload(
        file=temp_file_path)

    model_name = "gemini-2.0-flash-001"
    system_instruction = "You are an expert analyzing transcripts."

    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            system_instruction=system_instruction,
            contents=[document],
        )
    )
    print(f'{cache=}')

    model_name = "gemini-2.0-flash-001"
    system_instruction = "You are an expert analyzing historical text."

    response = client.models.generate_content(
      model=model_name,
      contents=apollo_17_summary,
      system_instruction=system_instruction)

    print(f'{response.usage_metadata=}')
    print('\n\n', response.text)


# def main():
#     import anthropic
#     client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
#
#     system = [{"type": "text",
#                 "text": "You are an AI assistant tasked with analyzing legal documents."},
#             {"type": "text",
#              "text": f"Here is the full text of a complex legal agreement:{legal_document}",
#              "cache_control": {"type": "ephemeral"}}]
#
#     response = client.messages.create(
#         model="claude-3-7-sonnet-20250219",
#         max_tokens=1024,
#         system=system,
#         messages=[{"role": "user", "content": "What are the key terms and conditions in this agreement?"}]
#     )
#     print(response.content[0].text)

if __name__ == "__main__":
    main()
