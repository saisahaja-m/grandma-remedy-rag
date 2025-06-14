from typing import List, Dict, Tuple
import json
from openai import OpenAI
from rag_func.utils.helpers import format_context_from_docs
from rag_func.utils.tools import open_ai_tools
from rag_func.prompt_providers.prompt_service.prompt_provider import ResponsePromptProvider

class RAGAssistantWithFunctions:
    def __init__(self, openai_api_key: str, rag_system):
        self.client = OpenAI(api_key=openai_api_key)
        self.rag_system = rag_system
        self.user_greetings = ["hello", "hi", "namaste", "hey", "greetings"]
        self.tools = open_ai_tools

    def classify_and_handle_query(self, user_input: str, chat_history: List[Dict], memories: List[str]) -> Tuple[str, List]:
        from rag_func.prompt_providers.prompt_service.prompt_provider import FunctionCallingPromptProvider
        prompt_provider = FunctionCallingPromptProvider()
        system_prompt = prompt_provider.get_user_prompt(user_input=user_input, chat_history_text=chat_history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            return self._handle_function_calls(
                response_message, messages, user_input, memories, chat_history
            )

        return "I'm sorry, I encountered an issue. Please try again.", []

    def _handle_function_calls(
            self, response_message, messages: List[Dict],user_input: str, memories: List[str],
            chat_history: List[Dict]) -> Tuple[str, List]:

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            } for tool_call in response_message.tool_calls]
        })

        response = ""
        reranked_docs = []

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "handle_greeting":
                response, reranked_docs = self._handle_greeting()

            elif function_name == "reject_non_health_query":
                response, reranked_docs = self._reject_non_health_query(function_args)

            elif function_name == "process_health_query":
                response, reranked_docs = self._process_health_query(
                    function_args, user_input, memories, chat_history
                )

        return response, reranked_docs

    def _handle_greeting(self) -> Tuple[str, List]:
        response = (
            "Namaste, beta! How wonderful to hear from you. What can Grandma help you with today? "
            "I have so many ancient remedies passed down through generations, "
            "I'm sure we can find something to soothe your woes!"
        )
        return response, []

    def _reject_non_health_query(self, args: Dict) -> Tuple[str, List]:
        topic = args.get("query_topic", "that topic")
        response = (
            f"Oh dear, Grandma's always happy to help with your aches, sniffles, and remedies passed down "
            f"through the years. But when it comes to things outside of health—like {topic}—"
            f"I'm afraid this old mind doesn't stretch quite that far! Now, if you've got a health "
            f"worry or a home remedy question, come sit beside me and ask away."
        )
        return response, []

    def _process_health_query(self, args: Dict, user_input: str,
                             memories: List[str], chat_history: List[Dict]) -> Tuple[str, List]:
        query = args.get("user_query", user_input)
        add_to_memory = args.get("add_to_memory", True)

        if add_to_memory:
            memories.append(user_input)

        response_generator, reranked_docs = process_query_with_rag(
            rag_system=self.rag_system, user_input=query, memories=memories, chat_history=chat_history
        )

        if hasattr(response_generator, '__iter__'):
            full_response = ''.join(response_generator)
        else:
            full_response = str(response_generator)

        return full_response, reranked_docs

def process_query_with_rag(rag_system, user_input, memories, chat_history):
    embeddings = rag_system["embeddings"]

    relevant_docs = rag_system["retriever"].get_relevant_documents(user_input, embeddings)
    docs = [doc for doc in relevant_docs if doc.page_content.strip()]

    reranked_docs = rag_system["reranker"].rerank(user_input, docs)
    context = format_context_from_docs(reranked_docs)

    prompt_provider = ResponsePromptProvider()

    prompt = prompt_provider.get_user_prompt(user_input=user_input, chat_history_text=chat_history,
                                             context=context, memories=memories, cached=False)

    response = rag_system["llm"].generate_response(prompt)

    return response, reranked_docs