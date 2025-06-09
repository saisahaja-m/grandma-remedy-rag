from rag_func.prompt_providers.templates.grandma_remedy_prompt_template import GRANDMA_REMEDY_PROMPT_TEMPLATE
from rag_func.prompt_providers.prompt_service.base_prompt_provider import BasePromptProvider
from rag_func.prompt_providers.templates.function_calling_prompt_template import FUNCTION_CALLING_PROMPT_TEMPLATE

class MetadatPromptProvider(BasePromptProvider):

    def get_user_prompt(self, user_input, chat_history_text, context, memories, cached):
        if cached:
            user_prompt = self._format_prompt(
                GRANDMA_REMEDY_PROMPT_TEMPLATE,
                user_input=user_input,
                chat_history_text=chat_history_text,
                context=context,
                memories=memories
            )
            prompt = [{
                "type": "text",
                "text": user_prompt,
                "cached": {'type': "emphimeral"}
            }]
            return prompt
        else:
            return self._format_prompt(
                GRANDMA_REMEDY_PROMPT_TEMPLATE,
                user_input=user_input,
                chat_history_text=chat_history_text,
                context=context,
                memories=memories
            )

class FunctionCallingPromptProvider(BasePromptProvider):

    def get_user_prompt(self, user_input, chat_history_text, memories):
        return self._format_prompt(
            FUNCTION_CALLING_PROMPT_TEMPLATE,
            user_input=user_input,
            memories=memories,
            chat_history_text=chat_history_text
        )

