
class BasePromptProvider:
    @staticmethod
    def _format_prompt(prompt, **kwargs):
        return prompt.format(**kwargs)