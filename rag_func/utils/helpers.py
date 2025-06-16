def format_chat_history(messages, max_tokens=1000):
    history_lines = []
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i].get("content", "")
        assistant_msg = messages[i + 1].get("content", "")
        history_lines.append(f"User: {user_msg}\nGrandma: {assistant_msg}\n")

    if len(messages) % 2 != 0:
        last_user_msg = messages[-1].get("content", "")
        history_lines.append(f"User: {last_user_msg}\n")

    history = "".join(history_lines)

    return history[-max_tokens:]

def format_context_from_docs(docs):
    return "\n\n".join([
        f"SOURCE: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    ])


def create_system_prompt(query, chat_history, context, memories):
    from rag_func.prompt_providers.templates.grandma_remedy_prompt_template import GRANDMA_REMEDY_PROMPT_TEMPLATE

    template = GRANDMA_REMEDY_PROMPT_TEMPLATE
    return template.format(
        query=query,
        chat_history=chat_history,
        context=context,
        memories=memories
    )

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