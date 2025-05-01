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


def create_system_prompt(query, chat_history, context):
    from rag_func.constants.config import APP_CONFIG

    template = APP_CONFIG["prompt_template"]
    return template.format(
        query=query,
        chat_history=chat_history,
        context=context
    )
