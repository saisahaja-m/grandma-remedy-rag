import streamlit as st
from datetime import datetime

class TaskTracker:
    def __init__(self):
        if "tasks" not in st.session_state:
            st.session_state.tasks = []
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def add_task(self, description):
        task_id = len(st.session_state.tasks) + 1
        task = {
            "id": task_id,
            "description": description,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "pending"
        }
        st.session_state.tasks.append(task)
        return task_id

    def update_task(self, task_id, new_description=None, new_status=None):
        for task in st.session_state.tasks:
            if task["id"] == task_id:
                if new_description:
                    task["description"] = new_description
                if new_status:
                    task["status"] = new_status
                return True
        return False

    def delete_task(self, task_id):
        before = len(st.session_state.tasks)
        st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] != task_id]
        return len(st.session_state.tasks) < before

    def get_all_tasks(self):
        return st.session_state.tasks

    def format_tasks(self):
        if not st.session_state.tasks:
            return "📭 You have no tasks yet."
        return "\n\n".join([
            f"🆔 {task['id']} | {task['status'].capitalize()} | {task['description']} (Added: {task['created_at']})"
            for task in st.session_state.tasks
        ])


def customised_user_prompt(user_query):
    task_tracker = TaskTracker()
    user_tasks = task_tracker.format_tasks()
    customised_prompt = f"""
    You are a helpful assistant that helps users manage their tasks effectively through natural conversations.

    Current tasks: {user_tasks}

    User input: {user_query}

    Your goal is to:
    1. Understand the user's intent from their input and chat history.
    2. Determine the most appropriate task management function to call (add_task, delete_task, get_all_tasks, update_task).
    3. If calling a function, provide the necessary parameters.
    4. If the user is asking to list tasks (get_all_tasks), the system will directly show the list.
    5. For other actions (add, delete, update), after the action is performed, you will provide a confirmation or a natural language summary of the action.
    6. If the user's intent is unclear or doesn't match a task management function, respond conversationally or ask for clarification.

    Always prefer accurate intent classification.
    """
    return customised_prompt


def main():
    from task_tracking.function_calling import classify_query_with_claude_functions

    st.set_page_config(page_title="Task Chatbot", page_icon="✅", layout="centered")
    st.title("Task Tracking Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Type your task command here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        current_customised_prompt = customised_user_prompt(user_query=user_input)
        assistant_response = classify_query_with_claude_functions(customised_prompt=current_customised_prompt)

        if assistant_response is None:
            assistant_response = "Sorry, I couldn't process that. Please try again."

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.write(assistant_response)


if __name__ == "__main__":
    main()
