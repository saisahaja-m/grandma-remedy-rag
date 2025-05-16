import streamlit as st
from openai import OpenAI
import requests, json, os
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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

def classify_query_with_openai_functions(user_input: str, chat_history: str) -> Dict:
    functions = [
        {
            "name": "classify_query",
            "description": "Classify the user query into add, update, delete, list or other task-related actions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["add", "update", "delete", "list", "other"]
                    },
                    "task_id": {"type": "integer"},
                    "description": {"type": "string"},
                    "new_description": {"type": "string"},
                    "new_status": {"type": "string"}
                },
                "required": ["query_type"]
            }
        }
    ]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4-0613",
        "messages": [
            {
                "role": "system",
                "content": """
                You are a task assistant that classifies user input into task operations: add, update, delete, list, or other.

                Rules:
                - If the input contains phrases like:
                    - "should go to dance class"
                    - "buy fruits"
                    - "go to shopping"
                  Then classify as: **add**
                
                - If the input contains phrases like:
                    - "came from dance class"
                    - "done with buying fruits"
                    - "done with shopping"
                  Then classify as: **update**
                
                - If the input contains the words "remove" or "delete" along with a task description, then classify as: **delete**
                
                - If the input asks about existing tasks (e.g., "What are my tasks?", "Show tasks", "List all tasks"), then classify as: **list**
                
                - Anything else: **other**
                
                Examples:
                - "I should go to dance class" => add
                - "Buy fruits later today" => add
                - "Came from dance class just now" => update
                - "Done with shopping" => update
                - "Delete the shopping task" => delete
                - "Remove buy fruits from my list" => delete
                - "What tasks do I have?" => list
                """
            },
            {"role": "user", "content": f"Query: {user_input}\nChat History: {chat_history}"}
        ],
        "functions": functions,
        "function_call": {"name": "classify_query"},
        "temperature": 0.0
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        response_data = response.json()
        function_call = response_data["choices"][0]["message"]["function_call"]
        return json.loads(function_call["arguments"])

    except Exception as e:
        st.error(f"Error with classification: {e}")
        return {"query_type": "other"}

def call_function(tracker, classification):
    qtype = classification.get("query_type")

    if qtype == "add":
        description = classification.get("description", "")
        if description:
            task_id = tracker.add_task(description)
            return f"✅ Task {task_id} added: {description}"
        return "⚠️ Please provide a task description."
    elif qtype == "update":
        updated = tracker.update_task(
            classification.get("task_id"),
            classification.get("new_description"),
            classification.get("new_status")
        )
        return "✅ Task updated!" if updated else "❌ Could not find or update the task."
    elif qtype == "delete":
        deleted = tracker.delete_task(classification.get("task_id"))
        return "🗑️ Task deleted." if deleted else "❌ Could not find the task to delete."
    elif qtype == "list":
        return tracker.format_tasks()
    else:
        return "🤖 I didn't understand that. Try commands like 'Add finish homework' or 'Delete task 2'."

def main():
    st.set_page_config(page_title="Task Chatbot", page_icon="✅", layout="centered")
    st.title("🧠 Task Tracking Chatbot")

    tracker = TaskTracker()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Type your task command here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        chat_history = "\n".join([m["content"] for m in st.session_state.messages if m["role"] == "user"])
        classification = classify_query_with_openai_functions(user_input, chat_history)
        reply = call_function(tracker, classification)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

if __name__ == "__main__":
    main()
