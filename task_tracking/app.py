import streamlit as st
from openai import OpenAI
import json, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class TaskTracker:
    def __init__(self):
        # Initialize session state variables if they don't exist
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


def classify_query_with_openai_functions(customised_prompt: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_task",
                "description": "Adds task to list that is given by user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of the task to add"
                        }
                    },
                    "required": ["description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_task",
                "description": "Deletes a task from the list by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "integer",
                            "description": "ID of the task to delete"
                        }
                    },
                    "required": ["task_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_all_tasks",
                "description": "Gets all the list of tasks that user have",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_task",
                "description": "Updates the description or status of a task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "integer",
                            "description": "ID of the task to update"
                        },
                        "new_description": {
                            "type": "string",
                            "description": "New description for the task"
                        },
                        "new_status": {
                            "type": "string",
                            "description": "New status for the task (e.g., 'pending', 'completed')"
                        }
                    },
                    "required": ["task_id"]
                }
            }
        }
    ]

    input_messages = [{"role": "user", "content": customised_prompt}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=input_messages,
        tools=tools,
        tool_choice="auto"
    )

    task_tracker = TaskTracker()
    result = None
    response_message = response.choices[0].message

    # Check if there's a tool call in the response
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        if function_name == "add_task":
            description = function_args.get("description")
            result = task_tracker.add_task(description)

        elif function_name == "update_task":
            task_id = function_args.get("task_id")
            new_description = function_args.get("new_description")
            new_status = function_args.get("new_status")
            result = task_tracker.update_task(task_id, new_description, new_status)

        elif function_name == "delete_task":
            task_id = function_args.get("task_id")
            result = task_tracker.delete_task(task_id)

        elif function_name == "get_all_tasks":
            tasks = task_tracker.get_all_tasks()
            result = task_tracker.format_tasks()

        # Add the assistant's response and function call to messages
        input_messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": tool_call.function.arguments
                    }
                }
            ],
            "content": None
        })

        # Add the function response
        input_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

        # Get the final response
        second_response = client.chat.completions.create(
            model="gpt-4",
            messages=input_messages
        )

        return second_response.choices[0].message.content

    return response_message.content


def customised_user_prompt(chat_history, user_query):
    task_tracker = TaskTracker()
    user_tasks = task_tracker.format_tasks()

    customised_prompt = f"""
    You are a helpful assistant that helps users manage their tasks effectively through natural conversations.

    Current tasks: {user_tasks}

    Chat history: {chat_history}

    User input: {user_query}

    Your goal is to:
    1. Understand the user's intent from their input and chat history
    2. Determine the most appropriate task management function to call
    3. Respond in a helpful, conversational way

    For add_task: Extract the task description and add it to the list
    For update_task: Identify which task to update and what changes to make
    For delete_task: Identify which task ID to delete
    For get_all_tasks: Simply return the current list of tasks

    Always prefer accurate intent classification over guessing. If you're unsure, default to asking the user for clarification.
    """
    return customised_prompt


def main():
    st.set_page_config(page_title="Task Chatbot", page_icon="✅", layout="centered")
    st.title("Task Tracking Chatbot")

    task_tracker = TaskTracker()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Type your task command here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])

        customised_prompt = customised_user_prompt(chat_history=chat_history, user_query=user_input)
        assistant_response = classify_query_with_openai_functions(customised_prompt)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.write(assistant_response)


if __name__ == "__main__":
    main()