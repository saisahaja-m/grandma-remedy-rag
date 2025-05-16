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
                "description": "Adds a task to the list based on user input.",
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
                "description": "Deletes a task from the list by its ID.",
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
                "description": "Retrieves and lists all current tasks.",
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
                "description": "Updates the description or status of an existing task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "integer",
                            "description": "ID of the task to update"
                        },
                        "new_description": {
                            "type": "string",
                            "description": "New description for the task (optional)"
                        },
                        "new_status": {
                            "type": "string",
                            "description": "New status for the task (e.g., 'pending', 'completed') (optional)"
                        }
                    },
                    "required": ["task_id"]
                }
            }
        }
    ]

    input_messages = [{"role": "user", "content": customised_prompt}]
    task_tracker = TaskTracker()

    response = client.chat.completions.create(
        model="gpt-4o",  # Using a recent model
        messages=input_messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls and response_message.tool_calls[0].function.name == "get_all_tasks":
        return task_tracker.format_tasks()

    if response_message.tool_calls:
        tool_responses = []

        input_messages.append({
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

        # Process each tool call and create corresponding tool response messages
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            result_from_tool_execution = None

            if function_name == "add_task":
                description = function_args.get("description")
                task_id = task_tracker.add_task(description)
                result_from_tool_execution = f"Task '{description}' (ID: {task_id}) has been successfully added."

            elif function_name == "update_task":
                task_id = function_args.get("task_id")
                new_description = function_args.get("new_description")
                new_status = function_args.get("new_status")
                if task_tracker.update_task(task_id, new_description, new_status):
                    update_details = []
                    if new_description: update_details.append(f"description to '{new_description}'")
                    if new_status: update_details.append(f"status to '{new_status}'")
                    result_from_tool_execution = f"Task ID {task_id} has been updated: {', '.join(update_details)}."
                else:
                    result_from_tool_execution = f"Could not find or update task with ID {task_id}."

            elif function_name == "delete_task":
                task_id = function_args.get("task_id")
                if task_tracker.delete_task(task_id):
                    result_from_tool_execution = f"Task ID {task_id} has been successfully deleted."
                else:
                    result_from_tool_execution = f"Could not find or delete task with ID {task_id}."

            # Add the tool response message for this specific tool call
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result_from_tool_execution)
            }

            input_messages.append(tool_response)
            tool_responses.append(tool_response)

        # Get the final response after all tool calls have been processed
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=input_messages
        )

        return second_response.choices[0].message.content

    elif response_message.content:
        return response_message.content

    return "I'm sorry, I encountered an issue. Please try again."


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
        assistant_response = classify_query_with_openai_functions(customised_prompt=current_customised_prompt)

        if assistant_response is None:
            assistant_response = "Sorry, I couldn't process that. Please try again."

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.write(assistant_response)


if __name__ == "__main__":
    main()
