from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


open_ai_tools = [
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

claude_tools = [
        {
            "type": "custom",
            "name": "add_task",
            "description": "Adds a task to the list based on user input.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the task to add"
                    }
                },
                "required": ["description"]
            }
        },
        {
            "type": "custom",
            "name": "delete_task",
            "description": "Deletes a task from the list by its ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task to delete"
                    }
                },
                "required": ["task_id"]
            }
        },
        {
            "type": "custom",
            "name": "get_all_tasks",
            "description": "Retrieves and lists all current tasks.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "type": "custom",
            "name": "update_task",
            "description": "Updates the description or status of an existing task.",
            "input_schema": {
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
    ]
