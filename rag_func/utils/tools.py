open_ai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "handle_greeting",
                    "description": "Handle user greetings with a warm, grandmotherly response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "greeting_type": {
                                "type": "string",
                                "description": "The type of greeting detected"
                            }
                        },
                        "required": ["greeting_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reject_non_health_query",
                    "description": "Politely reject queries that are outside the health and remedies domain",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_topic": {
                                "type": "string",
                                "description": "The topic that was identified as outside the health domain"
                            }
                        },
                        "required": ["query_topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "process_health_query",
                    "description": "Process health-related queries using RAG system to provide remedies and advice",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_query": {
                                "type": "string",
                                "description": "The health-related query from the user"
                            },
                            "add_to_memory": {
                                "type": "boolean",
                                "description": "Whether to add this query to conversation memory",
                                "default": True
                            }
                        },
                        "required": ["user_query"]
                    }
                }
            }
        ]