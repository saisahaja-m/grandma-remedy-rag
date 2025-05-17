from openai import OpenAI
import json
from task_tracking.app import TaskTracker
from task_tracking.constants import open_ai_tools, claude_tools, OPENAI_API_KEY, CLAUDE_API_KEY

def classify_query_with_openai_functions(customised_prompt: str):
    client = OpenAI(api_key=OPENAI_API_KEY)

    input_messages = [{"role": "user", "content": customised_prompt}]
    task_tracker = TaskTracker()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=input_messages,
        tools=open_ai_tools,
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
                    if new_description:
                        update_details.append(f"description to '{new_description}'")
                    if new_status:
                        update_details.append(f"status to '{new_status}'")
                    result_from_tool_execution = f"Task ID {task_id} has been updated: {', '.join(update_details)}."
                else:
                    result_from_tool_execution = f"Could not find or update task with ID {task_id}."

            elif function_name == "delete_task":
                task_id = function_args.get("task_id")
                if task_tracker.delete_task(task_id):
                    result_from_tool_execution = f"Task ID {task_id} has been successfully deleted."
                else:
                    result_from_tool_execution = f"Could not find or delete task with ID {task_id}."

            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result_from_tool_execution)
            }

            input_messages.append(tool_response)
            tool_responses.append(tool_response)

        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=input_messages
        )

        return second_response.choices[0].message.content

    elif response_message.content:
        return response_message.content

    return "I'm sorry, I encountered an issue. Please try again."


def classify_query_with_claude_functions(customised_prompt: str):
    import anthropic
    task_tracker = TaskTracker()

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    input_messages = [{"role": "user", "content": customised_prompt}]

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=input_messages,
        tools=claude_tools
    )
    tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)
    if not tool_use_block:
        return "No tool use block detected in Claude's response."

    function_name = tool_use_block.name
    tool_input = tool_use_block.input
    tool_call_id = tool_use_block.id
    result_from_tool_execution = None

    if function_name == "get_all_tasks":
        result_from_tool_execution = task_tracker.format_tasks()

    elif function_name == "add_task":
        description = tool_input.get("description")
        task_id = task_tracker.add_task(description)
        result_from_tool_execution = f"Task '{description}' (ID: {task_id}) has been successfully added."

    elif function_name == "update_task":
        task_id = tool_input.get("task_id")
        new_description = tool_input.get("new_description")
        new_status = tool_input.get("new_status")
        if task_tracker.update_task(task_id, new_description, new_status):
            update_details = []
            if new_description:
                update_details.append(f"description to '{new_description}'")
            if new_status:
                update_details.append(f"status to '{new_status}'")
            result_from_tool_execution = f"Task ID {task_id} has been updated: {', '.join(update_details)}."
        else:
            result_from_tool_execution = f"Could not find or update task with ID {task_id}."

    elif function_name == "delete_task":
        task_id = tool_input.get("task_id")
        if task_tracker.delete_task(task_id):
            result_from_tool_execution = f"Task ID {task_id} has been successfully deleted."
        else:
            result_from_tool_execution = f"Could not find or delete task with ID {task_id}."

    input_messages.append({"role": "assistant", "content": response.content})

    input_messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result_from_tool_execution}]
    })


    second_response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=input_messages,
            tools=claude_tools
        )
    text_block_str = (second_response.content[0])
    result = text_block_str.text

    return result

def classify_query_with_gemini_functions(customised_prompt: str):
    from google import genai
    from google.genai import types
    from task_tracking.constants import add_task, get_all_tasks, delete_task, update_task, GEMINI_API_KEY

    task_tracker = TaskTracker()

    tools = types.Tool(function_declarations=[add_task, get_all_tasks, delete_task, update_task])
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=["add_task", "get_all_tasks", "delete_task", "update_task"]
        )
    )

    config = types.GenerateContentConfig(
        temperature=0.0,
        tools=[tools],
        tool_config=tool_config
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [
        types.Content(role="user", parts=[types.Part(text=customised_prompt)])
    ]
    response = client.models.generate_content(model="gemini-2.0-flash", config=config, contents=contents)

    function_call = response.candidates[0].content.parts[0].function_call

    function_name = function_call.name
    tool_call_id = function_call.id
    args = function_call.args

    result_from_tool_execution = None

    if function_name == "get_all_tasks":
        result_from_tool_execution = task_tracker.format_tasks()

    elif function_name == "add_task":
        description = args.get("description")
        task_id = task_tracker.add_task(description)
        result_from_tool_execution = f"Task '{description}' (ID: {task_id}) has been successfully added."

    elif function_name == "update_task":
        task_id = args.get("task_id")
        new_description = args.get("new_description")
        new_status = args.get("new_status")
        if task_tracker.update_task(task_id, new_description, new_status):
            update_details = []
            if new_description:
                update_details.append(f"description to '{new_description}'")
            if new_status:
                update_details.append(f"status to '{new_status}'")
            result_from_tool_execution = f"Task ID {task_id} has been updated: {', '.join(update_details)}."
        else:
            result_from_tool_execution = f"Could not find or update task with ID {task_id}."

    elif function_name == "delete_task":
        task_id = args.get("task_id")
        if task_tracker.delete_task(task_id):
            result_from_tool_execution = f"Task ID {task_id} has been successfully deleted."
        else:
            result_from_tool_execution = f"Could not find or delete task with ID {task_id}."

    function_response_part = types.Part.from_function_response(
        name=function_call.name,
        response={"result": result_from_tool_execution},
    )

    contents.append(types.Content(role="model", parts=[types.Part(function_call=function_call)]))
    contents.append(types.Content(role="user", parts=[function_response_part]))

    final_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )

    return final_response.text





