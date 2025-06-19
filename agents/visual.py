import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents import split_point_marker, human_message_content_template
from agents.state import AgentState
from tools.visual import read_image_and_encode


# Logic to format messages for multimodal LLM understanding
# This function will be used as a pre_model_hook to modify messages
# before they are sent to the LLM during the agent's internal thought process.
def _format_messages_for_multimodal_llm(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Analyzes the message history, specifically looking for ToolMessage outputs
    that contain Base64 encoded image data. If found, it injects a new
    HumanMessage that explicitly includes this image data in a multimodal format
    (image_url type), so that the multimodal LLM can correctly "see" and
    process the image. This ensures the LLM interprets the image visually
    rather than as a plain text string.

    This hook helps bridge the gap between tool output (plain string) and
    multimodal LLM input requirements (structured image data).

    Args:
        messages (List[BaseMessage]): The current list of messages in the agent's state.

    Returns:
        List[BaseMessage]: The potentially modified list of messages, with Base64
                           image data properly formatted for a multimodal LLM.
    """
    formatted_messages = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "read_image_and_encode":
            # Assuming the content of ToolMessage from read_image_and_encode is a Base64 string
            base64_image_data = message.content

            # Create a multimodal HumanMessage to make the LLM 'see' the image
            multimodal_human_message = HumanMessage(
                content=[
                    {"type": "text", "text": "Image content from tool:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_data}"}},
                    # Optionally, include the original tool message content if it adds context
                    # {"type": "text", "text": f"Original tool output: {base64_image_data}"}
                ]
            )
            formatted_messages.append(multimodal_human_message)
            formatted_messages.append(AIMessage(content=f"Tool output for read_image_and_encode (Image sent for visual analysis)."))
        else:
            formatted_messages.append(message)
    return formatted_messages

def create_visual_agent(llm: BaseChatModel):
    """
    Creates and returns a LangChain Runnable for visual analysis.
    This agent is a ReAct agent that can use tools, specifically
    'read_image_and_encode', to process local image files.
    The underlying LLM must be multimodal (e.g., V-JEPA2 with GPU, GPT-4o, Gemini 1.5 Pro)
    to interpret the Base64 image data after it's read by the tool.
    """
    # Define the tools this agent can use
    tools = [read_image_and_encode]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    visual_agent_prompt_path = os.path.join(prompts_dir, 'visual_agent_prompt.txt')

    visual_prompt_content = ""
    try:
        with open(visual_agent_prompt_path, "r", encoding="utf-8") as f:
            visual_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {visual_agent_prompt_path}. Using fallback prompt content.")
        # Fallback to a default ReAct prompt if file not found, adhering to the new structure
        visual_prompt_content = (
            "You are an expert Visual Analysis AI. You have access to tools."
            "\n\nYour Goal: Analyze visual content. If a local file path is mentioned (e.g., image.png), "
            "use the 'read_image_and_encode' tool first."
            "\n\nAvailable Tools:\n{tools}\n\nTool Names:\n{tool_names}"  # Added for completeness in fallback
            "\n\nBegin!\nHuman Input: {input}\nThought:{agent_scratchpad}"
        )

    system_message_content = visual_prompt_content  # Default, will be updated if split_point_marker is found

    if split_point_marker in visual_prompt_content:
        parts = visual_prompt_content.split(split_point_marker, 1)
        system_message_content = parts[0] + "\nBegin!"  # Append 'Begin!' back to the system part
    else:
        print(f"Warning: Split marker '{split_point_marker}' not found in '{visual_agent_prompt_path}'. "
              "Ensure the file ends with exactly 'Begin!\\nHuman Input: {input}\\nThought:{agent_scratchpad}'. "
              "Using the entire file content as the system message and adding human input template explicitly."
              )

    # Construct the ChatPromptTemplate using from_messages
    visual_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content=human_message_content_template)
    ])

    # Create the ReAct agent
    visual_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=visual_prompt,
        name="visual_expert",
        debug=True,
        state_schema=AgentState,
        pre_model_hook=_format_messages_for_multimodal_llm
    )

    return visual_agent_runnable
