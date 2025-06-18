import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt.chat_agent_executor import create_react_agent

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
                          image outputs from tools formatted as multimodal content.
    """
    formatted_messages = []
    # Keep track if we've added an image for the current tool observation
    image_added_for_last_tool_observation = False

    for i, msg in enumerate(messages):
        # Add the original message first
        formatted_messages.append(msg)

        # Check if the current message is a ToolMessage from our image tool
        if isinstance(msg, ToolMessage) and msg.name == "read_image_and_encode":
            base64_image_data = msg.content
            # Check if the content looks like a data URI (our tool's output format)
            if base64_image_data.startswith("data:image/"):
                # If it's a valid Base64 image, inject a new HumanMessage
                # to represent this image as visual input to the LLM.
                # We place this *after* the ToolMessage, so the LLM sees the tool output
                # (the raw base64 string) and then the visual representation of that string.
                formatted_messages.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": f"Image Observation from '{msg.name}' tool:"},
                            {"type": "image_url", "image_url": {"url": base64_image_data}}
                        ]
                    )
                )
                image_added_for_last_tool_observation = True
            else:
                # If the tool returned an error or non-image string, don't format it as an image.
                # The LLM will process the plain text error message.
                image_added_for_last_tool_observation = False
        else:
            image_added_for_last_tool_observation = False
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
        print(f"Error: Prompt file not found at {visual_agent_prompt_path}")
        # Fallback to a default ReAct prompt if file not found
        visual_prompt_content = (
            "You are an expert Visual Analysis AI. You have access to tools.\n\n"
            "Your Goal: Analyze visual content. If a local file path is mentioned (e.g., image.png), "
            "use the 'read_image_and_encode' tool first.\n\n"
            "Available Tools:\n{tools}\n\nTool Names:\n{tool_names}\n\n"
            "Begin!\n\nHuman Input: {input}\nThought:{agent_scratchpad}"
        )

    visual_prompt = ChatPromptTemplate.from_template(visual_prompt_content)

    # Create the ReAct agent
    base_agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=visual_prompt,
        name="visual-agent",
        debug=True,
        pre_model_hook=_format_messages_for_multimodal_llm
    )

    return base_agent_executor
