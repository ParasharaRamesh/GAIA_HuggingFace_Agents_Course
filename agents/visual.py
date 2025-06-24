import os
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents.state import *
from tools.visual_tools import read_image_and_encode


def format_multimodal_messages(state: dict) -> dict:
    """
    This hook checks the last message in the state. If it's a ToolMessage
    containing Base64 image data, it reformats it into a HumanMessage
    with a proper image_url content block for the multimodal LLM to see.
    """
    messages: List[BaseMessage] = state["messages"]
    if not messages or not isinstance(messages[-1], ToolMessage):
        return

    last_message = messages[-1]
    # Check if the tool output contains a Base64 data URI
    if "data:image/" in last_message.content and ";base64," in last_message.content:
        print("  Hook: Found Base64 image in ToolMessage. Reformatting for multimodal LLM.")
        # Create a new HumanMessage with multimodal content
        new_human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "The image has been loaded successfully. You can now analyze it and answer the user's query."
                },
                {
                    "type": "image_url",
                    # The content of the tool message is the data URI itself
                    "image_url": {"url": last_message.content, "detail": "auto"}
                }
            ]
        )
        # Replace the last ToolMessage with our new HumanMessage
        return {"messages": add_messages(messages[:-1] + [new_human_message])}
    return {}


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
    visual_react_prompt_path = os.path.join(prompts_dir, 'visual_react_prompt.txt')

    with open(visual_react_prompt_path, "r", encoding="utf-8") as f:
        visual_prompt_content = f.read()

    # Construct the ChatPromptTemplate using from_messages
    react_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=visual_prompt_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])

    # Create the ReAct agent
    visual_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="visual",
        debug=True,
        state_schema=SubAgentState,
        pre_model_hook=format_multimodal_messages
    )

    return visual_agent_runnable
