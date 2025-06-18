# agents/visual_agent.py

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_react_agent, AgentExecutor

# Import the new image tool
from tools.image_tools import read_image_and_encode
# Also import any other tools it might need later, though for now it's just image_tools

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
    # The LLM here must be a multimodal LLM to understand the Base64 image data
    # that will be returned by the 'read_image_and_encode' tool.
    base_agent_executor = AgentExecutor(
        agent=create_react_agent(llm, tools, visual_prompt),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # We need a custom runnable to handle the agent's output.
    # When the agent decides to use the read_image_and_encode tool,
    # its output will contain ToolCall and then ToolMessage.
    # If the tool successfully returns base64 data, we need to ensure this
    # base64 data is formatted as a proper multimodal content block for the LLM
    # in the subsequent turns.

    # This is a critical adjustment for agents that generate multimodal content via tools.
    # The output of a ToolMessage is a string (the base64 data). We need to transform
    # the messages in the state to include this as an image_url type when sending back
    # to the LLM for reasoning.

    def _format_messages_for_multimodal_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "read_image_and_encode":
                # The content of this ToolMessage is the base64 string
                base64_image_data = msg.content
                # Assume the format is always data:image/<type>;base64,... as returned by the tool
                if base64_image_data.startswith("data:image/"):
                    formatted_messages.append(HumanMessage(
                        content=[
                            {"type": "text", "text": f"Observation from {msg.name} tool:"},
                            {"type": "image_url", "image_url": {"url": base64_image_data}}
                        ]
                    ))
                else:
                    # If tool returned an error or non-image string
                    formatted_messages.append(msg)
            else:
                formatted_messages.append(msg)
        return formatted_messages

    visual_agent_runnable = (
        RunnablePassthrough.assign(
            # 'input' still takes the last human message for the prompt
            input=lambda state: (
                next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
                if state.get("messages") else ""
            )
        )
        # We invoke the base agent executor. Its internal LLM will get the messages with images.
        | base_agent_executor.with_config(
            run_name="VisualAgentExecutor" # Optional: for better tracing
        )
        # Post-processing: We need to ensure the final output (messages from the agent)
        # handles the multimodal content correctly for the graph state.
        # LangGraph typically takes the last AI message from the agent.
        # This part might need further refinement based on specific LangGraph AgentState updates.
        # For now, base_agent_executor output messages are usually fine,
        # but if images are part of the 'observations' that need to be fed back,
        # the _format_messages_for_multimodal_llm would be integrated into the loop.
        # For simplicity, we assume the AgentExecutor handles this internally now
        # by passing raw messages to LLM and just returning final AI message.
    )

    return base_agent_executor # Return the AgentExecutor directly for simplicity in the graph node