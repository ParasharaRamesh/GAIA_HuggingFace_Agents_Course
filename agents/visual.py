import os
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents import create_clean_agent_messages_hook
from agents.state import GaiaState
from tools.visual import read_image_and_encode


# Logic to format messages for multimodal LLM understanding
# This function will be used as a pre_model_hook to modify messages
# before they are sent to the LLM during the agent's internal thought process.
def _format_messages_for_multimodal_llm(state: GaiaState) -> Dict[str, Any]:
    """
    First, cleans agent messages using the common hook, then analyzes the message history,
    specifically looking for ToolMessage outputs that contain Base64 encoded image data.
    If found, it injects a new HumanMessage that explicitly includes this image data
    in a multimodal format (image_url type), so that the multimodal LLM can correctly
    "see" and process the image. This ensures the LLM interprets the image visually
    rather than as a plain text string.

    This hook helps bridge the gap between tool output (plain string) and
    multimodal LLM input requirements (structured image data).

    Args:
        messages (List[BaseMessage]): The current list of messages in the agent's state.

    Returns:
        List[BaseMessage]: The potentially modified list of messages, with Base64
                            image data appropriately formatted for a multimodal LLM.
    """
    # Call the common cleaning hook first ---
    hook = create_clean_agent_messages_hook("visual")
    input_state = hook(state)

    cleaned_messages = input_state["messages"][1:]
    new_input = input_state["input"]
    print(f"New input from supervisor to this agent is {new_input}")

    print(f"now going to add the image data in base 64 format in the messages")
    for i, message in enumerate(cleaned_messages):  # Iterate over the cleaned messages
        if isinstance(message, ToolMessage) and message.content and "base64," in message.content:
            # Assuming the tool returns a string like "Base64 image data: data:image/png;base64,..."
            # Extract the actual base64 string
            try:
                base64_data = message.content.split("data:image/")[1].split(";base64,")[1]
                mime_type = message.content.split("data:image/")[1].split(";base64,")[0]
                image_url = f"data:image/{mime_type};base64,{base64_data}"

                # Create a new HumanMessage with multimodal content
                # This new message replaces the original ToolMessage (or is inserted after it)
                # to present the image to the LLM visually.
                cleaned_messages[i] = HumanMessage(  # Modify cleaned_messages
                    content=[
                        {"type": "text", "text": "Image successfully loaded:"},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                    ]
                )
                print(f"successfully added base 64 encoded image data to {message}")
            except IndexError:
                # Handle cases where the base64 string format might not be as expected
                cleaned_messages[i] = HumanMessage(
                    content=f"Image data received but could not be parsed: {message.content}")  # Modify cleaned_messages

    return {
        "messages": cleaned_messages,
        "input": new_input
    }


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

    visual_prompt_content = ""
    try:
        with open(visual_react_prompt_path, "r", encoding="utf-8") as f:
            visual_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {visual_react_prompt_path}")
        # --- MODIFIED FALLBACK PROMPT START ---
        # Fallback to a default ReAct prompt if file not found
        # IMPORTANT: The content of this fallback is also a SYSTEM MESSAGE.
        # It mirrors the new content for visual_react_prompt.txt.
        visual_prompt_content = (
            "You are an expert Visual Analysis AI. Your primary function is to accurately interpret and describe images, "
            "answer questions about visual content, or perform visual reasoning. You have access to a tool to read image files from the local file system. "
            "Follow the ReAct pattern carefully.\n\n"
            "**Your Goal:** Provide a detailed and accurate analysis, description, or answer related to the visual content provided or referenced in the conversation.\n\n"
            "**Constraints:**\n- You MUST only use the tools provided. Do not invent new tools.\n"
            "- Do not make assumptions. If information is missing, use your tools or signal you are STUCK.\n"
            "- Your response should directly address the user's query regarding the image(s).\n\n"
            "**Important Considerations and Limitations:**\n"
            "- You DO NOT have the capability to perform video analysis. If you are tasked with analyzing a video, you must acknowledge this limitation. In such a scenario, you are permitted to either:\n"
            "    - Attempt to make an educated guess or provide a speculative answer based *only* on the textual context provided, explicitly stating it's a guess.\n"
            "    - Simply state that you cannot perform video analysis and provide no further response.\n\n"
            "**Final Answer Format:**\n"
            "- **Successful Completion:** When you have successfully completed the visual analysis task, "
            "provide it clearly in the format: 'Final Answer: [A concise summary of the steps taken (e.g., \"read image.png, analyzed content\"), "
            "followed by your EXACT final answer/description, which must always be a textual description or response.]'\n"
            "- **Stuck/Cannot Proceed:** If the task is unclear or you cannot make progress or complete the task with your available tools "
            "(e.g., if an image file is unreadable), you MUST clearly state: 'Final Answer: STUCK - [brief reason for being stuck and what you need]'\n\n"
            "**Available Tools:**\n{tools}\n\n**Tool Names:**\n{tool_names}\n\n"
            "**ReAct Process:**\nYou should always think step-by-step.\n"
            "1.  **Understand:** Carefully examine the Human Input. If the input mentions a local image file path (e.g., 'image.png', 'path/to/image.jpg'), "
            "your first step MUST be to use the `read_image_and_encode` tool to get the image data.\n"
            "2.  **Thought:** Always articulate your thought process. Explain your plan, what information you need, what tools you intend to use (especially for reading images), and why.\n"
            "3.  **Action:** Choose the best tool for your current Thought.\n"
            "4.  **Observation:** Review the results of your tool execution. If you get Base64 image data, you can then perform visual analysis using your inherent multimodal capabilities.\n"
            "5.  **Respond/Refine/Iterate:** Based on the observation and your visual analysis, formulate a clear, concise, and comprehensive answer or description. "
            "If you are provided with an image directly (e.g., as part of a multimodal message) or if the task is purely textual, proceed with direct analysis.\n\n"
            "Begin!"
        )

    # Construct the ChatPromptTemplate using from_messages
    react_prompt = ChatPromptTemplate.from_messages([
        # 1. System Message: This sets the agent's persona and core instructions.
        #    The content comes directly from the 'visual_prompt_content' variable,
        #    which is now expected to be ONLY the system message content.
        SystemMessage(content=visual_prompt_content),

        # 2. MessagesPlaceholder: This is where LangGraph injects the historical messages
        #    from the overall graph's 'state.messages' into the agent's prompt.
        #    Our '_format_messages_for_multimodal_llm' hook will process these messages
        #    to ensure multimodal data is correctly formatted for the LLM.
        MessagesPlaceholder(variable_name="messages"),

        # 3. Human Message: This contains the specific task delegated by the supervisor
        #    ({input}) and the agent's internal thought/action/observation history
        #    for its *current* turn ({agent_scratchpad}).
        #    These two variables are dynamically filled by create_react_agent.
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])
    # Create the ReAct agent
    visual_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="visual",
        debug=True,
        state_schema=GaiaState,
        pre_model_hook=_format_messages_for_multimodal_llm
    )

    return visual_agent_runnable
