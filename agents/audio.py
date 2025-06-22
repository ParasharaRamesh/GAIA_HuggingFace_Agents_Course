# agents/audio_agent.py

import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from agents import create_clean_agent_messages_hook
from agents.state import GaiaState
# Import tools from tools/audio.py
from tools.audio import transcribe_audio, get_youtube_transcript


def create_audio_agent(llm: BaseChatModel):
    """
    Creates and returns a LangChain ReAct agent (Runnable) for audio processing tasks.
    This agent uses tools to transcribe audio or get YouTube transcripts.
    """
    # Expose relevant tools to the LLM.
    tools = [transcribe_audio, get_youtube_transcript]

    # ReAct prompt path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')  # Assumes 'agents' folder
    audio_react_prompt_path = os.path.join(prompts_dir, 'audio_react_prompt.txt')

    # load prompts
    react_prompt_content = ""
    try:
        with open(audio_react_prompt_path, "r", encoding="utf-8") as f:
            react_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {audio_react_prompt_path}")
        # --- MODIFIED FALLBACK PROMPT START ---
        # Fallback to a default prompt if file not found
        # IMPORTANT: The content of this fallback is also a SYSTEM MESSAGE.
        # It mirrors the new content for audio_react_prompt.txt.
        react_prompt_content = (
            "You are an expert Audio Analysis AI with advanced capabilities in processing and analyzing audio content, "
            "including transcribing speech and extracting information from audio sources like YouTube videos. "
            "Your primary function is to accurately and concisely fulfill audio-related tasks by utilizing the tools provided to you. Follow the ReAct pattern carefully.\n\n"
            "**Your Goal:** Accurately and concisely complete the delegated audio task. "
            "The task you receive may or may not contain multiple sub-tasks. You should use any provided sub-tasks as a high-level guide for your approach, "
            "but your ultimate priority is to successfully complete the overall task, even if small deviations from the suggested sub-task plan are necessary.\n\n"
            "**Constraints:**\n- You MUST only use the tools provided. Do not invent new tools.\n"
            "- Do not make assumptions. If information is missing, use your tools or signal you are STUCK.\n\n"
            "**Final Answer Format:**\n"
            "- **Successful Completion:** When you have a final answer for the current sub-task, provide it clearly in the format: 'Final Answer: [A concise summary of the steps taken to solve this task, followed by your EXACT final answer to the task.]'\n"
            "- **Stuck/Cannot Proceed:** If you encounter a situation where you cannot make progress or complete the task with your available tools, "
            "you MUST clearly state: 'Final Answer: STUCK - [brief reason for being stuck and what you need]'\n\n"
            "**Available Tools:**\n{tools}\n\n**Tool Names:**\n{tool_names}\n\n"
            "**ReAct Process:**\nYou should always think step-by-step.\n"
            "Your response MUST follow the Thought/Action/Action Input/Observation/Final Answer pattern.\n\nBegin!"
        )

    react_prompt = ChatPromptTemplate.from_messages([
        # 1. System Message: This sets the agent's persona and core instructions.
        #    The content comes directly from the 'react_prompt_content' variable,
        #    which is now expected to be ONLY the system message content.
        SystemMessage(content=react_prompt_content),

        # 2. MessagesPlaceholder: This is where LangGraph injects the historical messages
        #    from the overall graph's 'state.messages' into the agent's prompt.
        #    Our '_clean_agent_messages_hook' will process these messages to ensure
        #    only relevant ones (for *this* agent's current ReAct cycle) reach the LLM.
        MessagesPlaceholder(variable_name="messages"),

        # 3. Human Message: This contains the specific task delegated by the supervisor
        #    ({input}) and the agent's internal thought/action/observation history
        #    for its *current* turn ({agent_scratchpad}).
        #    These two variables are dynamically filled by create_react_agent.
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])

    audio_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="audio",  # Ensure consistent naming
        debug=True,
        state_schema=GaiaState,
        pre_model_hook = create_clean_agent_messages_hook("audio")
    )
    return audio_agent_runnable
