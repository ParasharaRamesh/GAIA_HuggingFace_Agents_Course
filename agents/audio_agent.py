# agents/audio_agent.py

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.agents import AgentFinish, AgentAction  # Keep AgentAction for internal agent parsing
from langchain_core.messages import HumanMessage, AIMessage, \
    ToolMessage  # Necessary for agent's internal thought process

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
        # Fallback to a default prompt if file not found
        react_prompt_content = "You are an audio processing agent. Use the provided tools to transcribe audio or get YouTube transcripts. Respond with your findings or 'Final Answer: ...' when done. If an error occurs, report it clearly."

    react_prompt = ChatPromptTemplate.from_template(react_prompt_content)

    # Create the ReAct agent executor directly
    # The `create_react_agent` function returns a Runnable that can be used as a node.
    # It takes care of the internal Thought-Action-Observation loop.
    audio_agent_runnable = AgentExecutor(
        agent=create_react_agent(
            llm,
            tools,
            react_prompt
        ),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return audio_agent_runnable
