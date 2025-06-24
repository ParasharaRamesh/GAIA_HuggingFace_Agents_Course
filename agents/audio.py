# agents/audio_agent.py

import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import *
# Import tools from tools/audio.py
from tools.audio_tools import transcribe_audio, get_youtube_transcript


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
    with open(audio_react_prompt_path, "r", encoding="utf-8") as f:
        react_prompt_content = f.read()

    react_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=react_prompt_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])

    audio_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="audio",
        debug=True,
        state_schema=SubAgentState
    )
    return audio_agent_runnable
