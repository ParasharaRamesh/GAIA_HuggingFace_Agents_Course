# agents/audio_agent.py

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

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

    audio_agent_runnable = create_react_agent(
        model=llm,  # can be a string also
        tools=tools,
        prompt=react_prompt,
        name="audio-agent",
        debug=True
    )

    return audio_agent_runnable
