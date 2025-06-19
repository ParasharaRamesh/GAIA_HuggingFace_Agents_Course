# agents/audio_agent.py

import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from agents import split_point_marker, human_message_content_template
from agents.state import AgentState
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
    full_prompt_content = ""
    try:
        with open(audio_react_prompt_path, "r", encoding="utf-8") as f:
            full_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {audio_react_prompt_path}. Using fallback prompt content.")
        # Fallback to a default prompt if file not found, adhering to the new structure
        full_prompt_content = (
            "You are an audio processing agent. Use the provided tools to transcribe audio or get YouTube transcripts."
            " Respond with your findings or 'Final Answer: ...' when done. If an error occurs, report it clearly."
            "\n\nAvailable Tools:\n{tools}\n\nTool Names:\n{tool_names}"  # Added for completeness in fallback
            "\n\nBegin!\nHuman Input: {input}\nThought:{agent_scratchpad}"
        )

    system_message_content = full_prompt_content  # Default, will be updated if split_point_marker is found

    if split_point_marker in full_prompt_content:
        parts = full_prompt_content.split(split_point_marker, 1)
        system_message_content = parts[0] + "\nBegin!"  # Append 'Begin!' back to the system part
    else:
        print(f"Warning: Split marker '{split_point_marker}' not found in '{audio_react_prompt_path}'. "
              "Ensure the file ends with exactly 'Begin!\\nHuman Input: {input}\\nThought:{agent_scratchpad}'. "
              "Using the entire file content as the system message and adding human input template explicitly."
              )

    # Construct the ChatPromptTemplate using from_messages
    react_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content=human_message_content_template)
    ])

    audio_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="audio",  # Ensure consistent naming
        debug=True,
        state_schema=AgentState
    )
    return audio_agent_runnable
