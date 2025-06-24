# agents/orchestrator.py
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

import os
from langchain_core.language_models import BaseChatModel
from tools.orchestrator_tools import *

def create_orchestrator_agent(orchestrator_llm: BaseChatModel):
    """
    Creates and returns a LangChain ReAct orchestrator agent (Runnable).
    This agent's role is to analyze the user's request and delegate tasks
    to specialized sub-agents using specific "delegation tools".
    """
    # 1. Define the tools available to the Orchestrator
    # These are the "delegation tools" that instruct the orchestrator how to route
    tools = [
        delegate_to_generic_agent,
        delegate_to_researcher_agent,
        delegate_to_audio_agent,
        delegate_to_visual_agent,
        provide_final_answer
    ]

    # 2. Load the orchestrator prompt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    orchestrator_prompt_path = os.path.join(prompts_dir, 'orchestrator_prompt.txt')

    with open(orchestrator_prompt_path, "r", encoding="utf-8") as f:
        orchestrator_prompt_content = f.read()

    # 3. Create the ChatPromptTemplate using from_messages
    orchestrator_prompt = ChatPromptTemplate.from_messages([
        # 1. System Message: Sets the orchestrator's persona and core instructions.
        SystemMessage(content=orchestrator_prompt_content),

        # 2. MessagesPlaceholder: LangGraph injects historical messages here.
        #    We'll use a pre-model hook to manage these for the orchestrator.
        MessagesPlaceholder(variable_name="messages"),

        # 3. Human Message: Contains the initial input and orchestrator's scratchpad.
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])

    # 4. Create the ReAct orchestrator agent executor
    orchestrator_agent_runnable = create_react_agent(
        model=orchestrator_llm,
        tools=tools,
        prompt=orchestrator_prompt,
        name="orchestrator",
        debug=True,
        state_schema=GaiaState
    )

    return orchestrator_agent_runnable