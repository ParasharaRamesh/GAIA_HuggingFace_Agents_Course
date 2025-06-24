# agents/generic_agent.py

import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents import create_clean_agent_messages_hook
from agents.state import *
from tools.search_tools import web_search, web_scraper


def create_generic_agent(llm: BaseChatModel):
    """
    Creates and returns a LangChain ReAct agent (Runnable) for generic tasks.
    This agent has access to web search and web scraping tools and can
    attempt to guess or "hallucinate" with context if it cannot find a direct answer.
    """
    tools = [web_search, web_scraper]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    generic_react_prompt_path = os.path.join(prompts_dir, 'generic_react_prompt.txt')

    with open(generic_react_prompt_path, "r", encoding="utf-8") as f:
        react_prompt_content = f.read()

    # Construct the ChatPromptTemplate using from_messages
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

    generic_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="generic",
        debug=True,
        state_schema=SubAgentState,
        # pre_model_hook=create_clean_agent_messages_hook("generic") #TODO.x need to change this possibly
    )

    return generic_agent_runnable
