# agents/generic_agent.py

import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent


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
        SystemMessage(content=react_prompt_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])

    generic_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="generic",
        debug=True,
        state_schema=SubAgentState
    )

    return generic_agent_runnable
