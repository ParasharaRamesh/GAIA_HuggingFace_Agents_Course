import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent, AgentState

from agents.state import SubAgentState
from tools.search_tools import web_search, wikipedia_search, arxiv_search, web_scraper

def create_researcher_agent(llm: BaseChatModel):
    """
    Creates and returns a LangChain ReAct agent (Runnable) for conducting research.
    This agent uses various web search and scraping tools.
    """
    # Expose relevant tools to the LLM.
    tools = [web_search, wikipedia_search, arxiv_search, web_scraper]

    # prompt path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    researcher_react_prompt_path = os.path.join(prompts_dir, 'researcher_react_prompt.txt')

    # load prompts
    with open(researcher_react_prompt_path, "r", encoding="utf-8") as f:
        react_prompt_content = f.read()

    react_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=react_prompt_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])

    # Create the ReAct agent executor directly
    researcher_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="researcher",
        debug=True,
        state_schema=SubAgentState
    )
    return researcher_agent_runnable
