import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents.state import *
from tools.interpreter_tools import read_file, write_file, run_shell_command, run_python_script
from tools.search_tools import web_search, web_scraper

def create_code_agent(llm: BaseChatModel):
    """
    Creates and returns a LangChain ReAct agent (Runnable) for code execution tasks.
    This agent uses tools for filesystem interaction and code execution.
    """
    # Define the tools available to the CodeAgent
    tools = [
        read_file,
        write_file,
        run_shell_command,
        run_python_script,
        web_search,
        web_scraper
    ]

    # Construct the path to the prompt template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    code_react_prompt_path = os.path.join(prompts_dir, 'interpreter_react_prompt.txt')

    # Load prompt content
    with open(code_react_prompt_path, "r", encoding="utf-8") as f:
        react_prompt_content = f.read()

    # Construct the ChatPromptTemplate using from_messages
    react_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=react_prompt_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{input}\nThought:{agent_scratchpad}")
    ])

    # Create the ReAct agent executor directly
    code_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="code",
        debug=True,
        state_schema=SubAgentState
    )
    return code_agent_runnable
