# agents/code_agent.py

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_react_agent, AgentExecutor
from tools.interpreter import read_file, write_file, run_shell_command, run_python_script
from tools.search import web_search, web_scraper

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
    code_react_prompt_path = os.path.join(prompts_dir, 'code_react_prompt.txt')

    # Load prompt content
    react_prompt_content = ""
    try:
        with open(code_react_prompt_path, "r", encoding="utf-8") as f:
            react_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {code_react_prompt_path}")
        # Fallback to a default prompt if file not found
        react_prompt_content = "You are a code execution agent. Use the provided tools to write, run, and debug code. Respond with your findings or 'Final Answer: ...' when done. If an error occurs, report it clearly."

    react_prompt = ChatPromptTemplate.from_template(react_prompt_content)

    # Create the ReAct agent executor directly
    code_agent_runnable = AgentExecutor(
        agent=create_react_agent(llm, tools, react_prompt),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return code_agent_runnable