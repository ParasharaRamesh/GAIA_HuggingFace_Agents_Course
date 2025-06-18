import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from tools.search import web_search, wikipedia_search, arxiv_search, web_scraper


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
    react_prompt_content = ""
    try:
        with open(researcher_react_prompt_path, "r", encoding="utf-8") as f:
            react_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {researcher_react_prompt_path}")
        # Fallback to a default prompt if file not found
        react_prompt_content = "You are a research agent. Use the provided tools to find information. Respond with your findings or 'Final Answer: ...' when done. If an error occurs, report it clearly."

    react_prompt = ChatPromptTemplate.from_template(react_prompt_content)

    # Create the ReAct agent executor directly
    researcher_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="researcher-agent",
        debug=True
    )

    return researcher_agent_runnable
