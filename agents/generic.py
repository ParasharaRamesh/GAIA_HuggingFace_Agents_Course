# agents/generic_agent.py

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

# Import tools: web_search and web_scraper from tools/search.py
from tools.search import web_search, web_scraper


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

    react_prompt_content = ""
    try:
        with open(generic_react_prompt_path, "r", encoding="utf-8") as f:
            react_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {generic_react_prompt_path}")
        # Fallback to a default prompt if file not found
        react_prompt_content = (
            "You are a versatile generic agent. Your primary goal is to address the user's "
            "query and complete assigned tasks. You have access to web search and web scraping "
            "tools. If you cannot find a definitive answer, you are encouraged to use your "
            "general knowledge to provide a contextual guess or 'hallucination', explicitly "
            "stating that it is a speculative answer. Always aim to provide a 'Final Answer: ...' "
            "when you believe you have addressed the core of the query, even if speculative. "
            "If an error occurs or you are stuck, report it clearly."
            "\n\nAvailable Tools:\n{tools}\n\nTool Names:\n{tool_names}\n\nBegin!\n\nHuman Input: {input}\nThought:{agent_scratchpad}"
        )

    react_prompt = ChatPromptTemplate.from_template(react_prompt_content)

    generic_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="generic-agent",
        debug=True
    )

    return generic_agent_runnable
