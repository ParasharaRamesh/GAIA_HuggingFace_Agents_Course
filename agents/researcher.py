import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents import split_point_marker, human_message_content_template
from tools.search import web_search, wikipedia_search, arxiv_search, web_scraper
from agents.state import AgentState

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
    full_prompt_content = ""
    try:
        with open(researcher_react_prompt_path, "r", encoding="utf-8") as f:
            full_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {researcher_react_prompt_path}. Using fallback prompt content.")
        # Fallback to a default prompt if file not found, adhering to the new structure
        full_prompt_content = (
            "You are a research agent. Use the provided tools to find information."
            " Respond with your findings or 'Final Answer: ...' when done. If an error occurs, report it clearly."
            "\n\nAvailable Tools:\n{tools}\n\nTool Names:\n{tool_names}"  # Added for completeness in fallback
            "\n\nBegin!\nHuman Input: {input}\nThought:{agent_scratchpad}"
        )

    system_message_content = full_prompt_content  # Default, will be updated if split_point_marker is found

    if split_point_marker in full_prompt_content:
        parts = full_prompt_content.split(split_point_marker, 1)
        system_message_content = parts[0] + "\nBegin!"  # Append 'Begin!' back to the system part
    else:
        print(f"Warning: Split marker '{split_point_marker}' not found in '{researcher_react_prompt_path}'. "
              "Ensure the file ends with exactly 'Begin!\\nHuman Input: {input}\\nThought:{agent_scratchpad}'. "
              "Using the entire file content as the system message and adding human input template explicitly."
              )

    # Construct the ChatPromptTemplate using from_messages
    react_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message_content),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content=human_message_content_template)
    ])

    # Create the ReAct agent executor directly
    researcher_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="research_expert",  # Ensure consistent naming
        debug=True,
        state_schema=AgentState
    )
    return researcher_agent_runnable
