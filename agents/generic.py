# agents/generic_agent.py

import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents import split_point_marker, human_message_content_template
from agents.state import AgentState
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

    full_prompt_content  = ""
    try:
        with open(generic_react_prompt_path, "r", encoding="utf-8") as f:
            full_prompt_content  = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {generic_react_prompt_path}")
        # Fallback to a default prompt if file not found
        full_prompt_content = (
            "You are a versatile generic agent. Your primary goal is to address the overall query presented in the conversation. You operate in a Thought-Action-Observation loop."
            "\n\n**Workflow and Rules:**\n1.  **Understand:** Carefully read the Human Input and the context provided in the conversation history. Infer your specific task from this.\n2.  **Thought:** Always articulate your thought process. Explain your plan, what information you need, what tools you intend to use, and why.\n3.  **Action:** Choose the best tool for your current Thought.\n    * `Action: web_search` - Use this for general information retrieval from the internet.\n    * `Action: web_scraper` - Use this to extract content from a specific URL.\n4.  **Observation:** Review the results of your tool execution.\n5.  **Refine/Iterate:** Based on the observation, refine your thought and take further actions until you can formulate a comprehensive response."
            "\n\n**Crucial Directive: Contextual Guessing / Hallucination**\nIf, after utilizing your available tools (web_search, web_scraper), you are unable to find a definitive answer or sufficient information to fully address the query, you are empowered to provide a **contextual guess or \"hallucination.\"** When doing so:\n* Explicitly state that your answer is **speculative, a guess, or based on general knowledge** rather than direct evidence from the tools.\n* Ensure your guess is **logically consistent with the context** and any partial information you *were* able to find. Do not make wild, irrelevant guesses."
            "\n\n**Final Answer:**\nWhen you believe you have fully addressed the query, either with definitive information or a well-reasoned speculative answer, provide your final response using the `Final Answer:` format. The content after `Final Answer:` should be a concise summary of your findings."
            "\n\nAvailable Tools:\n{tools}\n\nTool Names:\n{tool_names}"
            "\n\nBegin!\nHuman Input: {input}\nThought:{agent_scratchpad}"
        )

    system_message_content = full_prompt_content  # Default, will be updated if split_point_marker is found

    if split_point_marker in full_prompt_content:
        # Split the content. The first part is the system instructions up to (but not including) the marker.
        # We then append "\nBegin!" back to the system message content to complete that instruction.
        parts = full_prompt_content.split(split_point_marker, 1)  # Split only once
        system_message_content = parts[0] + "\nBegin!"  # Append 'Begin!' back to the system part
        # The 'human_message_content_template' is already set to the desired end part
    else:
        print(f"Warning: Specific split marker '{split_point_marker}' not found in '{generic_react_prompt_path}'. "
              "Ensure the file ends with exactly 'Begin!\\nHuman Input: {input}\\nThought:{agent_scratchpad}'. "
              "Using the entire file content as the system message and adding human input template explicitly."
              "This might lead to redundancy if the file structure does not match expected split."
              )
        # If the marker isn't found, we'll assume the entire file content is part of the system message
        # and rely on the explicit human_message_content_template.

    # Construct the ChatPromptTemplate using from_messages
    react_prompt = ChatPromptTemplate.from_messages([
        # 1. System Message: Contains all the agent's core instructions, rules, and tool information.
        #    {tools} and {tool_names} are already present in system_message_content extracted from the file.
        SystemMessage(content=system_message_content),

        # 2. Messages Placeholder: This is CRUCIAL. It dynamically inserts the entire conversational history
        #    (including previous thoughts, tool calls, and observations). `create_react_agent` uses this
        #    to construct the 'agent_scratchpad' internally from the messages.
        MessagesPlaceholder(variable_name="messages"),

        # 3. Human Message: Represents the current user input and prompts for the agent's next thought.
        #    {input} and {agent_scratchpad} are variables that `create_react_agent` expects to fill.
        #    The template comes from the standardized end of your prompt files.
        HumanMessage(content=human_message_content_template)
    ])

    generic_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="generic",
        debug=True,
        state_schema=AgentState
    )

    return generic_agent_runnable
