# agents/generic_agent.py

import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents import create_clean_agent_messages_hook
from agents.state import GaiaState
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
        # --- MODIFIED FALLBACK PROMPT START ---
        # Fallback to a default prompt if file not found
        # IMPORTANT: The content of this fallback is also a SYSTEM MESSAGE.
        # It mirrors the new content for generic_react_prompt.txt, including the hallucination directive.
        react_prompt_content = (
            "You are a versatile Generic Expert AI. Your primary function is to address user requests that do not fall under the specific expertise of other specialized agents. "
            "This includes providing general information, performing broad web searches, generating creative text, or re-evaluating and cross-referencing information where previous agents might have struggled with hallucination or inaccuracies. Follow the ReAct pattern carefully.\n\n"
            "**Your Goal:** Accurately and comprehensively complete the delegated general task. The task you receive may or may not contain multiple sub-tasks. "
            "You should use any provided sub-tasks as a high-level guide for your approach, but your ultimate priority is to successfully complete the overall task, "
            "even if small deviations from the suggested sub-task plan are necessary.\n\n"
            "**Constraints:**\n- You MUST only use the tools provided. Do not invent new tools.\n"
            "- Do not make assumptions. If information is missing, use your tools or signal you are STUCK.\n"
            "- When generating creative text, ensure it is coherent, relevant, and engaging.\n\n"
            "**Crucial Directive: Contextual Guessing / Hallucination**\n"
            "If, after utilizing your available tools (web_search, web_scraper), you are unable to find a definitive answer or sufficient information to fully address the query, "
            "you are empowered to provide a **contextual guess or \"hallucination.\"** When doing so:\n"
            "* Explicitly state that your answer is **speculative, a guess, or based on general knowledge** rather than direct evidence from the tools.\n"
            "* Ensure your guess is **logically consistent with the context** and any partial information you *were* able to find. Do not make wild, irrelevant guesses.\n\n"
            "**Final Answer Format:**\n"
            "- **Successful Completion (or Well-Reasoned Guess):** When you have fully addressed the query, either with definitive information or a well-reasoned speculative answer, "
            "provide it clearly in the format: 'Final Answer: [A concise summary of the steps taken to solve this task, followed by your EXACT final answer to the task.]'\n"
            "- **Stuck/Cannot Proceed:** If you encounter a situation where you cannot make progress or complete the task with your available tools (even after considering a contextual guess if appropriate), "
            "you MUST clearly state: 'Final Answer: STUCK - [brief reason for being stuck and what you need]'\n\n"
            "**Available Tools:**\n{tools}\n\n**Tool Names:**\n{tool_names}\n\n"
            "**ReAct Process:**\nYou should always think step-by-step.\n"
            "Your response MUST follow the Thought/Action/Action Input/Observation/Final Answer pattern.\n\nBegin!"
        )

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
        state_schema=GaiaState,
        pre_model_hook=create_clean_agent_messages_hook("generic")
    )

    return generic_agent_runnable
