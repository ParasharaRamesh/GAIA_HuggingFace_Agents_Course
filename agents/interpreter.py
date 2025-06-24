import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from agents import create_clean_agent_messages_hook
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
    react_prompt_content = ""
    try:
        with open(code_react_prompt_path, "r", encoding="utf-8") as f:
            react_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {code_react_prompt_path}")
        # --- MODIFIED FALLBACK PROMPT START ---
        # Fallback to a default prompt if file not found
        # IMPORTANT: The content of this fallback is also a SYSTEM MESSAGE.
        # It mirrors the new content for code_react_prompt.txt.
        react_prompt_content = (
            "You are an expert Python programmer and a powerful Code Execution AI. "
            "Your mission is to solve complex problems by diligently writing, executing, and debugging Python code. "
            "You are highly skilled in interacting with the filesystem (reading/writing files) and executing shell commands to achieve your tasks. "
            "You operate in an iterative Thought-Action-Observation loop.\n\n"
            "**Your Goal:** Accurately and precisely complete the delegated coding task. "
            "The task you receive may or may not contain multiple sub-tasks. You should use any provided sub-tasks as a high-level guide for your approach, "
            "but your ultimate priority is to successfully complete the overall task, even if small deviations from the suggested sub-task plan are necessary. "
            "Your aim is to provide robust, verified solutions.\n\n"
            "**Constraints:**\n- You MUST only use the tools provided. Do not invent new tools.\n"
            "- Do not make assumptions. If information is missing, use your tools or signal you are STUCK.\n"
            "- When writing code, ensure it is robust, handles edge cases, provides clear output, and is well-commented.\n"
            "- Always consider security implications when executing commands or writing to the filesystem.\n\n"
            "**Important Operational Guidelines:**\n"
            "* **Strict Flat File Structure:** All files you create or modify, and all files your Python scripts create, **MUST** reside directly in the current execution directory ('.'). "
            "**You are strictly forbidden from creating, reading, or using any subdirectories whatsoever.** This means:\n"
            "    * When using `read_file`, `write_file`, or `run_python_script`, you **must use only the filename** (e.g., `my_script.py`, `output.csv`).\n"
            "    * **Do NOT** include any path separators (`/` or `\\`) in your file paths.\n"
            "    * Ensure any Python code you write (saved via `write_file` and run via `run_python_script`) also adheres to this flat file structure, "
            "saving all its output files directly in the current directory.\n"
            "* **Error Handling & Giving Up:** If you encounter a series of errors and find yourself unable to make progress after several attempts "
            "(e.g., if you are stuck in a loop of trying to fix a persistent error), you should report a failure to the system. "
            "The system expects you to recognize when a task is not solvable within a reasonable number of attempts.\n"
            "* **Verification:** Before concluding, always verify your solution by running the code and inspecting its output.\n\n"
            "**Final Answer Format:**\n"
            "- **Successful Completion:** When you have successfully completed the task and verified your solution, "
            "provide it clearly in the format: 'Final Answer: [A concise summary of the steps taken to solve this task "
            "(e.g., \"created file X, ran script Y, debugged error Z\"), followed by your EXACT final answer/result to the task, "
            "including any numerical results, textual findings, or paths to generated files (e.g., \"The calculated value is 123. The plot is saved at my_plot.png\").]'\n"
            "- **Stuck/Cannot Proceed:** If you encounter a situation where you cannot make progress or complete the task with your available tools, "
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

    # Create the ReAct agent executor directly
    code_agent_runnable = create_react_agent(
        model=llm,
        tools=tools,
        prompt=react_prompt,
        name="code",
        debug=True,
        state_schema=SubAgentState,
        pre_model_hook=create_clean_agent_messages_hook("code")
    )
    return code_agent_runnable
