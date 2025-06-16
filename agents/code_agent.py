import os
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.agents import AgentFinish

# Import tools from interpreter.py - make sure this path is correct relative to your project structure
from tools.interpreter import read_file, write_file, run_shell_command, run_python_script

# Import AgentState and HistoryEntry from state.py - ensure state.py is accessible
from state import AgentState, HistoryEntry, HistoryEntryStatus

class CodeAgent:
    """
    An agent responsible for writing, executing, and debugging Python code.
    It uses a ReAct pattern with tools for filesystem interaction and code execution.
    All file operations (read, write, script execution, and files created by scripts)
    are strictly enforced to occur directly in the current execution directory ('.').
    No subdirectories are allowed.
    """
    def __init__(self, llm: BaseChatModel):
        """
        Initializes the CodeAgent with an LLM.
        All file operations will be relative to the current working directory ('.').
        """
        self.llm = llm

        # Define the tools available to the CodeAgent
        self.tools = [
            read_file,
            write_file,
            run_shell_command,
            run_python_script
        ]

        # Construct the path to the ReAct prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, '..', 'prompts')
        self.code_react_prompt_path = os.path.join(prompts_dir, 'code_react_prompt.txt')

        # Initialize the LangChain AgentExecutor
        self.agent_executor = self._initialize_agent_executor()

    def _initialize_agent_executor(self) -> AgentExecutor:
        """
        Loads the prompt and creates the ReAct AgentExecutor.
        """
        with open(self.code_react_prompt_path, "r", encoding="utf-8") as f:
            prompt_template_str = f.read()

        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        agent = create_react_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20, # Max internal ReAct iterations per invocation
            max_execution_time=300 # Max time for the agent to run in seconds
        )

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the CodeAgent's task based on the current AgentState.

        Args:
            state (AgentState): The current state of the multi-agent system.

        Returns:
            Dict[str, Any]: The updated AgentState after the CodeAgent's execution.
        """
        task_for_react_agent = state["active_agent_task"]
        query = state["query"]
        history_entries = state["history"]

        # Create a mutable copy of the state to update
        new_state = state.copy()
        new_state["active_agent_name"] = "CodeAgent"
        new_state["active_agent_output"] = None # Use active_agent_output as defined in AgentState
        new_state["active_agent_error_message"] = None # Use active_agent_error_message as defined in AgentState

        react_agent_input_text = (
            f"Overall Query: {query}\n\n"
            f"Current Task for Code Agent: {task_for_react_agent}\n\n"
            f"***IMPORTANT FILE SYSTEM RULE:***\n"
            f"All file operations (read, write, script execution, and files created by your scripts) "
            f"MUST be performed directly in the current execution directory ('.'). "
            f"You are **strictly forbidden** from creating, reading, or using any subdirectories whatsoever. "
            f"When specifying file paths for `read_file`, `write_file`, or `run_python_script`, "
            f"you **must use only the filename** (e.g., 'my_script.py', 'output.csv', 'plot.png'). "
            f"Do NOT include any path separators ('/' or '\\') in your file paths. "
            f"Ensure any Python code you write also adheres to this flat file structure.\n\n"
            f"History of previous agent interactions and observations:\n{self._format_history(history_entries)}\n"
            f"Begin!\n\n"
        )

        agent_output = None
        error_message = None
        status = HistoryEntryStatus.IN_PROGRESS
        extracted_tool_calls_for_history = []

        try:
            result = self.agent_executor.invoke({"input": react_agent_input_text})

            if isinstance(result, dict) and "output" in result:
                agent_output = result["output"]
                status = HistoryEntryStatus.SUCCESS
            else:
                agent_output = str(result)
                status = HistoryEntryStatus.SUCCESS

        except AgentFinish as e:
            agent_output = e.return_values["output"]
            status = HistoryEntryStatus.SUCCESS
        except Exception as e:
            error_message = str(e)
            status = HistoryEntryStatus.FAILED
            print(f"[CodeAgent.call]: AgentExecutor error: {e}")

        new_state["active_agent_output"] = agent_output # Update active_agent_output
        new_state["active_agent_error_message"] = error_message # Update active_agent_error_message

        new_state["history"].append(HistoryEntry(
            agent_name="CodeAgent",
            timestamp=datetime.now().isoformat(),
            input={
                "task_for_react_agent": task_for_react_agent,
                "full_input_to_react_agent": react_agent_input_text,
                "query": query
            },
            output=agent_output,
            status=status,
            tool_calls=extracted_tool_calls_for_history,
            error=error_message
        ))

        # If the agent produces a "Final Answer", populate the final_answer field.
        if agent_output and isinstance(agent_output, str) and agent_output.strip().startswith("Final Answer:"):
            new_state["final_answer"] = agent_output.replace("Final Answer:", "").strip()

        return new_state

    def _format_history(self, history_entries: List[HistoryEntry]) -> str:
        """
        Helper method to format a list of history entries into a readable string
        for the LLM, providing context of previous interactions.
        """
        formatted_history = []
        for entry in history_entries:
            formatted_history.append(f"--- History Entry for {entry['agent_name']} ---")
            formatted_history.append(f"Timestamp: {entry['timestamp']}")
            formatted_history.append(f"Status: {entry['status']}")
            formatted_history.append(f"Input Task: {entry['input'].get('task_for_react_agent', 'N/A')}")
            formatted_history.append(f"Output: {entry['output']}")
            if entry['error']:
                formatted_history.append(f"Error: {entry['error']}")
            formatted_history.append("--------------------")
        if not formatted_history:
            return "No previous history."
        return "\n".join(formatted_history)