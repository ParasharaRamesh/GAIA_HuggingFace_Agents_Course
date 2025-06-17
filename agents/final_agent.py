import os
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Import AgentState and HistoryEntry from state.py
from .state import AgentState, HistoryEntry, HistoryEntryStatus

class FinalAgent:
    """
    The final agent in the workflow, responsible for synthesizing all gathered
    information into the definitive final answer for the user's query.
    It uses an LLM to formulate a coherent response based on the active agent's output
    and the overall history.
    """
    def __init__(self, llm: BaseChatModel):
        """
        Initializes the FinalAgent with an LLM for answer synthesis.

        Args:
            llm (BaseChatModel): The language model to synthesize the final answer.
        """
        self.llm = llm

        # Construct the path to the Final Agent's prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, '..', 'prompts')
        self.final_agent_prompt_path = os.path.join(prompts_dir, 'final_agent_prompt.txt')

    def __call__(self, state: AgentState) -> AgentState: # Corrected return type hint to AgentState
        """
        Synthesizes the final answer based on the current AgentState.

        Args:
            state (AgentState): The current state of the multi-agent system.

        Returns:
            AgentState: The updated AgentState with the final_answer populated.
        """
        query = state["query"]
        active_agent_output = state.get("active_agent_output")
        history_entries = state["history"] # History included for prompt formatting
        planner_feedback = state.get("planner_feedback", "No specific feedback.")

        # Create a mutable copy of the state to update
        new_state = state.copy()
        new_state["active_agent_name"] = "FinalAgent"
        new_state["active_agent_output"] = None # Reset output, as this agent's output is the final_answer
        new_state["active_agent_error_message"] = None

        # Load the prompt template
        with open(self.final_agent_prompt_path, "r", encoding="utf-8") as f:
            prompt_template_str = f.read()

        # Format the prompt with current state information
        formatted_prompt = prompt_template_str.format(
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query=query,
            active_agent_output=active_agent_output if active_agent_output else "No specific output from previous agent.",
            planner_feedback=planner_feedback,
            history=self._format_history(history_entries) # History re-added to prompt formatting
        )

        messages = [SystemMessage(content=formatted_prompt)]
        messages.append(HumanMessage(content=f"Please provide the final answer to the query: {query}"))

        final_answer_text = None
        error_message = None
        status = HistoryEntryStatus.IN_PROGRESS

        try:
            llm_response = self.llm.invoke(messages)
            final_answer_text = llm_response.content
            status = HistoryEntryStatus.SUCCESS

        except Exception as e:
            error_message = str(e)
            status = HistoryEntryStatus.FAILED
            print(f"[FinalAgent.call]: LLM invocation error: {e}")

        new_state["final_answer"] = final_answer_text # This is the main output of this agent
        new_state["active_agent_output"] = final_answer_text # Also set as active_agent_output for consistency
        new_state["active_agent_error_message"] = error_message

        # Keep history update comprehensive for audit trail, as it's separate from prompt input
        new_state["history"].append(HistoryEntry(
            agent_name="FinalAgent",
            timestamp=datetime.now().isoformat(),
            input={
                "query": query,
                "context_from_previous_agent": active_agent_output,
                "planner_feedback_at_final_stage": planner_feedback
            },
            output=final_answer_text,
            status=status,
            tool_calls=[], # FinalAgent does not make tool calls
            error=error_message
        ))

        return new_state

    def _format_history(self, history_entries: List[HistoryEntry]) -> str:
        """
        Helper method to format a list of history entries into a readable string
        for the LLM, providing context of previous interactions.
        This method is important for the overall audit log and other agents' context,
        and its formatted output is used in the FinalAgent's prompt.
        """
        formatted_history = []
        for entry in history_entries:
            formatted_history.append(f"--- History Entry for {entry['agent_name']} ---")
            formatted_history.append(f"Timestamp: {entry['timestamp']}")
            formatted_history.append(f"Status: {entry['status']}")
            # Ensure safe access to nested input dict
            input_task = entry['input'].get('task_for_react_agent') or \
                         entry['input'].get('task_for_visual_agent') or \
                         entry['input'].get('query', 'N/A')
            formatted_history.append(f"Input Task/Query: {input_task}")
            formatted_history.append(f"Output: {entry['output']}")
            if entry['error']:
                formatted_history.append(f"Error: {entry['error']}")
            formatted_history.append("--------------------")
        if not formatted_history:
            return "No previous history."
        return "\n".join(formatted_history)