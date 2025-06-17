# agents/planner_agent.py

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState, HistoryEntry, HistoryEntryStatus  # Note the '.' for relative import


class PlannerAgent:
    """
    The central orchestration agent responsible for planning the overall workflow,
    delegating tasks to specialized agents, and verifying progress.
    It analyzes the current AgentState and decides the next steps, including
    which agent to activate next and what task to assign.
    This agent does NOT use tools directly but directs other agents that do.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initializes the PlannerAgent with an LLM for strategic decision-making.
        """
        self.llm = llm

        # Construct the path to the Planner Agent's prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, '..', 'prompts')
        planner_prompt_path = os.path.join(prompts_dir, 'planner_agent_prompt.txt')

        # Load the planner prompt content
        planner_prompt_content = ""
        try:
            with open(planner_prompt_path, 'r', encoding='utf-8') as f:
                planner_prompt_content = f.read()
        except FileNotFoundError:
            print(f"Error: Planner prompt file not found at {planner_prompt_path}")
            # Provide a basic fallback prompt or raise an error
            planner_prompt_content = (
                "You are a sophisticated workflow planner. Your task is to decide "
                "the next agent and task based on the provided state. "
                "Output a JSON with 'next_agent' and 'next_task'. "
                "Current state: {current_state_summary}"
            )

        # Create the ChatPromptTemplate for the planner
        self.planner_prompt_template = ChatPromptTemplate.from_messages([
            ("system", planner_prompt_content),
            ("human", "{user_query}\n\n{current_state_summary}\n\n{conversation_history}\n\n{audit_history}")
        ])

    def _format_state_summary(self, state: AgentState) -> str:
        """Helper to format key parts of the AgentState for the planner's prompt."""
        summary = f"Overall Query: {state.get('query', 'N/A')}\n"
        summary += f"Current High-Level Roadmap: {state.get('high_level_roadmap', [])}\n"
        summary += f"Current Roadmap Step Index: {state.get('current_roadmap_step_index', 0)}\n"
        summary += f"Active Agent: {state.get('active_agent_name', 'None')}\n"
        summary += f"Active Agent's Last Task: {state.get('active_agent_task', 'None')}\n"
        summary += f"Active Agent's Last Output: {str(state.get('active_agent_output', 'None'))[:500]}...\n"  # Truncate long outputs
        summary += f"Planner Feedback from Last Turn: {state.get('planner_feedback', 'None')}\n"
        summary += f"Active Agent's Last Error: {state.get('active_agent_error_message', 'None')}\n"
        summary += f"Final Answer so far: {state.get('final_answer', 'None')}\n"
        return summary

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Formats the conversation history for the LLM."""
        if not history:
            return "No prior direct conversation between Planner and Agent."
        formatted = ["--- Conversation History ---"]
        for entry in history:
            formatted.append(f"{entry.get('role', 'unknown').capitalize()}: {entry.get('message', 'N/A')}")
        return "\n".join(formatted)

    def _format_audit_history(self, history_entries: List[HistoryEntry]) -> str:
        """
        Helper method to format a list of history entries (audit log) into a readable string
        for the LLM, providing context of all significant operations.
        """
        if not history_entries:
            return "No prior audit log entries."
        formatted_history = ["--- Overall Audit Log ---"]
        for entry in history_entries:
            # Ensure safe access to nested input dict and handle tool_calls gracefully
            input_content = entry['input'].get('task_for_react_agent') or \
                            entry['input'].get('task_for_visual_agent') or \
                            entry['input'].get('query', 'N/A')
            tool_calls_summary = "None"
            if entry.get('tool_calls'):
                tool_calls_summary = "; ".join(
                    [f"{tc.get('action_type', 'Tool')}: {tc.get('action_details', '')[:100]}..." for tc in
                     entry['tool_calls']])

            formatted_history.append(
                f"Agent: {entry['agent_name']} | Status: {entry['status']} | Timestamp: {entry['timestamp']}")
            formatted_history.append(f"  Input Task: {input_content}")
            formatted_history.append(f"  Output: {str(entry['output'])[:200]}...")  # Truncate for brevity
            if tool_calls_summary != "None":
                formatted_history.append(f"  Tool Calls: {tool_calls_summary}")
            if entry['error']:
                formatted_history.append(f"  Error: {entry['error']}")
            formatted_history.append("---")
        return "\n".join(formatted_history)

    def __call__(self, state: AgentState) -> AgentState:
        """
        Executes the planning logic based on the current AgentState.
        Determines the next agent to activate and the task for that agent.
        """
        print("\n--- PlannerAgent: Executing planning logic ---")
        current_state_summary = self._format_state_summary(state)
        conversation_history = self._format_conversation_history(state.get("conversation_history_with_agent", []))
        audit_history = self._format_audit_history(state.get("history", []))

        # Prepare the input for the LLM
        messages = self.planner_prompt_template.format_messages(
            user_query=state.get("query", "No specific query provided."),
            current_state_summary=current_state_summary,
            conversation_history=conversation_history,
            audit_history=audit_history
        )

        try:
            llm_response = self.llm.invoke(messages)
            raw_response_content = llm_response.content.strip()

            # Attempt to parse the JSON response
            parsed_response = json.loads(raw_response_content)

            next_agent = parsed_response.get("next_agent")
            next_task = parsed_response.get("next_task")
            planner_feedback = parsed_response.get("planner_feedback")
            roadmap_action = parsed_response.get("roadmap_update", {}).get("action")
            roadmap_value = parsed_response.get("roadmap_update", {}).get("value")

            if not next_agent or not next_task:
                raise ValueError("Planner did not return 'next_agent' or 'next_task' in expected JSON format.")

            new_state = state.copy()
            new_state["active_agent_name"] = next_agent
            new_state["active_agent_task"] = next_task
            new_state["planner_feedback"] = planner_feedback
            new_state["active_agent_error_message"] = None  # Clear previous error when planner acts

            # Update roadmap based on planner's decision
            if roadmap_action == "increment_step":
                new_state["current_roadmap_step_index"] = new_state.get("current_roadmap_step_index", 0) + 1
            elif roadmap_action == "set_roadmap":
                new_state["high_level_roadmap"] = roadmap_value if isinstance(roadmap_value, list) else [roadmap_value]
                new_state["current_roadmap_step_index"] = 0  # Reset index on new roadmap
            elif roadmap_action == "add_step":
                if isinstance(new_state["high_level_roadmap"], list):
                    new_state["high_level_roadmap"].append(roadmap_value)
                else:
                    new_state["high_level_roadmap"] = [roadmap_value]

            # Record planner's action in audit history
            new_state["history"].append(HistoryEntry(
                agent_name="PlannerAgent",
                timestamp=datetime.now().isoformat(),
                input={
                    "query": state["query"],
                    "current_state_summary": current_state_summary
                },
                output={
                    "next_agent": next_agent,
                    "next_task": next_task,
                    "planner_feedback": planner_feedback,
                    "roadmap_update": parsed_response.get("roadmap_update")
                },
                status=HistoryEntryStatus.SUCCESS,
                tool_calls=[],  # Planner does not make direct tool calls
                error=None
            ))

            # Append planner's decision to conversation history
            new_state["conversation_history_with_agent"].append({
                "role": "planner",
                "message": f"Decided to activate '{next_agent}' with task: '{next_task}'"
            })

            print(f"--- PlannerAgent: Decided to activate '{next_agent}' with task: '{next_task}' ---")
            return new_state

        except json.JSONDecodeError as e:
            error_message = f"Planner LLM response was not valid JSON: {e}\nRaw Response: {raw_response_content}"
            print(f"--- PlannerAgent Error: {error_message} ---")
            new_state = state.copy()
            new_state["active_agent_error_message"] = error_message
            new_state["active_agent_name"] = "PlannerAgent"  # Indicate Planner itself encountered an error
            new_state["active_agent_task"] = "Handle JSON parsing error"
            new_state["history"].append(HistoryEntry(
                agent_name="PlannerAgent",
                timestamp=datetime.now().isoformat(),
                input={"query": state["query"], "current_state_summary": current_state_summary},
                output={"error": error_message},
                status=HistoryEntryStatus.FAILED,
                tool_calls=[],
                error=error_message
            ))
            new_state["conversation_history_with_agent"].append({
                "role": "planner",
                "message": f"Error parsing response: {error_message}"
            })
            return new_state
        except Exception as e:
            error_message = f"An unexpected error occurred in PlannerAgent: {e}"
            print(f"--- PlannerAgent Error: {error_message} ---")
            new_state = state.copy()
            new_state["active_agent_error_message"] = error_message
            new_state["active_agent_name"] = "PlannerAgent"  # Indicate Planner itself encountered an error
            new_state["active_agent_task"] = "Handle unexpected error"
            new_state["history"].append(HistoryEntry(
                agent_name="PlannerAgent",
                timestamp=datetime.now().isoformat(),
                input={"query": state["query"], "current_state_summary": current_state_summary},
                output={"error": error_message},
                status=HistoryEntryStatus.FAILED,
                tool_calls=[],
                error=error_message
            ))
            new_state["conversation_history_with_agent"].append({
                "role": "planner",
                "message": f"An unexpected error occurred: {error_message}"
            })
            return new_state