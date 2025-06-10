# Defines the Planner agent
# agents/planner.py
import operator
from typing import List, Any, Literal
from datetime import datetime

# Import the core state definitions and helper from agents.state
from agents.state import AgentState, PlanStep, PlanStepStatus, HistoryEntry

'''TODO.x:
1. How is the model switch to a reasoning model happening? doesnt seem like it is happening. Perhaps for first version not really needed!
2. The planner_prompt might not be aware of the planstep type, how to make it aware in the prompt?



'''

class GlobalPlanner:
    """
    The GlobalPlanner is responsible for generating the overall plan of action,
    re-planning when errors occur, and making high-level decisions for the agent's workflow.
    It orchestrates the sequence of operations by modifying the `current_plan` in the AgentState.
    """

    def __init__(self):
        # TODO.0: Initialize LLM for planning here. For now, it's a dummy.
        self.planner_llm = None  # Placeholder for actual LLM
        # TODO.1: Define a standard prompt for the GlobalPlanner.
        self.planner_prompt = """
        You are an expert autonomous planning agent. Your task is to break down complex user queries
        into a sequence of actionable steps (PlanSteps) for other specialized agents to execute.
        
        Allowed agents are: web_search_agent, wiki_agent, arxiv_agent, audio_agent, code_agent, visual_agent, summarizer_agent, final_answer_agent.

        Consider the following rules:
        1. Each PlanStep must specify an 'agent_to_call' from the allowed list.
        2. Provide clear 'input_for_agent' arguments for the chosen agent.
        3. Define 'expected_output_type' for verification.
        4. Provide 'details' and 'substeps' for clarity.
        5. The 'status' of each step should initially be 'PENDING'.
        6. If re-planning due to a 'failure' or 'error_message', analyze the error and adjust the plan.
           Consider retrying the current step if appropriate or generating a new approach.
        7. The final step of any successful plan should lead to a 'final_answer_agent'.

        
        """

    def __call__(self, state: AgentState) -> AgentState:
        """
        Generates or updates the `current_plan` based on the current `AgentState`.
        This method acts as the main planning node in the LangGraph.

        Args:
            state (AgentState): The current state of the agent system.

        Returns:
            AgentState: The updated state with a new or modified `current_plan`.
        """
        print("\n--- GlobalPlanner Node ---")
        print(f"Current query: {state['query']}")
        print(f"Current step index: {state['current_step_index']}")
        print(f"Verification status: {state['verification_status']}")
        print(f"Error message: {state['error_message']}")
        print(f"Retry count: {state['retry_count']}")

        # LangGraph nodes operate by returning a new state dictionary or a modified copy.
        # It's good practice to work on a copy to avoid unexpected mutations elsewhere
        # if the original dict reference was used.
        updated_state = state.copy()

        # Determine if we need a new plan or a re-plan
        if not updated_state.get('current_plan') or updated_state['verification_status'] == 'failure':
            print("Generating new plan or re-planning due to failure...")
            # The _generate_or_replan method will modify updated_state in place
            # or return a new dict. Ensure it operates on the copy.
            self._generate_or_replan(updated_state)
            action = "Generated new plan" if not state.get('current_plan') else "Re-planned due to failure"
            status = "SUCCESS"  # Planner assumes its own planning was successful
        else:
            print("Plan already exists and no failure. Proceeding with existing plan.")
            action = "No re-planning needed"
            status = "SKIPPED"  # This node skipped detailed re-planning

        # Append to history
        updated_state['history'].append(create_history_entry(
            node="global_planner_node",
            action=action,
            input_to_node={"query": state['query'], "current_step_index": state['current_step_index'],
                           "verification_status": state['verification_status'], "error_message": state['error_message'],
                           "retry_count": state['retry_count']},
            output_from_node={"plan_length": len(updated_state['current_plan']) if updated_state['current_plan'] else 0,
                              "new_step_index": updated_state['current_step_index'],
                              "new_retry_count": updated_state['retry_count']},
            status=status,
            model_used="dummy-planner-llm",  # Placeholder
            # Ensure a copy of the plan is stored in history, not a reference
            plan_snapshot=[step.copy() for step in updated_state['current_plan']] if updated_state[
                'current_plan'] else []
        ))

        return updated_state

    def _generate_or_replan(self, state: AgentState) -> None:
        """
        Internal method to generate a new plan or adjust an existing one based on context.
        This is where the LLM planning logic will eventually reside.
        Modifies the state dictionary in place.
        """
        error_message = state.get('error_message')
        current_step_index = state['current_step_index']
        max_retries = 3  # Define a max retry count

        # --- Dummy LLM Planning Logic (to be replaced with actual LLM calls) ---
        # This part will eventually involve querying self.planner_llm with self.planner_prompt

        # Scenario 1: Re-plan due to failure and retries are available
        if error_message and state.get('current_plan') and state['retry_count'] < max_retries:
            print(f"Attempting to fix failed step at index {current_step_index}: {error_message}")

            # Copy existing plan to modify
            new_plan_list = [step.copy() for step in state['current_plan']]

            # If the current_step_index is valid, mark it for retry
            if 0 <= current_step_index < len(new_plan_list):
                new_plan_list[current_step_index]['status'] = 'PENDING'
                # Clear error message for next attempt
                state['error_message'] = None
                state['verification_status'] = None
            else:
                # If current_step_index is out of bounds (e.g., plan finished before error),
                # regenerate a fresh plan.
                print("Current step index out of bounds during retry. Generating fresh plan.")
                new_plan_list = self._create_initial_dummy_plan(state['query'])
                state['current_step_index'] = 0  # Reset index for new plan
                state['error_message'] = None  # Clear error
                state['verification_status'] = None  # Clear status

            state['current_plan'] = new_plan_list
            state['retry_count'] += 1  # Increment retry

            print(f"Retry attempt {state['retry_count']} for current step.")

        # Scenario 2: Initial plan or re-plan when retries are exhausted or no existing plan
        else:
            print("Generating a fresh plan (or retries exhausted/no existing plan).")
            # This is where the LLM would analyze the query and generate the initial sequence of steps.
            # Or, if retries are exhausted, it would generate an alternative plan.
            new_plan_list = self._create_initial_dummy_plan(state['query'])

            state['current_plan'] = new_plan_list
            state['current_step_index'] = 0  # Always reset index for a new plan
            state['retry_count'] = 0  # Reset retry count for a new plan
            state['error_message'] = None  # Clear any previous error
            state['verification_status'] = None  # Reset verification status

        # Note: This function modifies the state dictionary in place, so no return is strictly needed
        # if the caller is working with a mutable copy. However, returning it explicitly is also fine.
        # Here we rely on the state object being mutable and passed by reference (or a copy modified).

    def _create_initial_dummy_plan(self, query: str) -> List[PlanStep]:
        """
        Helper to create a simple, dummy plan. This would be generated by an LLM.
        """
        print(f"DEBUG: Creating dummy plan for query: '{query}'")
        # TODO.2: Replace this with actual LLM call to generate a plan based on the prompt.
        # This dummy plan demonstrates a sequence of different agent calls.
        plan: List[PlanStep] = [
            PlanStep(
                agent_to_call="web_search_agent",
                input_for_agent={"query": query, "max_results": 3},
                expected_output_type="web_results",
                details=f"Search the web for information about: {query}",
                substeps=["Formulate search query", "Execute web search", "Extract relevant snippets"],
                status="PENDING"
            ),
            PlanStep(
                agent_to_call="summarizer_agent",  # Will be a generic summarizer or part of Researcher
                input_for_agent={"content": "current_agent_output", "summary_type": "concise"},
                expected_output_type="summary_text",
                details="Summarize the gathered web search results.",
                substeps=["Identify core information", "Condense findings", "Format as coherent summary"],
                status="PENDING"
            ),
            PlanStep(
                agent_to_call="final_answer_agent",
                input_for_agent={"summary_data": "current_agent_output", "original_query": query},
                expected_output_type="final_answer_text",
                details="Synthesize a final answer to the original query.",
                substeps=["Integrate all findings", "Formulate concise answer", "Assess confidence"],
                status="PENDING"
            )
        ]
        # Example of how to add a conditional step for specific queries
        if "video" in query.lower() or "youtube" in query.lower():
            print("DEBUG: Adding video step to plan.")
            plan.insert(1, PlanStep(
                agent_to_call="audio_agent",  # Assuming audio agent handles YouTube transcripts
                input_for_agent={"youtube_url": "extracted_from_query_or_search_results"},  # Placeholder
                expected_output_type="transcript_text",
                details="Get transcript for relevant video.",
                substeps=["Extract video URL", "Download audio", "Transcribe audio"],
                status="PENDING"
            ))
        elif "code" in query.lower() or "programming" in query.lower():
            print("DEBUG: Adding code step to plan.")
            plan.insert(1, PlanStep(
                agent_to_call="code_agent",
                input_for_agent={"code_task": "execute_given_code"},  # Placeholder
                expected_output_type="code_execution_output",
                details="Execute code or assist with programming task.",
                substeps=["Understand code/task", "Execute or generate code", "Handle output/errors"],
                status="PENDING"
            ))

        return plan