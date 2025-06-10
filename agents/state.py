import operator
from typing import TypedDict, List, Dict, Any, Union, Literal, Optional, Annotated
from datetime import datetime
import inspect

# --- Enums/Literals for State Attributes ---

"""
Literal type to represent the execution status of a single step within the overall plan.
Crucial for tracking progress and for the GlobalPlanner to decide on re-planning or continuation.
"""
PlanStepStatus = Literal["PENDING", "IN_PROGRESS", "SUCCESS", "FAILED"]

"""
Literal type to represent the outcome of a verification check on an agent's output.
Crucial for the VerifierAgent to signal if the output is acceptable or if re-attempts are needed.
"""
VerificationStatus = Literal["success", "failure"]

"""
Literal type to indicate the agent's confidence in its final answer.
Crucial for the system to decide if the answer is ready for presentation or requires further investigation.
"""
ConfidenceLevel = Literal["high", "low"]

"""
Literal type to indicate the agent's status at each entry point of the history. Crucial for debugging/audit purposes
"""
HistoryEntryStatus = Literal["SUCCESS", "FAILED", "SKIPPED", "IN_PROGRESS"]


class PlanStep(TypedDict):
    """
    A single, actionable step in the agent's multi-step plan.
    The GlobalPlanner defines these steps to guide the execution of specialized agents
    and achieve the overall user query goal.

    Attributes:
        agent_to_call (str): The unique string identifier of the specialized agent node that will execute this step.
            The planner must select from a predefined list of available agent nodes.
            This field dictates which agent (e.g., 'researcher_agent_node', 'audio_agent_node',
            'code_agent_node', 'final_answer_agent_node') the workflow will dispatch to for this step.
        input_for_agent (Any): A dictionary (`dict`) containing all necessary arguments and data for the `agent_to_call`
            to perform its specific task. The planner must correctly format this to match the
            expected input signature of the target agent's tools.
        expected_output_type (str): A descriptive string indicating the anticipated format or nature of the output
            from `agent_to_call`. The planner should specify this clearly to aid the
            VerifierAgent in assessing success. Examples: 'web_results_json', 'transcript_text',
            'code_execution_output', 'final_answer_text'.
        details (str): A concise, human-readable summary of this step's objective.
            The planner should make this clear for self-reflection and for the VerifierAgent
            to understand the step's intent.
        substeps (List[str]): A list of smaller, internal actions or objectives the `agent_to_call` should
            accomplish within this single `PlanStep`. These guide the executing agent's
            internal reasoning and are useful for detailed verification.
        status (PlanStepStatus): The current execution status of this step. The planner must set this to 'PENDING'
            for all new plan steps. It tracks progress for the workflow and aids re-planning.
    """
    agent_to_call: str
    input_for_agent: Any
    expected_output_type: str
    details: str
    substeps: List[str]
    status: PlanStepStatus


class HistoryEntry(TypedDict):
    """
    Records a chronological event within the agent's workflow.
    The GlobalPlanner uses this history to understand past actions, results, and errors
    for informed decision-making, especially during re-planning.

    Attributes:
        timestamp (str): ISO formatted timestamp when this entry was recorded.
        node (str): The name of the LangGraph node that processed the state to create this entry
            (e.g., 'planner_node', 'researcher_agent_node', 'verifier_node').
        action (str): A brief description of the specific action or purpose of this history entry.
        input_to_node (Any): A snapshot of the relevant input state or data the node received for processing.
        output_from_node (Any): A snapshot of the relevant output or state changes produced by the node.
        status (HistoryEntryStatus): The outcome of the node's execution for this specific entry.
        model_used (Optional[str]): The LLM model name used by the node, if applicable. Optional.
        plan_snapshot (Optional[List[PlanStep]]): A snapshot of the `current_plan` *after* this node's processing.
            Useful for understanding plan evolution and context for re-planning. Optional.
    """
    timestamp: str
    node: str
    action: str
    input_to_node: Any
    output_from_node: Any
    status: Optional[HistoryEntryStatus]
    error_details: Optional[str]
    model_used: Optional[str]
    plan_snapshot: Optional[List[PlanStep]]


# --- Main AgentState TypedDict ---

class AgentState(TypedDict):
    """
    The central, mutable state representing the agent's entire workflow.
    This state is passed between LangGraph nodes. The GlobalPlanner receives and updates
    this state to manage planning, execution, and re-planning.
    Lists are automatically appended by LangGraph due to `Annotated[..., operator.add]`.

    Attributes:
        query (str): The original user query that triggered the agent's workflow. This is the ultimate goal.
        file_path (Optional[str]): The local path to an uploaded file, if the `query` involves one. Optional.
        current_plan (Annotated[List["PlanStep"], operator.add]): The ordered list of `PlanStep`s currently guiding the agent's execution.
            The planner is responsible for generating and (during re-planning) modifying this list.
            New steps are appended automatically by LangGraph's state graph.
        current_step_index (int): The 0-based index of the `PlanStep` currently being executed or just completed.
            The planner uses this to know which step needs attention (e.g., on failure).
        current_agent_output (Any): The output (e.g., text, JSON) produced by the most recently executed `PlanStep`'s agent.
            This is often the input for the next step or for verification by the VerifierAgent.
        verification_status (Optional[VerificationStatus]): The outcome of the most recent output verification ('success' or 'failure').
            The planner must check this after a step to decide on continuation or re-planning. Optional.
        retry_count (int): The number of retries attempted for the `current_step_index`.
            The planner uses this to decide if a step should be re-attempted or if re-planning is necessary.
        history (Annotated[List["HistoryEntry"], operator.add]): A chronological, append-only record of all significant actions, inputs, and outputs
            from executed nodes. The GlobalPlanner uses this comprehensive context for reflection,
            re-planning, and auditing.
        final_answer (Optional[str]): The final, synthesized answer to the `query`, populated by the `final_answer_agent_node`
            once all necessary steps are completed and verified. Optional.
        confidence (ConfidenceLevel): The agent's confidence level ('high' or 'low') in its `final_answer`.
            The planner might use this to decide if further refinement is needed before concluding.
        error_message (Optional[str]): A detailed message if an error occurred during the execution or verification of the `current_step_index`.
            The planner must read this to understand and address failures during re-planning. Optional.
    """
    query: str
    file_path: Optional[str]
    current_plan: Annotated[List["PlanStep"], operator.add]
    current_step_index: int
    current_agent_output: Any
    verification_status: Optional[VerificationStatus]
    retry_count: int
    history: Annotated[List["HistoryEntry"], operator.add]
    final_answer: Optional[str]
    confidence: ConfidenceLevel
    error_message: Optional[str]
