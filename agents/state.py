import operator
from typing import TypedDict, List, Dict, Any, Union, Literal, Optional
from datetime import datetime

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
    Represents a single, actionable step within the agent's overall plan.
    Each step guides the execution flow by specifying which agent to call and what its objective is.

    Attributes:
        agent_to_call (str): The identifier (string name) of the specific agent node responsible for executing this step.
            Crucial for LangGraph to route the state to the correct agent function for execution.
        input_for_agent (Any): The specific input data or instructions tailored for the `agent_to_call` for this step.
            Crucial for providing context and necessary data for the executing agent to perform its task.
        expected_output_type (str): A description of the expected type or format of the output from the `agent_to_call` for this step.
            Crucial for guiding the executing agent towards a specific result and for the VerifierAgent to know what to check.
        details (str): A high-level, human-readable description of the objective or goal for this plan step.
            Crucial for the GlobalPlanner's reasoning, for logging, and for the VerifierAgent to understand the step's intent.
        substeps (List[str]): A list of high-level checkpoints or sub-goals that the executing agent should aim to achieve within this single plan step.
            Crucial for guiding the internal reasoning of the `agent_to_call` and for the VerifierAgent to perform more granular checks.
        status (PlanStepStatus): The current execution status of this specific plan step (e.g., PENDING, IN_PROGRESS, SUCCESS, FAILED).
            Crucial for tracking progress, determining where to resume execution, and for the GlobalPlanner to perform targeted re-planning.
    """
    agent_to_call: str
    input_for_agent: Any
    expected_output_type: str
    details: str
    substeps: List[str]
    status: PlanStepStatus


class HistoryEntry(TypedDict):
    """
    Represents a single, immutable entry in the chronological history log of the agent's actions and observations.
    Each entry provides an audit trail of what happened, when, by which component, and with what outcome.

    Attributes:
        timestamp (str): An ISO-formatted string indicating when this history event occurred.
            Crucial for maintaining a chronological order and for analyzing the sequence of operations.
        node (str): The identifier (string name) of the LangGraph node or agent that performed this action.
            Crucial for attributing actions and outputs to specific components of the agent system.
        action (str): A high-level description of the specific action performed by the node (e.g., "Generated Plan", "Executed Web Search Tool").
            Crucial for providing context on what the entry represents at a glance.
        input_to_node (Any): A representation of the primary input data or context that the `node` received for this specific action.
            Crucial for debugging and understanding why a particular action was taken. Should be concise if the full input is very large.
        output_from_node (Any): A representation of the primary output or result produced by the `node` for this action.
            Crucial for understanding the outcome of an action. Should be a summary or key findings if the full output is too large.
        status (Optional[Literal["SUCCESS", "FAILED", "SKIPPED", "IN_PROGRESS"]]): The overall status or outcome of this specific history event/action within the node.
            Crucial for quickly assessing whether an operation succeeded or failed.
        error_details (Optional[str]): If the 'status' is "FAILED", this field contains a detailed message or reason for the failure.
            Crucial for in-depth debugging and for the GlobalPlanner to inform its re-planning strategy.
        model_used (Optional[str]): The identifier of the LLM model (e.g., "gemini-pro", "claude-3-opus") that was primarily used for this action, if applicable.
            Crucial for audit trails, performance analysis, and for supporting dynamic model switching or heterogeneous agent models.
        plan_snapshot (Optional[List[PlanStep]]): An optional snapshot of the `current_plan` at the moment this history entry was recorded.
            Crucial specifically when the GlobalPlanner generates or re-generates a plan, providing a historical record of plan evolution.
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
    Represents the complete and dynamic state of the multi-agent system at any given point in its execution.
    This is the central blackboard where all information flows between different agent nodes.
    It is crucial for enabling agents to read necessary context, update progress, and pass data to subsequent steps,
    and provides the LLM with all necessary information to make informed decisions for planning, execution, and re-planning.

    Attributes:
        query (str): The original user query or GAIA question that initiated the agent's task.
            Crucial as the ultimate goal and overall context for all subsequent planning and execution.
        file_path (Optional[str]): An optional file path to a local resource (e.g., audio, image) provided with the initial query.
            Crucial for agents that need to access local file content for processing.
        current_plan (List[PlanStep]): The ordered list of `PlanStep`s representing the agent's current strategy to fulfill the `query`.
            Crucial for guiding the agent's sequential execution and serves as the primary instructions for the executor.
        current_step_index (int): The integer index indicating the current step being executed within the `current_plan`.
            Crucial for tracking immediate progress, routing to the next step, and for resuming execution after a pause or re-plan.
        current_agent_output (Any): The most recent primary output generated by the last executed specialized agent node.
            Crucial for immediate processing by subsequent nodes (e.g., for verification, summarization, or final answer synthesis).
        verification_status (Optional[VerificationStatus]): The outcome of the most recent verification attempt on `current_agent_output` (e.g., "success" or "failure").
            Crucial for the router to decide whether to proceed, retry the current step, or trigger a re-planning phase.
        retry_count (int): A counter for how many times the *current* `PlanStep` has been retried after a `FAILED` verification status.
            Crucial for preventing infinite loops and for triggering a re-plan by the GlobalPlanner after a threshold is met.
        history (List[HistoryEntry]): A chronological, append-only list of `HistoryEntry` objects, detailing all significant actions and observations.
            Crucial for debugging, providing comprehensive context for the LLM (especially the GlobalPlanner for reflection and re-planning), and auditing.
        final_answer (Optional[str]): The agent's synthesized answer to the original `query` once all necessary steps are completed and verified.
            Crucial as the ultimate output of the agent's task.
        confidence (ConfidenceLevel): The agent's assessment of its confidence level ("high" or "low") in the `final_answer`.
            Crucial for informing downstream systems or the user about the reliability of the provided answer.
        error_message (Optional[str]): A detailed message regarding any *current* error encountered during the most recent step's execution or verification.
            Crucial for providing immediate error feedback to the GlobalPlanner, enabling rapid problem-solving or re-planning for the current issue.
    """
    query: str
    file_path: Optional[str]
    current_plan: List[PlanStep]
    current_step_index: int
    current_agent_output: Any
    verification_status: Optional[VerificationStatus]
    retry_count: int
    history: List[HistoryEntry]
    final_answer: Optional[str]
    confidence: ConfidenceLevel
    error_message: Optional[str]

    __annotations__ = {
        "current_plan": (List[PlanStep], operator.add),
        "history": (List[HistoryEntry], operator.add),
    }
