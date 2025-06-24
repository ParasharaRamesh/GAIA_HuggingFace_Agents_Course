from typing import List, Optional, Annotated, Any, Dict
from langgraph.prebuilt.chat_agent_executor import AgentState


class GaiaState(AgentState):
    """
    Represents the simplified state of the LangGraph workflow inherits from AgentState.

    Attributes:
        input (str): The initial user input that kicked off the workflow.
        final_answer (Optional[str]): The definitive, synthesized answer to the `query`,
                                      populated by the `final_agent` when ready.
        subagent_input (Optional[Dict[str, Any]]): Stores the input provided to the currently executing sub-agent.
                                      This helps in tracking what specific task or information was delegated.
        subagent_output (Optional[str]): Stores the output received from the executed sub-agent.
                                         This allows the orchestrator to process and synthesize the sub-agent's results.
        current_agent_name (Optional[str]): Tracks the name of the sub-agent currently active or
                                            most recently active in the workflow.

        messages, is_last_step and remaining_steps are derived from AgentState class

    """
    input: str
    final_answer: Optional[str]
    subagent_input: Optional[Dict[str, Any]]
    subagent_output: Optional[str]
    current_agent_name: Optional[str]

# Define the isolated state for each sub-agent
class SubAgentState(AgentState):
    """
    Represents the isolated state for a sub-agent's internal ReAct loop.
    This state is only visible to the sub-agent during its execution and
    is designed to be compatible with LangGraph's `create_react_agent`.

    Attributes:
        input (str): The specific input or query given to this sub-agent for its current task.
                     This is distinct from the overall workflow's `input`.

        messages, is_last_step and remaining_steps are derived from AgentState class
    """
    input: str