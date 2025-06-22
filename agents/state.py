import operator
from typing import List, Optional, Annotated
from langchain_core.messages import BaseMessage # Import BaseMessage from langchain_core.messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt.chat_agent_executor import AgentState


class GaiaState(AgentState):
    """
    Represents the simplified state of the LangGraph workflow inherits from AgentState.

    Attributes:
        input (str): The initial user input that kicked off the workflow.
        final_answer (Optional[str]): The definitive, synthesized answer to the `query`,
                                      populated by the `final_agent` when ready.

        messages, is_last_step and remaining_steps are derived from AgentState class

    """
    input: str
    final_answer: Optional[str]

