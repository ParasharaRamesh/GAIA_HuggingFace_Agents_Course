import operator
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage # Import BaseMessage from langchain_core.messages
from langgraph.managed import RemainingSteps


class AgentState(TypedDict):
    """
    Represents the simplified state of the LangGraph workflow.

    Attributes:
        input (str): The initial user input that kicked off the workflow.
        messages (List[BaseMessage]): A list of all messages (inputs, outputs, thoughts)
                                      in the conversation. New messages are appended.
        final_answer (Optional[str]): The definitive, synthesized answer to the `query`,
                                      populated by the `final_agent` when ready.
        remaining_steps (int): The number of steps required to complete the `query`.
    """
    input: str
    messages: Annotated[List[BaseMessage], operator.add]
    final_answer: Optional[str]
    remaining_steps: Optional[RemainingSteps] = 30

