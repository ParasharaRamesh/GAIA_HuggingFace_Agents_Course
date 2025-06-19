import operator
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage # Import BaseMessage from langchain_core.messages

class AgentState(TypedDict):
    """
    Represents the simplified state of the LangGraph workflow.

    Attributes:
        query (str): The initial user query that kicked off the workflow.
        messages (List[BaseMessage]): A list of all messages (inputs, outputs, thoughts)
                                      in the conversation. New messages are appended.
        final_answer (Optional[str]): The definitive, synthesized answer to the `query`,
                                      populated by the `final_agent` when ready.
        remaining_steps (int): The number of steps required to complete the `query`.
    """
    query: str
    messages: Annotated[List[BaseMessage], operator.add]
    final_answer: Optional[str]
    remaining_steps: int

