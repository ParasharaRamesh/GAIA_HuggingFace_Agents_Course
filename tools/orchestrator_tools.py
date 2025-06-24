from langchain_core.tools import tool
from agents import *

'''Hooks used by orchestrator'''
def extract_final_answer_hook(state: GaiaState) -> dict:
    """
    A post-model hook for the orchestrator (supervisor) agent.
    It inspects the last AI message in the 'messages' list of the AgentState
    for a "Final Answer:" pattern and extracts it into the 'final_answer' field.

    Args:
        state (GaiaState): The current state of the LangGraph workflow.

    Returns:
        dict: A dictionary representing an update to the AgentState (e.g., {"final_answer": "..."}).
              Returns an empty dict if no final answer is found, indicating no state update.
    """
    messages = state.get("messages", [])  # Access messages from the state
    if not messages:
        return {}

    # The last message is the supervisor's output that we need to inspect
    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        content = last_message.content
        if content:
            # Use regex to find "Final Answer:" (case-insensitive) anywhere in the content
            # and capture everything after it. re.DOTALL makes '.' match newlines too.
            match = re.search(r"final answer:(.*)", content, re.IGNORECASE | re.DOTALL)
            if match:
                final_answer_content = match.group(1).strip()
                print(f"DEBUG: Final answer extracted by hook: {final_answer_content}")  # For debugging
                return {"final_answer": final_answer_content}  # Return the update for the 'final_answer' field
    return {}  # No updat


'''Tools used by orchestrator'''
@tool
def delegate_to_generic_agent(query: str) -> str:
    """
    Delegates the task to the 'generic' agent. Use this as a default when no other
    specialized agent is a clear fit, or for general information, text generation,
    or cross-referencing where other agents might have struggled.
    The 'query' should be a clear, self-contained instruction for the generic agent.
    """
    return f"Delegating to generic agent with query: {query}"

@tool
def provide_final_answer(answer: str) -> str:
    """
    Provides the final answer to the user's overall request. Use this tool when
    you have successfully gathered all necessary information and formulated a
    comprehensive response. This signals the end of the workflow.
    """
    return f"Final Answer: {answer}"