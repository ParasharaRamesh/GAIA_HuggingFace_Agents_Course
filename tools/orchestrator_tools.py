from langchain_core.tools import tool
from agents import *


'''Tools used by orchestrator'''
@tool(return_direct=True)
def delegate_to_generic_agent(query: str) -> str:
    """
    Delegates the task to the 'generic' agent. Use this as a default when no other
    specialized agent is a clear fit, or for general information, text generation,
    or cross-referencing where other agents might have struggled.
    The 'query' should be a clear, self-contained instruction for the generic agent.
    """
    return f"Delegating to generic agent with {query = }"

@tool(return_direct=True)
def delegate_to_researcher_agent(query: str) -> str:
    """
    Delegates the task to the 'researcher' agent. Use this for tasks that
    require deep research, fact-checking, or looking up scientific papers
    on Arxiv and Wikipedia.
    The 'query' should be a clear, self-contained instruction.
    """
    return f"Delegating to researcher agent with {query = }"

@tool(return_direct=True)
def provide_final_answer(answer: str) -> str:
    """
    Provides the final answer to the user's overall request. Use this tool when
    you have successfully gathered all necessary information and formulated a
    comprehensive response. This signals the end of the workflow.
    """
    return f"Final Answer: {answer}"