from langchain_core.tools import tool
from agents import *


'''Tools used by orchestrator'''
@tool(return_direct=True)
def delegate_to_generic_agent(query: str) -> str:
    """
    Use this for simple, general, or conversational questions.
    **DO NOT use this for any questions that involve academic papers, scientific research, financial analysis, or deep factual lookups.**
    For those, you MUST use the 'delegate_to_researcher_agent'.
    """
    return f"Delegating to generic agent with query: {query}"


@tool(return_direct=True)
def delegate_to_researcher_agent(query: str) -> str:
    """
    Use this for any task that requires deep research, fact-checking, or looking up information in scientific papers from Arxiv or articles from Wikipedia.
    **This is the ONLY tool for academic or research-related questions.**
    If the user asks about a paper, a specific verifiable fact, or a complex topic, you MUST use this tool.
    """
    return f"Delegating to researcher agent with query: {query}"

@tool(return_direct=True)
def provide_final_answer(answer: str) -> str:
    """
    Provides the final answer to the user's overall request. Use this tool when
    you have successfully gathered all necessary information and formulated a
    comprehensive response. This signals the end of the workflow.
    """
    return f"Final Answer: {answer}"