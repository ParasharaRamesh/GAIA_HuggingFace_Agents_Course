from typing import Optional

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
def delegate_to_audio_agent(query: str, file_path: Optional[str] = None, youtube_url: Optional[str] = None) -> str:
    """
    Delegates a task to the 'audio' agent. Use this for any task that involves
    transcribing an audio file or a YouTube video. You MUST provide either a
    'file_path' to a local audio file or a 'youtube_url'.
    """
    return f"Delegating to audio agent with query: {query}, file_path: {file_path}, youtube_url: {youtube_url}"

@tool(return_direct=True)
def delegate_to_visual_agent(query: str, file_path: Optional[str] = None) -> str:
    """
    Delegates a task to the 'visual' agent. Use this for any task that involves
    analyzing an image. You MUST provide a 'file_path' to a local image file.
    """
    return f"Delegating to visual agent with query: {query}, file_path: {file_path}"

@tool(return_direct=True)
def delegate_to_code_agent(query: str, code_path: Optional[str] = None, input_path: Optional[str] = None) -> str:
    """
    Delegates a task to the 'code' agent. Use this for any task that involves generating and running code.

    code_path could refer to the local path to the provided python file to be run & executed / generated python file to be run & executed
    input_path could refer to the local path to any provided input file which needs to be parsed by the python program to be generated/excecuted

    """
    return f"Delegating to code agent with query: {query}, code_path: {code_path}, input_path: {input_path}"


@tool(return_direct=True)
def provide_final_answer(answer: str) -> str:
    """
    Provides the final answer to the user's overall request. Use this tool when
    you have successfully gathered all necessary information and formulated a
    comprehensive response. This signals the end of the workflow.
    """
    return f"Final Answer: {answer}"