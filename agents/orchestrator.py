import os
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph_supervisor import create_supervisor

from agents.state import AgentState
from agents.visual import create_visual_agent
from agents.audio import create_audio_agent
from agents.researcher import create_researcher_agent
from agents.interpreter import create_code_agent
from agents.generic import create_generic_agent


def extract_final_answer_hook(state: AgentState) -> dict:
    """
    A post-model hook for the orchestrator (supervisor) agent.
    It inspects the last AI message in the 'messages' list of the AgentState
    for a "Final Answer:" pattern and extracts it into the 'final_answer' field.

    Args:
        state (AgentState): The current state of the LangGraph workflow.

    Returns:
        dict: A dictionary representing an update to the AgentState (e.g., {"final_answer": "..."}).
              Returns an empty dict if no final answer is found, indicating no state update.
    """
    messages = state.get("messages", []) # Access messages from the state
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
                print(f"DEBUG: Final answer extracted by hook: {final_answer_content}") # For debugging
                return {"final_answer": final_answer_content} # Return the update for the 'final_answer' field
    return {} # No updat

def create_master_orchestrator_workflow(
    orchestrator_llm: BaseChatModel,
    visual_llm: BaseChatModel,
    audio_llm: BaseChatModel, # Corrected type hint
    researcher_llm: BaseChatModel,
    interpreter_llm: BaseChatModel,
    generic_llm: BaseChatModel,
):
    """
    Creates and returns the main LangGraph workflow orchestrated by a supervisor.
    This supervisor acts as the central planner, delegating tasks to specialized agents.
    Each agent receives its own dedicated LLM instance for maximum flexibility.
    The workflow utilizes the shared AgentState and populates its 'final_answer' field
    via a post-model hook when the orchestrator provides the final response.
    """
    # 1. Instantiate specialized agents, passing their specific LLM
    generic = create_generic_agent(llm=generic_llm) # Corrected parameter name to 'llm'
    audio = create_audio_agent(llm=audio_llm)
    research = create_researcher_agent(llm=researcher_llm)
    code = create_code_agent(llm=interpreter_llm)
    visual = create_visual_agent(llm=visual_llm)

    specialized_agents = [
        visual,
        audio,
        research,
        code,
        generic
    ]

    # 2. Load the orchestrator prompt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    orchestrator_prompt_path = os.path.join(prompts_dir, 'orchestrator_prompt.txt')

    orchestrator_prompt_content = ""
    try:
        with open(orchestrator_prompt_path, "r", encoding="utf-8") as f:
            orchestrator_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Orchestrator prompt file not found at {orchestrator_prompt_path}")
        orchestrator_prompt_content = (
            "You are a master orchestrator managing a team of specialized experts."
            "Available agents: {agent_names}. Delegate tasks and provide a final answer when the overall goal is achieved."
            "\n\n{input}\nThought:{agent_scratchpad}"
        )

    # 3. Create the supervisor workflow
    workflow = create_supervisor(
        agents=specialized_agents,
        model=orchestrator_llm,
        prompt=orchestrator_prompt_content,
        state_schema=AgentState,
        post_model_hook=extract_final_answer_hook
    )

    return workflow