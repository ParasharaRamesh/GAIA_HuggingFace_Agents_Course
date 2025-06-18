# Langgraph orchestration code

import io
from PIL import Image as PILImage

from langgraph.graph import StateGraph
from IPython.display import Image, display
from langgraph.graph.graph import START, END
from typing import Dict, Any

from .state import *

from .researcher_agent import ResearcherAgent
from .planner_agent import PlannerAgent
from .visual_agent import VisualAgent
from .audio_agent import AudioAgent
from .generic_agent import FinalAgent
from .code_agent import CodeAgent

# Dummy LLM for instantiation (replace with your actual LLM, e.g., OpenAI, Gemini)
# This is here just so the agent classes can be initialized.
class DummyLLM:
    def __init__(self):
        print("DummyLLM: Initialized (placeholder)")
    def invoke(self, messages, **kwargs):
        return "dummy response"

# --- Routing Functions (Skeletons with empty logic) ---

def route_from_planner(state: AgentState) -> str:
    """
    Routes based on the planner_agent's decision stored in active_agent_name.
    """
    # The planner_agent's __call__ method should have already updated
    # state.active_agent_name with its chosen next agent.
    next_agent_name = state.get("active_agent_name") # Get the agent name decided by the planner

    if next_agent_name:
        print(f"Routing from Planner to: {next_agent_name}")
        return next_agent_name
    else:
        # This case should ideally not happen if planner always sets active_agent_name
        # Fallback to final_agent if the planner fails to specify
        print("Error: Planner did not specify a next agent. Routing to final_agent.")
        return "final_agent"


def route_from_specialized_agent(state: AgentState) -> str:
    """
    Routes back to the planner_agent after a specialized agent has executed.
    This ensures the Planner always regains control to review output and plan next steps.
    """
    print(f"Routing from {state.get('active_agent_name', 'a specialized agent')} back to Planner for re-evaluation.")
    return "planner_agent"


# --- Build the LangGraph Workflow ---

# Initialize the workflow with a generic dictionary for state
workflow = StateGraph(AgentState)

# Instantiate your agents with the dummy LLM
dummy_llm_instance = DummyLLM()

planner_agent = PlannerAgent(llm=dummy_llm_instance)
final_agent = FinalAgent(llm=dummy_llm_instance)
visual_agent = VisualAgent(llm=dummy_llm_instance)
code_agent = CodeAgent(llm=dummy_llm_instance)
audio_agent = AudioAgent(llm=dummy_llm_instance)
researcher_agent = ResearcherAgent(llm=dummy_llm_instance)


# Add all nodes to the graph
workflow.add_node("planner_agent", planner_agent)
workflow.add_node("researcher_agent", researcher_agent)
workflow.add_node("audio_agent", audio_agent)
workflow.add_node("code_agent", code_agent)
workflow.add_node("visual_agent", visual_agent)
workflow.add_node("final_agent", final_agent)


# Set the entry point
workflow.set_entry_point("planner_agent")

# Set the finish point
workflow.set_finish_point("final_agent")


# Add edges:

# 1. From Planner (conditional to various specialized agents)
workflow.add_conditional_edges(
    "planner_agent",                 # Source node
    route_from_planner,        # Router function
    {                          # Mapping of router output to target node
        "researcher_agent": "researcher_agent",
        "audio_agent": "audio_agent",
        "code_agent": "code_agent",
        "visual_agent": "visual_agent",
        "final_agent": "final_agent"
    }
)

# 2. From each specialized agent back to the Planner (for review/re-planning)
workflow.add_edge("researcher_agent", "planner_agent")
workflow.add_edge("audio_agent", "planner_agent")
workflow.add_edge("code_agent", "planner_agent")
workflow.add_edge("visual_agent", "planner_agent")

# Compile the graph
app = workflow.compile()
