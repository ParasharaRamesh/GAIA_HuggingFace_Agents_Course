# Langgraph orchestration code

import io
from PIL import Image as PILImage

from langgraph.graph import StateGraph
from IPython.display import Image, display
from langgraph.graph.graph import START, END
from typing import Dict, Any

# Import your agent classes from their respective files
from planner import Planner
from researcher import ResearcherAgent
from visual_agent import VisualAgent
from audio_agent import AudioAgent
from final_agent import FinalAgent
from code_agent import CodeAgent

from state import *

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
    Placeholder router for the Planner.
    This logic will be filled in later. For now, it simply points to 'researcher' for visualization.
    """
    print("Routing logic: From Planner (empty logic). Directing to 'researcher'.")
    return "researcher" # Hardcoded for initial graph visualization


def route_from_specialized_agent(state: AgentState) -> str:
    """
    Placeholder router for specialized agents (Researcher, Audio, Code, Visual).
    This logic will be filled in later for verification/re-planning.
    For now, it simply points back to 'planner' for visualization.
    """
    print("Routing logic: From Specialized Agent (empty logic). Directing to 'planner'.")
    return "planner" # Hardcoded for initial graph visualization


# --- Build the LangGraph Workflow ---

# Initialize the workflow with a generic dictionary for state
workflow = StateGraph(AgentState)

# Instantiate your agents with the dummy LLM
dummy_llm_instance = DummyLLM()

planner = Planner(llm=dummy_llm_instance)
researcher = ResearcherAgent(llm=dummy_llm_instance)
audio_agent = AudioAgent(llm=dummy_llm_instance)
code_agent = CodeAgent(llm=dummy_llm_instance)
visual_agent = VisualAgent(llm=dummy_llm_instance)
final_agent = FinalAgent(llm=dummy_llm_instance)


# Add all nodes to the graph
workflow.add_node("planner", planner)
workflow.add_node("researcher", researcher)
workflow.add_node("audio_agent", audio_agent)
workflow.add_node("code_agent", code_agent)
workflow.add_node("visual_agent", visual_agent)
workflow.add_node("final_agent", final_agent)


# Set the entry point
workflow.set_entry_point("planner")

# Set the finish point
workflow.set_finish_point("final_agent")


# Add edges:

# 1. From Planner (conditional to various specialized agents)
workflow.add_conditional_edges(
    "planner",                 # Source node
    route_from_planner,        # Router function
    {                          # Mapping of router output to target node
        "researcher": "researcher",
        "audio_agent": "audio_agent",
        "code_agent": "code_agent",
        "visual_agent": "visual_agent",
        "final_agent": "final_agent"
    }
)

# 2. From each specialized agent back to the Planner (for review/re-planning)
workflow.add_edge("researcher", "planner")
workflow.add_edge("audio_agent", "planner")
workflow.add_edge("code_agent", "planner")
workflow.add_edge("visual_agent", "planner")

# Compile the graph
app = workflow.compile()
