from langchain_core.language_models import BaseChatModel  # For type hinting the LLM
from agents.state import AgentState

class PlannerAgent:
    def __init__(self, llm: BaseChatModel):
        """Initializes the Planner node."""
        self.llm = llm
        print("Planner: Initialized (empty logic)")

    def __call__(self, state: AgentState) -> AgentState:
        """Planner: Placeholder for planning and verification logic."""
        print("Executing: Planner node (empty logic)")
        return state # Returns an empty dict for now, no state change