# Contains the node with the final answer of the agent
from typing import Dict, Any

from agents.model import BaseChatModel


class FinalAgent:
    def __init__(self, llm: BaseChatModel):
        """Initializes the FinalAgent node."""
        self.llm = llm
        print("FinalAgent: Initialized (empty logic)")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """FinalAgent: Placeholder for final answer synthesis logic."""
        print("Executing: FinalAgent node (empty logic)")
        return {}
