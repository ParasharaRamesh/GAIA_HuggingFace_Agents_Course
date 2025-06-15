from typing import Dict, Any

from agents.model import BaseChatModel


class GenericAgent:
    def __init__(self, llm: BaseChatModel):
        """Initializes the GenericAgent node."""
        self.llm = llm
        print("GenericAgent: Initialized (empty logic)")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """GenericAgent: Placeholder for last-resort answer generation."""
        print("Executing: GenericAgent node (empty logic)")
        return {}