# agents/state.py (ADD THIS FUNCTION AT THE END OF THE FILE)

import inspect
import operator
from typing import Literal
from agents.state import *


def create_type_string(typed_dict_class: type) -> str:
    """
    Generates a TypeScript-like string representation of a TypedDict class.
    This includes the class's primary docstring and a listing of its attributes with their type hints.
    It's designed to be used in LLM prompts to provide clear schema definitions.

    Args:
        typed_dict_class (type): The TypedDict class (e.g., PlanStep, HistoryEntry, AgentState)
                                 for which to generate the string.

    Returns:
        str: A formatted string representing the TypedDict's structure suitable for an LLM prompt.
    """
    class_name = typed_dict_class.__name__
    # Get the main docstring for the class
    class_doc = typed_dict_class.__doc__ if typed_dict_class.__doc__ else f"Represents the {class_name} structure."

    # Start the string with the class name and its main docstring
    type_string = f"### {class_name}\n"
    type_string += f"{class_doc.strip()}\n\n"
    type_string += f"```typescript\ntype {class_name} = {{\n"

    # Iterate through the type hints to get field names and their string representations
    for field_name, field_type in typed_dict_class.__annotations__.items():
        # Exclude internal LangGraph annotations if they are not meant for the prompt
        if field_name == '__annotations__' and isinstance(field_type, tuple) and field_type[1] == operator.add:
            continue  # Skip LangGraph's internal merge annotation if it's there

        # Get a cleaner string representation of the type
        # This tries to remove "typing." prefix and module paths for brevity in the prompt
        type_str = str(field_type).replace("typing.", "").replace("agents.state.", "").replace("<class '", "").replace(
            "'>", "")
        # Handle specific Literal types for better readability if needed
        if isinstance(field_type, type(Literal)):
            type_str = str(field_type).replace("typing.Literal", "Literal").replace("'", '"')

        type_string += f"    {field_name}: {type_str};\n"

    type_string += "}}\n```\n"
    return type_string

if __name__ == "__main__":
    print("\n--- Testing create_type_string for AgentState ---")
    print(create_type_string(AgentState))