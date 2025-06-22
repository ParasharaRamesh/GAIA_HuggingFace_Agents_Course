import operator
from typing import Literal, List, Dict, Any, Callable
import re
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from agents.state import GaiaState

''' util functions '''
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


def create_clean_agent_messages_hook(agent_name: str) -> Callable[[GaiaState], Dict[str, Any]]:
    """
    Factory function to create a pre-model hook that cleans message history for a specific agent.
    It now explicitly handles extracting the supervisor's 'Action Input' as the effective
    'Human Input' for the delegated agent, and manages the agent's scratchpad.

    Retains:
    1. The first SystemMessage from the history.
    2. The most recent 'Action Input' from an AIMessage (presumably the supervisor's delegation)
       transformed into a HumanMessage. This becomes the primary input for the agent's current turn.
       If no such delegation is found, it falls back to the last actual HumanMessage.
    3. All subsequent AIMessages and ToolMessages where:
       a. Their 'name' field matches the provided 'agent_name'.
       b. They are an AIMessage and their 'name' field is None (assuming it's the agent's direct thought).

    Messages older than the effective 'Human Input' (excluding the SystemMessage and agent's scratchpad)
    are removed.

    Args:
        agent_name (str): The name of the agent this hook is being created for.

    Returns:
        Callable[[List[BaseMessage]], Dict[str, Any]]: The actual hook function
                                                        that `create_react_agent` expects.
    """

    def _clean_agent_messages_hook_instance(state: GaiaState) -> Dict[str, Any]:
        new_messages: List[BaseMessage] = []

        # 1. Collect the first SystemMessage if present
        system_msg = None
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                system_msg = message
                break
        if system_msg:
            new_messages.append(system_msg)

        # Initialize delegated_content to ensure it's always defined
        delegated_content = ""
        effective_human_message = None
        start_index_for_agent_scratchpad = 0

        # 2. Identify the effective Human Input for this agent's turn.
        #    This is primarily the 'Action Input' from the most recent supervisor delegation.
        for i in reversed(range(len(messages))):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                match = re.search(r"Action Input:\s*(.*)", msg.content, re.IGNORECASE | re.DOTALL)
                if match:
                    # Extract and store the delegated content
                    delegated_content = match.group(1).strip().strip('"')
                    effective_human_message = HumanMessage(content=delegated_content)
                    start_index_for_agent_scratchpad = i + 1
                    break
            # Fallback to the last actual HumanMessage if no explicit supervisor delegation message is found
            elif isinstance(msg, HumanMessage) and effective_human_message is None:
                effective_human_message = msg
                # Capture its content as the "input" in this fallback scenario
                delegated_content = msg.content
                start_index_for_agent_scratchpad = i + 1

        print(f"Delegated Action Input of Supervisor extracted is {delegated_content}")
        # Add the effective Human Message (either delegated input or original human query)
        if effective_human_message:
            new_messages.append(effective_human_message)
        else:
            # If no effective human message at all (unlikely in typical flows), start scratchpad from beginning
            start_index_for_agent_scratchpad = 0

        # 3. Collect agent's own ReAct scratchpad (AIMessage, ToolMessage matching its name)
        #    starting from after the effective Human Message.
        for i in range(start_index_for_agent_scratchpad, len(messages)):
            current_message = messages[i]

            # Skip SystemMessage if it's already added at the beginning
            if current_message is system_msg and current_message in new_messages:
                continue

            # Skip the effective HumanMessage if it was just added to prevent duplicates
            if current_message is effective_human_message:
                continue

            msg_name = getattr(current_message, 'name', None)

            # Keep if the message's name explicitly matches the current agent's name
            if msg_name == agent_name:
                new_messages.append(current_message)
                continue

            # Also, keep AIMessages that do not have an explicit 'name' set,
            # assuming these are direct outputs (thoughts, actions) from the LLM itself
            # within the agent's current chain.
            if isinstance(current_message, AIMessage) and msg_name is None:
                new_messages.append(current_message)
                continue

        # Return the final cleaned message list AND the updated 'input' key.
        # Returning 'input' here tells LangGraph to merge this value into the state's 'input' field.
        return {
            "messages": new_messages,
            "input": delegated_content
        }

    return _clean_agent_messages_hook_instance
