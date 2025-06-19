import operator
from typing import Literal, List, Dict, Any, Callable

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, RemoveMessage, AIMessage, ToolMessage

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


def create_clean_agent_messages_hook(agent_name: str) -> Callable[[List[BaseMessage]], Dict[str, Any]]:
    """
    Factory function to create a pre-model hook that cleans message history for a specific agent.
    It retains:
    1. The first SystemMessage from the history.
    2. The last HumanMessage from the history.
    3. All subsequent AIMessages and ToolMessages where:
       a. Their 'name' field matches the provided 'agent_name'.
       b. They are an AIMessage and their 'name' field is None (assuming it's the agent's direct thought).
    Messages older than the last HumanMessage (excluding the SystemMessage) are removed.

    This ensures the agent maintains a clear context of its instructions, the user's latest query,
    and its own ongoing internal thought-action-observation cycle, while pruning irrelevant past history.

    Args:
        agent_name (str): The name of the agent this hook is being created for.
                          This should match the 'name' attribute you set when
                          creating your agent (e.g., 'researcher-agent').

    Returns:
        Callable[[List[BaseMessage]], Dict[str, Any]]: The actual hook function
                                                        that `create_react_agent` expects.
    """

    # This inner function is the actual hook that will be passed to pre_model_hook.
    # It "remembers" the agent_name from its outer scope (the factory function).
    def _clean_agent_messages_hook(messages: List[BaseMessage]) -> Dict[str, Any]:
        new_messages: List[BaseMessage] = []

        # 1. Collect the first SystemMessage if present
        system_msg = None
        for message in messages:
            if isinstance(message, SystemMessage):
                system_msg = message
                break
        if system_msg:
            new_messages.append(system_msg)

        # 2. Find the index of the last HumanMessage
        last_human_message_idx = -1
        for i in reversed(range(len(messages))):
            if isinstance(messages[i], HumanMessage):
                last_human_message_idx = i
                break

        # 3. If a HumanMessage is found, add it and then iterate from its position
        #    to collect relevant subsequent messages.
        if last_human_message_idx != -1:
            for i in range(last_human_message_idx, len(messages)):
                current_message = messages[i]

                # Skip if this is the SystemMessage and it's already been added (at the beginning)
                if current_message is system_msg and current_message in new_messages:
                    continue

                # Always include the HumanMessage (which is at last_human_message_idx)
                if isinstance(current_message, HumanMessage):
                    new_messages.append(current_message)
                    continue

                # For AIMessages and ToolMessages, apply name-based filtering
                if isinstance(current_message, (AIMessage, ToolMessage)):
                    msg_name = getattr(current_message, 'name', None)

                    # Keep if the message's name explicitly matches the current agent's name
                    if msg_name == agent_name:
                        new_messages.append(current_message)
                        continue

                    # Also, keep AIMessages that do not have an explicit 'name' set.
                    # These are often the direct outputs (thoughts, actions) from the LLM itself
                    # within the agent's chain, before a specific tool or agent name is assigned.
                    if isinstance(current_message, AIMessage) and msg_name is None:
                        new_messages.append(current_message)
                        continue

                # Any other message types or named messages that don't match the agent_name criteria are skipped.

        # Return the final cleaned message list in the format required by LangGraph's pre_model_hook.
        # RemoveMessage(id=RemoveMessage.REMOVE_ALL_MESSAGES) is crucial for overwriting the state.
        return {
            "messages": [RemoveMessage(id=RemoveMessage.REMOVE_ALL_MESSAGES), *new_messages]
        }

    return _clean_agent_messages_hook
