from functools import partial

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

from agents.state import GaiaState, SubAgentState
from agents.generic import create_generic_agent
from agents.llm import create_orchestrator_llm, create_generic_llm
from agents.orchestrator import create_orchestrator_agent

# Helper functions
def find_last_tool_call_id(messages: list) -> str | None:
    """Finds the ID of the last tool call in the message history."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[0]['id']
    return None

# Nodes
def router_node(state: GaiaState) -> dict:
    """
    This node prepares the state for the next agent's turn.
    It sets the next agent's name and input, and clears old output.
    """
    print("---ROUTER NODE---")
    updates = {"subagent_output": None}  # Always clear the last output
    last_message = state['messages'][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]['name']

        # We can create a convention for our delegation tools
        if tool_name.startswith("delegate_to_"):
            # This logic works for ANY agent, not just generic
            agent_name = tool_name.replace("delegate_to_", "").replace("_agent", "")
            arguments = last_message.tool_calls[0]['args']

            print(f"  Delegating to '{agent_name}' with args: {arguments}")
            updates['current_agent_name'] = agent_name
            # Pass the entire arguments dict. This future-proofs us for agents that might need a 'file_path' in addition to a 'query'.
            updates['subagent_input'] = arguments

    return updates

def sub_agent_node(state: GaiaState, agent_runnable, agent_name: str) -> dict:
    """
    This node function manages the execution of a sub-agent.
    It creates an isolated environment, runs the agent, and processes its output.
    """
    print(f"---SUB AGENT NODE: {agent_name}---")

    # Part A: Prepare the "Bubble" state for the sub-agent
    task_input = state.get("subagent_input")
    sub_agent_bubble_state = SubAgentState(
        input=task_input,
        messages=[HumanMessage(content=task_input)]
    )
    print(f"  Prepared bubble state for {agent_name}.")

    # Part B: Execute the Agent
    final_sub_agent_state = agent_runnable.invoke(sub_agent_bubble_state)
    print(f"  {agent_name} agent finished execution.")

    # Part C: Process the Result & Clean Up
    final_answer = final_sub_agent_state['messages'][-1].content
    tool_call_id = find_last_tool_call_id(state['messages'])

    # Create the ToolMessage report for the orchestrator
    report_message = ToolMessage(
        content=final_answer,
        tool_call_id=tool_call_id
    )
    print("Prepared report for orchestrator.")

    # Return all updates to the main state
    return {
        "messages": [report_message],
        "subagent_output": final_answer,
        "current_agent_name": None, # Clear name
        "subagent_input": None      # Clear input
    }

#Routing functions
def route_by_agent_name(state: GaiaState) -> str:
    """
    Determines the next step after the orchestrator has run by inspecting the agent state
    """
    if next_agent := state.get("current_agent_name"):
        print(f"Routing to agent: {next_agent}")
        return next_agent
    else:
        print("No agent designated. Ending workflow.")
        return END

# Entire workflow
def create_worfklow():
    try:
        orchestrator_llm = create_orchestrator_llm()
        print("Orchestrator LLM initialized\n")

        generic_llm = create_generic_llm()
        print("Generic LLM initialized.\n")

        #TODO. introduce other LLMs later on
    except Exception as e:
        print(f"Error initializing LLMs. Ensure API keys are set: {e}\n")
        raise

    workflow = StateGraph(GaiaState)

    orchestrator_agent = create_orchestrator_agent(orchestrator_llm)
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.set_entry_point("orchestrator")

    workflow.add_node("router", router_node)

    generic_agent = create_generic_agent(generic_llm)
    generic_agent_node_func = partial(
        sub_agent_node,
        agent_runnable=generic_agent,
        agent_name="generic"
    )
    workflow.add_node("generic", generic_agent_node_func)

    # add edges
    workflow.add_edge("orchestrator", "router")  # Orchestrator always goes to the router node
    workflow.add_conditional_edges(
        "router",  # The router node then decides the path
        route_by_agent_name,
        {"generic": "generic", END: END}
    )

    workflow.add_edge("generic", "orchestrator")

    app = workflow.compile()
    print("LangGraph workflow compiled successfully.")
    return app
