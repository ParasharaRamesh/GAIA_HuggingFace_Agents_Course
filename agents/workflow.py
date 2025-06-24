from functools import partial

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.chat_agent_executor import AgentState

from agents.audio import create_audio_agent
from agents.researcher import create_researcher_agent
from agents.state import GaiaState, SubAgentState
from agents.generic import create_generic_agent
from agents.llm import create_orchestrator_llm, create_generic_llm, create_researcher_llm, create_audio_llm, \
    create_visual_llm
from agents.orchestrator import create_orchestrator_agent
from agents.visual import create_visual_agent
from tools.visual_tools import read_image_and_encode


# Helper functions
def find_last_tool_call_id(messages: list) -> str | None:
    """Finds the ID of the last tool call in the message history."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[0]['id']
    return None


# Nodes
def router_node(state: GaiaState) -> dict:
    print("---ROUTER NODE---")
    updates = {"subagent_output": None}

    # Find the last AIMessage to get the tool call
    last_ai_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            last_ai_message = message
            break

    if last_ai_message and last_ai_message.tool_calls:
        tool_call = last_ai_message.tool_calls[0]
        tool_name = tool_call['name']

        if tool_name.startswith("delegate_to_"):
            agent_name = tool_name.replace("delegate_to_", "").replace("_agent", "")
            arguments = tool_call['args']
            print(f"  Delegating to '{agent_name}' with args: {arguments}")
            updates['current_agent_name'] = agent_name
            updates['subagent_input'] = arguments

        elif tool_name == 'provide_final_answer':
            final_answer = tool_call['args']['answer']
            print(f"  Final answer provided by orchestrator.")
            updates['final_answer'] = final_answer

    return updates


def sub_agent_node(state: GaiaState, agent_runnable, agent_name: str) -> dict:
    """
    This node function manages the execution of a sub-agent.
    It creates an isolated environment, runs the agent, and processes its output.
    """
    print(f"---SUB AGENT NODE: {agent_name}---")

    task_args = state.get("subagent_input", {})
    final_answer = ""

    if agent_name == 'visual':
        final_answer = pre_visual_state_logic(agent_runnable, task_args)
    else:
        final_answer = pre_subagent_state_logic(agent_name, agent_runnable, task_args)

    print(f" {agent_name} agent finished execution.")
    tool_call_id = find_last_tool_call_id(state['messages'])
    report_message = ToolMessage(
        content=final_answer,
        tool_call_id=tool_call_id
    )
    print("Prepared report for orchestrator.")

    # Return all updates to the main state
    return {
        "messages": [report_message],
        "subagent_output": final_answer,
        "current_agent_name": None,  # Clear name
        "subagent_input": None  # Clear input
    }


def pre_subagent_state_logic(agent_name, agent_runnable, task_args):
    formatted_input_string = " | ".join([f"{key}=>'{value}'" for key, value in task_args.items()])
    sub_agent_bubble_state = SubAgentState(
        input=formatted_input_string,
        messages=[HumanMessage(content=formatted_input_string)]
    )
    print(f"Prepared bubble state for {agent_name} with formatted input: {formatted_input_string}")
    final_sub_agent_state = agent_runnable.invoke(sub_agent_bubble_state)
    final_answer = final_sub_agent_state['messages'][-1].content
    return final_answer


def pre_visual_state_logic(agent_runnable, task_args):
    print("  Handling special case for visual agent.")
    query = task_args.get('query', 'Describe this image.')
    file_path = task_args.get('file_path')
    if not file_path:
        final_answer = "Error: No file_path provided for visual agent."
    else:
        encoded_image = read_image_and_encode(file_path)  #

        if "Error:" in encoded_image:
            final_answer = encoded_image
        else:
            multimodal_message = HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": encoded_image}}
                ]
            )
            # Invoke the simple LLM chain directly with the multimodal message
            response_message = agent_runnable.invoke([multimodal_message])
            final_answer = response_message.content

    return final_answer


# Routing functions
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

        researcher_llm = create_researcher_llm()
        print("Researcher LLM initialized.\n")

        audio_llm = create_audio_llm()
        print("Audio LLM initialized.\n")

        visual_llm = create_visual_llm()
        print("Visual LLM initialized.\n")

        # TODO. introduce other LLMs later on
    except Exception as e:
        print(f"Error initializing LLMs. Ensure API keys are set: {e}\n")
        raise

    workflow = StateGraph(GaiaState)

    # orchestrator
    orchestrator_agent = create_orchestrator_agent(orchestrator_llm)
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.set_entry_point("orchestrator")

    # generic
    generic_agent = create_generic_agent(generic_llm)
    generic_agent_node_func = partial(
        sub_agent_node,
        agent_runnable=generic_agent,
        agent_name="generic"
    )
    workflow.add_node("generic", generic_agent_node_func)
    workflow.add_edge("generic", "orchestrator")

    # researcher
    researcher_agent = create_researcher_agent(researcher_llm)
    researcher_agent_node_func = partial(
        sub_agent_node,
        agent_runnable=researcher_agent,
        agent_name="researcher"
    )
    workflow.add_node("researcher", researcher_agent_node_func)
    workflow.add_edge("researcher", "orchestrator")

    # audio
    audio_agent = create_audio_agent(audio_llm)
    audio_agent_node_func = partial(
        sub_agent_node,
        agent_runnable=audio_agent,
        agent_name="audio"
    )
    workflow.add_node("audio", audio_agent_node_func)
    workflow.add_edge("audio", "orchestrator")

    # visual
    visual_agent = create_visual_agent(visual_llm)
    visual_agent_node_func = partial(
        sub_agent_node,
        agent_runnable=visual_agent,
        agent_name="visual"
    )
    workflow.add_node("visual", visual_agent_node_func)
    workflow.add_edge("visual", "orchestrator")

    # router
    workflow.add_node("router", router_node)
    workflow.add_edge("orchestrator", "router")

    # conditional edges
    workflow.add_conditional_edges(
        "router",
        route_by_agent_name,
        {
            "generic": "generic",
            "researcher": "researcher",
            "audio": "audio",
            "visual": "visual",
            END: END
        }
    )

    app = workflow.compile()
    print("LangGraph workflow compiled successfully.")
    return app
