import datetime
from typing import Any, List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser  # Not strictly used
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

# Import AgentState and HistoryEntry from state.py
from state import AgentState, HistoryEntry, HistoryEntryStatus
from utils.state import create_type_string

# Import tools from audio.py (assuming tools directory is alongside this file or in PYTHONPATH)
from tools.audio import transcribe_audio, get_youtube_transcript


class AudioAgent:
    """
    Agent responsible for handling audio-related tasks, primarily transcription.
    It can transcribe audio files or get transcripts from YouTube URLs.
    This agent handles its own internal tool calls and reports errors back to the Planner.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        # Expose relevant tools to the LLM.
        self.tools = [transcribe_audio, get_youtube_transcript]

        # Define the prompt for the audio agent.
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are the AudioAgent. Your primary goal is to process audio-related tasks "
             "based on the provided `active_agent_task` from the Planner. "
             "Use the available tools (transcribe_audio, get_youtube_transcript) effectively. "
             "If the task involves a YouTube URL, use `get_youtube_transcript` and provide the full URL. "
             "If it involves a local audio file, use `transcribe_audio` and provide the file path.\n"
             "Your output should be the transcribed text or a concise summary if very long.\n"
             "If you encounter an error with a tool or cannot complete the task, report it concisely.\n"
             "\n--- Current AgentState Schema ---\n{agent_state_schema}\n"
             "\n--- HistoryEntry Schema ---\n{history_entry_schema}\n"
             ),
            ("human", "User Query (Overall Goal): {query}\n"
                      "High-level roadmap step: {current_roadmap_step}\n"
                      "**Active Agent Task:** {active_agent_task}\n"
                      "Conversation History with Planner for this sub-task: {conversation_history_with_agent}\n"
                      "Previous output from this agent (if re-attempting this task): {active_agent_output}\n"
                      "Planner's latest feedback for this task: {planner_feedback}\n"
                      "Previous error encountered by this agent (if any): {active_agent_error_message}\n"
                      "\nPerform the audio transcription task and return the text."
             )
        ])

        # Bind the tools to the LLM for function calling capabilities
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def invoke(self, state: AgentState) -> AgentState:
        """
        Executes the audio agent's task based on the current AgentState and updates the state
        with transcription results or an error message.
        """
        print(f"\n--- AudioAgent: Starting task for roadmap step {state['current_roadmap_step_index']} ---")

        # Determine the current high-level roadmap step for context in prompt
        current_roadmap_step = state['high_level_roadmap'][state['current_roadmap_step_index']] \
            if state['high_level_roadmap'] and state['current_roadmap_step_index'] < len(state['high_level_roadmap']) \
            else "N/A"

        # Prepare input for the LLM call using the prompt template
        llm_input = {
            "query": state["query"],
            "current_roadmap_step": current_roadmap_step,
            "active_agent_task": state["active_agent_task"],
            "conversation_history_with_agent": state["conversation_history_with_agent"],
            "active_agent_output": state["active_agent_output"],
            "planner_feedback": state["planner_feedback"],
            "active_agent_error_message": state["active_agent_error_message"],
            "history": state["history"],
            "agent_state_schema": create_type_string(AgentState),
            "history_entry_schema": create_type_string(HistoryEntry)
        }

        agent_output = None
        error_message = None
        tool_calls = []
        status = HistoryEntryStatus.SUCCESS

        try:
            # Invoke the LLM with the prepared input. The LLM might decide to call a tool.
            response = self.llm_with_tools.invoke(self.prompt_template.invoke(llm_input))

            # Check if the LLM decided to call any tools
            if response.tool_calls:
                tool_calls = response.tool_calls
                tool_output_messages = []

                # Iterate through all tool calls suggested by the LLM
                for tool_call in tool_calls:
                    print(f"--- AudioAgent: Calling tool '{tool_call.name}' with args: {tool_call.args} ---")
                    try:
                        # Dynamically find and invoke the tool function from the imported tools
                        tool_func = None
                        for t in self.tools:
                            if t.name == tool_call.name:
                                tool_func = t
                                break

                        if tool_func:
                            tool_result = tool_func.invoke(tool_call.args)
                            tool_output_messages.append(f"Tool {tool_call.name} Output:\n{tool_result}")
                        else:
                            raise ValueError(f"Tool '{tool_call.name}' not recognized by AudioAgent.")

                    except Exception as tool_err:
                        # Capture tool-specific errors
                        tool_output_messages.append(f"Tool {tool_call.name} Error: {tool_err}")
                        error_message = str(tool_err)
                        status = HistoryEntryStatus.FAILED
                        print(f"--- AudioAgent: Tool error occurred: {tool_err} ---")

                # Aggregate tool outputs (or errors) into the agent's main output
                if tool_output_messages:
                    agent_output = "\n".join(tool_output_messages)
                else:
                    agent_output = response.content if response.content else "No direct output from LLM after tool call attempt."

            else:
                # If LLM did not call any tools, its direct response is the agent's output
                agent_output = response.content

        except Exception as e:
            # Capture any errors during the agent's overall execution (LLM call or processing)
            error_message = str(e)
            status = HistoryEntryStatus.FAILED
            print(f"--- AudioAgent: Agent execution error: {e} ---")

        # Create a mutable copy of the state to update
        new_state = state.copy()
        new_state["active_agent_name"] = "AudioAgent"
        new_state["active_agent_output"] = agent_output
        new_state["active_agent_error_message"] = error_message  # Update with error if any

        # Add the agent's response to the conversation history for the current sub-task
        new_state["conversation_history_with_agent"].append({
            "role": "agent",
            "message": agent_output if agent_output is not None else (
                error_message if error_message else "No output generated.")
        })

        # Add a record of this agent's operation to the overall workflow history
        new_state["history"].append(HistoryEntry(
            agent_name="AudioAgent",
            timestamp=datetime.now().isoformat(),
            input={
                "task": state["active_agent_task"],
                "query": state["query"],
                "current_roadmap_step": current_roadmap_step
            },
            output=agent_output,
            status=status,
            tool_calls=[tc.dict() for tc in tool_calls] if tool_calls else None,
            # Convert tool_calls to dict for HistoryEntry
            error=error_message
        ))

        return new_state