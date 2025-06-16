# agents/audio_agent.py

from datetime import datetime
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.agents import AgentFinish, AgentAction

# Import AgentState and HistoryEntry from state.py
from state import AgentState, HistoryEntry, HistoryEntryStatus
from utils.state import create_type_string # Assuming this exists or remove if not used

# Import tools from tools/audio.py
from tools.audio import transcribe_audio, get_youtube_transcript

class AudioAgent:
    """
    Agent responsible for processing audio, specifically transcribing audio files or YouTube video transcripts.
    It takes an audio processing task from the Planner and returns the transcript.
    This agent handles its own internal tool calls and reports errors back to the Planner.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        # Expose relevant tools to the LLM.
        self.tools = [transcribe_audio, get_youtube_transcript]

        # ReAct prompt path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, '..', 'prompts') # Assumes 'agents' folder
        audio_react_prompt_path = os.path.join(prompts_dir, 'audio_react_prompt.txt') # New prompt file

        react_prompt_content = ""
        try:
            with open(audio_react_prompt_path, 'r', encoding='utf-8') as f:
                react_prompt_content = f.read()
            print(f"--- AudioAgent: Loaded ReAct prompt from {audio_react_prompt_path} ---")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find ReAct prompt file for AudioAgent. "
                f"Please ensure it is in the '{prompts_dir}' directory.\n"
                f"Expected prompt at: {audio_react_prompt_path}\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading AudioAgent ReAct prompt: {e}")

        # Create the ReAct agent's prompt template.
        self.react_prompt_template = ChatPromptTemplate.from_messages([
            ("system", react_prompt_content),
            ("human", "{input}\n{agent_scratchpad}")
        ])

        # Create the ReAct agent (this is a Runnable)
        self.agent_runnable = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.react_prompt_template
        )

        # Create the AgentExecutor to run the ReAct agent
        self.agent_executor = AgentExecutor(
            agent=self.agent_runnable,
            tools=self.tools,
            verbose=False, # Set to True for detailed internal agent logs during development
            handle_parsing_errors=True,
            return_intermediate_steps=True, # Essential for extracting tool calls
            # max_iterations=5 # Optional: Limit iterations if needed
        )

    def __call__(self, state: AgentState) -> AgentState:
        print(f"\n--- AudioAgent: Starting task for roadmap step {state['current_roadmap_step_index']} ---")

        current_roadmap_step = state['high_level_roadmap'][state['current_roadmap_step_index']] \
                               if state['high_level_roadmap'] and state['current_roadmap_step_index'] < len(state['high_level_roadmap']) \
                               else "N/A"

        agent_output = None
        error_message = None
        status = HistoryEntryStatus.SUCCESS
        extracted_tool_calls_for_history = [] # To capture internal tool calls

        react_agent_input_text = ""
        try:
            # Prepare the input string for the internal ReAct agent.
            react_agent_input_text = (
                f"Overall User Query: {state['query']}\n"
                f"Current High-Level Roadmap Step: {current_roadmap_step}\n"
                f"Specific Task for Audio Agent: {state['active_agent_task']}\n"
                f"Conversation History with Planner for this task: {state['conversation_history_with_agent']}\n"
                f"Previous output from this agent (if re-attempting): {state['active_agent_output']}\n"
                f"Planner's latest feedback: {state['planner_feedback']}\n"
                f"Previous error (if any): {state['active_agent_error_message']}\n\n"
                f"Proceed with the 'Specific Task for Audio Agent' using your tools. Provide a concise final answer."
            )

            # Invoke the AgentExecutor.
            executor_result = self.agent_executor.invoke({"input": react_agent_input_text})

            agent_output = executor_result.get("output")

            # Extract tool calls from intermediate_steps
            intermediate_steps = executor_result.get("intermediate_steps", [])
            for action, observation in intermediate_steps:
                if isinstance(action, AgentAction):
                    extracted_tool_calls_for_history.append({
                        "tool_name": action.tool,
                        "tool_input": action.tool_input,
                        "tool_output": observation
                    })
                elif isinstance(action, AgentFinish):
                    # AgentFinish means the agent has decided on its final answer
                    # The 'log' attribute of AgentFinish often contains the Thought leading to it.
                    extracted_tool_calls_for_history.append({
                        "agent_finish_log": action.log,
                        "final_observation": observation
                    })
                else: # Fallback for unexpected action types
                    extracted_tool_calls_for_history.append({
                        "action_type": str(type(action)),
                        "action_details": str(action),
                        "observation": str(observation)
                    })

            if not agent_output:
                agent_output = "Internal ReAct agent completed but returned no specific output."
                status = HistoryEntryStatus.FAILED

        except Exception as e:
            error_message = str(e)
            status = HistoryEntryStatus.FAILED
            print(f"--- AudioAgent: AgentExecutor error: {e} ---")

        # Update state
        new_state = state.copy()
        new_state["active_agent_name"] = "AudioAgent"
        new_state["active_agent_output"] = agent_output
        new_state["active_agent_error_message"] = error_message

        new_state["conversation_history_with_agent"].append({
            "role": "agent",
            "message": agent_output if agent_output is not None else (error_message if error_message else "No output generated.")
        })

        new_state["history"].append(HistoryEntry(
            agent_name="AudioAgent",
            timestamp=datetime.now().isoformat(),
            input={
                "task_for_react_agent": state["active_agent_task"],
                "full_input_to_react_agent": react_agent_input_text,
                "query": state["query"]
            },
            output=agent_output,
            status=status,
            tool_calls=extracted_tool_calls_for_history,
            error=error_message
        ))

        return new_state