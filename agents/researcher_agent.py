from datetime import datetime
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel  # For type hinting the LLM

from langchain.agents import create_react_agent, AgentExecutor

# Import AgentState and HistoryEntry from state.py
from .state import AgentState, HistoryEntry, HistoryEntryStatus

# Import tools from search.py (assuming tools directory is alongside this file or in PYTHONPATH)
from tools.search import web_search, wikipedia_search, arxiv_search, web_scraper

class ResearcherAgent:
    """
    Agent responsible for conducting research using various web search and scraping tools.
    It takes a research task from the Planner and returns relevant information.
    This agent handles its own internal tool calls and reports errors back to the Planner.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        # Expose relevant tools to the LLM.
        self.tools = [web_search, wikipedia_search, arxiv_search, web_scraper]

        # prompt path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, '..', 'prompts')
        researcher_react_prompt_path = os.path.join(prompts_dir, 'researcher_react_prompt.txt')

        # load prompts
        react_prompt_content = ""
        try:
            with open(researcher_react_prompt_path, 'r', encoding='utf-8') as f:
                react_prompt_content = f.read()

            print(f"[ResearcherAgent.init]: Loaded react prompt from {researcher_react_prompt_path}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading ResearcherAgent prompts: {e}")

        # Create the ReAct agent's prompt template. The `human` message uses specific placeholders
        # that create_react_agent expects for its internal loop.
        self.react_prompt_template = ChatPromptTemplate.from_messages([
            ("system", react_prompt_content),
            ("human", "{input}\n{agent_scratchpad}")  # These are standard ReAct agent placeholders
        ])

        # Create the ReAct agent (this is a Runnable)
        self.agent_runnable = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.react_prompt_template
        )

        # Create the AgentExecutor to run the ReAct agent (handles the Thought/Action/Observation loop)
        self.agent_executor = AgentExecutor(
            agent=self.agent_runnable,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            return_intermediate_steps=True
        )
        print(f"[ResearcherAgent.init]: LLM successfully bound with tools and initialized!")

    def __call__(self, state: AgentState) -> AgentState:
        """
        Executes the researcher's task based on the current AgentState and updates the state
        with research findings or an error message.
        """
        print(f"[ResearcherAgent.call]: Starting task for roadmap step {state['current_roadmap_step_index']}")

        # Determine the current high-level roadmap step for context in prompt
        current_roadmap_step = state['high_level_roadmap'][state['current_roadmap_step_index']] \
            if state['high_level_roadmap'] and state['current_roadmap_step_index'] < len(state['high_level_roadmap']) \
            else "N/A"

        agent_output = None
        error_message = None
        status = HistoryEntryStatus.SUCCESS
        extracted_tool_calls_for_history = []
        react_agent_input_text = ""

        try:
            # Prepare the input string for the internal ReAct agent.
            # This is what the ReAct agent will reason over.
            react_agent_input_text = (
                f"Overall User Query: {state['query']}\n"
                f"Current High-Level Roadmap Step: {current_roadmap_step}\n"
                f"Specific Task for Researcher: {state['active_agent_task']}\n"
                f"Conversation History with Planner for this task: {state['conversation_history_with_agent']}\n"
                f"Previous output from this agent (if re-attempting): {state['active_agent_output']}\n"
                f"Planner's latest feedback: {state['planner_feedback']}\n"
                f"Previous error (if any): {state['active_agent_error_message']}\n\n"
                f"Proceed with the 'Specific Task for Researcher' using your tools. Provide a concise final answer."
            )

            # Invoke the AgentExecutor. It will run the ReAct loop and return the final answer.
            executor_result = self.agent_executor.invoke({"input": react_agent_input_text})

            # get the agent output
            agent_output = executor_result.get("output")
            if not agent_output:  # Fallback if agent_executor output is missing
                agent_output = "Internal ReAct agent completed but returned no specific output."
                status = HistoryEntryStatus.FAILED

            # Extract tool calls from intermediate_steps if available
            intermediate_steps = executor_result.get("intermediate_steps", [])
            for action, observation in intermediate_steps:
                if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                    extracted_tool_calls_for_history.append({
                        "tool_name": action.tool,
                        "tool_input": action.tool_input,
                        "tool_output": observation  # The observation is the tool's raw output
                    })
                # Handle cases where action might not directly have tool/tool_input attributes
                # (e.g., if it's an AgentFinish or other non-tool action)
                elif hasattr(action, 'log'):  # For AgentFinish, log often contains the final answer thought
                    extracted_tool_calls_for_history.append({
                        "log_entry": action.log,
                        "observation": observation  # The observation in this case might be the final answer itself
                    })
                else:
                    extracted_tool_calls_for_history.append({
                        "action_type": str(type(action)),
                        "action_details": str(action),
                        "observation": str(observation)
                    })

        except Exception as e:
            error_message = str(e)
            status = HistoryEntryStatus.FAILED
            print(f"[ResearcherAgent.call]: AgentExecutor error: {e}")

        # Create a mutable copy of the state to update
        new_state = state.copy()
        new_state["active_agent_name"] = "ResearcherAgent"
        new_state["active_agent_output"] = agent_output
        new_state["active_agent_error_message"] = error_message

        new_state["conversation_history_with_agent"].append({
            "role": "agent",
            "message": agent_output if agent_output is not None else (
                error_message if error_message else "No output generated.")
        })

        new_state["history"].append(HistoryEntry(
            agent_name="ResearcherAgent",
            timestamp= datetime.now().isoformat(),
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
