# Defines the Planner agent
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, List  # Add any other types you need for other parts of planner.py

# Assuming AgentState, PlanStep, HistoryEntry are in agents.state as per your uploaded file
from agents.state import AgentState, PlanStep, HistoryEntry

# Assuming create_type_string is now located in utils/state.py as per your latest info
from utils.state import create_type_string  # <-- NEW IMPORT

'''TODO.x:
1. How is the model switch to a reasoning model happening? doesnt seem like it is happening. Perhaps for first version not really needed!
2. The planner_prompt might not be aware of the planstep type, how to make it aware in the prompt?



'''


class GlobalPlanner:
    """
    The GlobalPlanner is responsible for generating the overall plan of action,
    re-planning when errors occur, and making high-level decisions for the agent's workflow.
    It orchestrates the sequence of operations by modifying the `current_plan` in the AgentState.
    """

    def __init__(self, llm, planner_prompt_filename: str = "planner_prompt.txt"):
        self.llm = llm

        # prompt path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, '..', 'prompts')
        planner_prompt_path = os.path.join(prompts_dir, planner_prompt_filename)

        # prompt injection
        self.planner_prompt_template = None
        with open(planner_prompt_path, 'r', encoding='utf-8') as f:
            self.planner_prompt_template = f.read()

        agent_state_schema = create_type_string(AgentState)
        plan_step_schema = create_type_string(PlanStep)
        history_entry_schema = create_type_string(HistoryEntry)

        self.planner_prompt_template = (self.planner_prompt_template
                                        .replace("agent_state_schema_placeholder", agent_state_schema)
                                        .replace("plan_step_schema_placeholder", plan_step_schema)
                                        .replace("history_entry_schema_placeholder", history_entry_schema)
                                        )
        print("Loaded Planner agent")



if __name__ == '__main__':
    GlobalPlanner("blah blah")
