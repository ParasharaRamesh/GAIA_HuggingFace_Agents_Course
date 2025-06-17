import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Import AgentState and HistoryEntry from state.py
from state import AgentState, HistoryEntry, HistoryEntryStatus


class VisualAgent:
    """
    Agent responsible for analyzing images in conjunction with textual queries.
    It takes an image file path (if provided in state) and a textual task/query,
    then uses a multimodal LLM to provide an answer.

    Crucially, if asked about video *visuals*, it will hallucinate an answer.
    For video *speech*, it relies on pre-transcribed text provided by the Planner.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initializes the VisualAgent with a multimodal LLM.

        Args:
            llm (BaseChatModel): The multimodal language model capable of processing
                                 both text and image inputs (e.g., Google Gemini Pro Vision).
        """
        self.llm = llm

        # Construct the path to the Visual Agent's prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, '..', 'prompts')
        self.visual_agent_prompt_path = os.path.join(prompts_dir, 'visual_agent_prompt.txt')

    def __call__(self, state: AgentState) ->AgentState:
        """
        Executes the VisualAgent's task based on the current AgentState.

        Args:
            state (AgentState): The current state of the multi-agent system.

        Returns:
            Dict[str, Any]: The updated AgentState after the VisualAgent's execution.
        """
        task_for_visual_agent = state["active_agent_task"]
        query = state["query"]
        history_entries = state["history"]
        image_file_path = state.get("file_path")  # Get file_path, which might be an image

        # Create a mutable copy of the state to update
        new_state = state.copy()
        new_state["active_agent_name"] = "VisualAgent"
        new_state["active_agent_output"] = None
        new_state["active_agent_error_message"] = None

        # Load the prompt template
        with open(self.visual_agent_prompt_path, "r", encoding="utf-8") as f:
            prompt_template_str = f.read()

        # Format the prompt with current state information
        formatted_prompt = prompt_template_str.format(
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query=query,
            active_agent_task=task_for_visual_agent,
            image_path=image_file_path if image_file_path else "No image provided.",
            history=self._format_history(history_entries)
        )

        # Prepare multimodal input for the LLM
        messages = [SystemMessage(content=formatted_prompt)]

        # If an image file path is provided, add it to the HumanMessage content
        human_message_content = []
        human_message_content.append({"type": "text", "text": f"Task: {task_for_visual_agent}"})
        if image_file_path:
            # Ensure the file path is accessible by the LLM (e.g., as a local file URL)
            # This assumes the LLM provider can access local file paths directly or via an integration.
            # For cloud-based LLMs, this might require uploading the image to a temporary URL.
            # For local execution with models like Llama.cpp, direct paths often work.
            # Assuming LangChain handles conversion for supported LLMs like Gemini.
            human_message_content.append(
                {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(image_file_path)}"}})

        messages.append(HumanMessage(content=human_message_content))

        agent_output = None
        error_message = None
        status = HistoryEntryStatus.IN_PROGRESS

        try:
            # Invoke the multimodal LLM
            llm_response = self.llm.invoke(messages)
            agent_output = llm_response.content
            status = HistoryEntryStatus.SUCCESS

        except Exception as e:
            error_message = str(e)
            status = HistoryEntryStatus.FAILED
            print(f"[VisualAgent.call]: LLM invocation error: {e}")

        new_state["active_agent_output"] = agent_output
        new_state["active_agent_error_message"] = error_message

        new_state["history"].append(HistoryEntry(
            agent_name="VisualAgent",
            timestamp=datetime.now().isoformat(),
            input={
                "task_for_visual_agent": task_for_visual_agent,
                "query": query,
                "image_file_path": image_file_path
            },
            output=agent_output,
            status=status,
            tool_calls=[],  # VisualAgent does not make tool calls directly
            error=error_message
        ))

        # If the agent produces a "Final Answer", populate the final_answer field.
        if agent_output and isinstance(agent_output, str) and agent_output.strip().startswith("Final Answer:"):
            new_state["final_answer"] = agent_output.replace("Final Answer:", "").strip()

        return new_state

    def _format_history(self, history_entries: List[HistoryEntry]) -> str:
        """
        Helper method to format a list of history entries into a readable string
        for the LLM, providing context of previous interactions.
        """
        formatted_history = []
        for entry in history_entries:
            formatted_history.append(f"--- History Entry for {entry['agent_name']} ---")
            formatted_history.append(f"Timestamp: {entry['timestamp']}")
            formatted_history.append(f"Status: {entry['status']}")
            formatted_history.append(f"Input Task: {entry['input'].get('task_for_visual_agent', 'N/A')}")
            formatted_history.append(f"Output: {entry['output']}")
            if entry['error']:
                formatted_history.append(f"Error: {entry['error']}")
            formatted_history.append("--------------------")
        if not formatted_history:
            return "No previous history."
        return "\n".join(formatted_history)