import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


def create_visual_agent(llm: BaseChatModel):
    """
    Creates and returns a LangChain Runnable for visual analysis.
    This agent is a ReAct agent that can use tools, specifically
    'read_image_and_encode', to process local image files.
    The underlying LLM must be multimodal (e.g., V-JEPA2 with GPU, GPT-4o, Gemini 1.5 Pro)
    to interpret the Base64 image data after it's read by the tool.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    visual_prompt_path = os.path.join(prompts_dir, 'visual_prompt.txt')

    with open(visual_prompt_path, "r", encoding="utf-8") as f:
        visual_prompt_content = f.read()

    # Construct the ChatPromptTemplate using from_messages
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=visual_prompt_content),
        MessagesPlaceholder(variable_name="messages")
    ])

    # Create the agent
    visual_agent_runnable = prompt | llm
    return visual_agent_runnable
