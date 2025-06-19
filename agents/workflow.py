import os

from langchain_core.language_models import BaseChatModel
from langchain_huggingface.chat_models import ChatHuggingFace

from agents.orchestrator import create_master_orchestrator_workflow
from langchain.chat_models import init_chat_model
from langchain_community.llms import HuggingFaceEndpoint  # For HF models
from langchain_openai import ChatOpenAI  # For OpenRouter (using OpenAI compatible API)
from langchain_groq import ChatGroq  # For Groq

# Load environment variables (ensure .env file is present or keys are set globally)
from dotenv import load_dotenv

load_dotenv()


# --- Helper Functions to Instantiate LLMs from Providers ---
def _try_init_llm(provider: str, model_id: str, **kwargs) -> BaseChatModel | None:
    """
    Attempts to instantiate an LLM using init_chat_model for a given provider and model.
    Handles API key retrieval and prints success/failure messages.
    """
    api_key_env_var = ""
    if provider == "huggingface":
        api_key_env_var = "HF_TOKEN"
    elif provider == "openai" or provider == "openrouter":  # OpenRouter uses 'openai' provider string
        api_key_env_var = "OPENROUTER_API_KEY"
    elif provider == "groq":
        api_key_env_var = "GROQ_API_KEY"

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        print(f"Warning: {api_key_env_var} not found for {provider.capitalize()} LLM.")
        return None

    try:
        # Pass API key directly in kwargs based on provider
        if provider == "huggingface":
            kwargs["huggingfacehub_api_token"] = api_key
        elif provider == "openai" or provider == "openrouter":  # For OpenRouter, use openai provider string
            kwargs["api_key"] = api_key
        elif provider == "groq":
            kwargs["groq_api_key"] = api_key

        llm = init_chat_model(
            model=model_id,
            model_provider=provider if provider != "openrouter" else "openai",  # Use 'openai' for OpenRouter
            temperature=0.3,
            max_tokens=512,
            **kwargs
        )
        print(f"Successfully instantiated {provider.capitalize()} LLM: {model_id}")
        return llm
    except Exception as e:
        print(f"Failed to instantiate {provider.capitalize()} LLM {model_id}: {e}")
        return None


def _create_hf_llm(model_id: str) -> BaseChatModel | None:
    """Attempts to instantiate a ChatHuggingFace LLM that supports bind_tools."""
    hf_token = os.getenv("HF_TOKEN")  # Or HUGGINGFACEHUB_API_TOKEN
    if not hf_token:
        print("HF_TOKEN not found for HuggingFace LLM.")
        return None
    try:
        # ChatHuggingFace acts as a wrapper. It requires an existing LLM instance
        # to be passed to its 'llm' parameter.
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=model_id,
            temperature=0.3,
            max_new_tokens=512,  # Use max_new_tokens here as it's for HuggingFaceEndpoint
            huggingfacehub_api_token=hf_token
        )

        llm = ChatHuggingFace(llm=llm_endpoint)  # Pass the instantiated LLM

        print(f"Successfully instantiated HuggingFace LLM: {model_id}")
        return llm
    except Exception as e:
        print(f"Failed to instantiate HuggingFace LLM {model_id}: {e}")
        return None


def _create_openrouter_llm(model_id: str) -> BaseChatModel | None:
    """Attempts to instantiate an OpenRouter LLM using ChatOpenAI."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("OPENROUTER_API_KEY not found for OpenRouter LLM.")
        return None
    try:
        llm = ChatOpenAI(
            model=model_id,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            temperature=0.3,
            max_tokens=512
        )
        print(f"Successfully instantiated OpenRouter LLM: {model_id}")
        return llm
    except Exception as e:
        print(f"Failed to instantiate OpenRouter LLM {model_id}: {e}")
        return None


def _create_groq_llm(model_id: str) -> BaseChatModel | None:
    """Attempts to instantiate a Groq LLM."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("GROQ_API_KEY not found for Groq LLM.")
        return None
    try:
        llm = ChatGroq(
            model_name=model_id,
            groq_api_key=groq_key,
            temperature=0.3,
            max_tokens=512
        )
        print(f"Successfully instantiated Groq LLM: {model_id}")
        return llm
    except Exception as e:
        print(f"Failed to instantiate Groq LLM {model_id}: {e}")
        return None


# LLM Initializations
def create_orchestrator_llm():
    # Try HuggingFace first
    llm = _create_hf_llm("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    if llm: return llm

    # Then OpenRouter
    llm = _create_openrouter_llm("deepseek/deepseek-r1-distill-qwen-14b:free")
    if llm: return llm

    # Finally Groq
    llm = _create_groq_llm("deepseek-r1-distill-llama-70b")
    if llm: return llm

    raise ValueError("Failed to instantiate Orchestrator LLM from any provider.")


def create_visual_llm():
    # HuggingFace (some multi-modal models like Llava might be available as endpoints)
    llm = _create_hf_llm(
        "meta-llama/Llama-3.2-11B-Vision-Instruct")  # Check if this specific endpoint is available or other Llava models
    if llm: return llm

    # OpenRouter (often has access to multi-modal models)
    llm = _create_openrouter_llm(
        "meta-llama/llama-3.2-11b-vision-instruct:free")  # Example, check OpenRouter's list for actual ID
    # google/gemma-3-27b-it:free also works
    if llm: return llm

    # Fallback if no multi-modal found: a generic LLM (won't handle images directly)
    print("Warning: No multi-modal LLM found for Visual Agent. Falling back to generic LLM.")
    return create_generic_llm()  # Fallback to a generic text-only LLM


def create_audio_llm():
    # HuggingFace
    llm = _create_hf_llm("Qwen/QwQ-32B")  # A good general-purpose model
    if llm: return llm

    # OpenRouter
    llm = _create_openrouter_llm("google/gemma-2-9b-it:free")
    if llm: return llm

    # Groq
    llm = _create_groq_llm("qwen/qwen3-32b")  # Groq's fast Llama3
    if llm: return llm

    raise ValueError("Failed to instantiate Audio LLM from any provider.")


def create_researcher_llm():
    """Researcher: Some normal LLM (can be the same as audio/generic)."""
    # HuggingFace
    llm = _create_hf_llm("Qwen/QwQ-32B")  # A good general-purpose model
    if llm: return llm

    # OpenRouter
    llm = _create_openrouter_llm("google/gemma-2-9b-it:free")
    if llm: return llm

    # Groq
    llm = _create_groq_llm("qwen/qwen3-32b")  # Groq's fast Llama3
    if llm: return llm

    raise ValueError("Failed to instantiate Audio LLM from any provider.")


def create_interpreter_llm():
    """Code: Some LLM for coding tasks."""
    # Prioritize HuggingFace for code models
    llm = _create_hf_llm("Qwen/Qwen2.5-Coder-32B-Instruct")
    if llm: return llm

    # Then try OpenRouter
    llm = _create_openrouter_llm("qwen/qwen-2.5-coder-32b-instruct:free")
    if llm: return llm

    # Finally Groq
    llm = _create_groq_llm("qwen-qwq-32b")
    if llm: return llm

    raise ValueError("Failed to instantiate Code Interpreter LLM from any provider.")


def create_generic_llm():
    """Generic: Some normal free LLM."""
    # HuggingFace
    llm = _create_hf_llm("Qwen/QwQ-32B")  # A good general-purpose model
    if llm: return llm

    # OpenRouter
    llm = _create_openrouter_llm("google/gemma-2-9b-it:free")
    if llm: return llm

    # Groq
    llm = _create_groq_llm("qwen/qwen3-32b")  # Groq's fast Llama3
    if llm: return llm

    raise ValueError("Failed to instantiate Audio LLM from any provider.")


def create_worfklow():
    try:
        orchestrator_llm = create_orchestrator_llm()
        print("Orchestrator LLM initialized")

        visual_llm = create_visual_llm()
        print("Visual LLM initialized")

        audio_llm = create_audio_llm()
        print("Audio LLM initialized")

        researcher_llm = create_researcher_llm()
        print("Researcher LLM initialized")

        interpreter_llm = create_interpreter_llm()
        print("Code Interpreter LLM initialized")

        generic_llm = create_generic_llm()
        print("Generic LLM initialized.")
    except Exception as e:
        print(f"Error initializing LLMs. Ensure API keys are set: {e}")
        # You might want to raise the exception or handle it more gracefully
        raise

    # --- Create the Master Orchestrator Workflow ---
    print("Creating master orchestrator workflow...")
    orchestrator_compiled_app = create_master_orchestrator_workflow(
        orchestrator_llm=orchestrator_llm,
        visual_llm=visual_llm,
        audio_llm=audio_llm,
        researcher_llm=researcher_llm,
        interpreter_llm=interpreter_llm,
        generic_llm=generic_llm,
    ).compile()

    return orchestrator_compiled_app
