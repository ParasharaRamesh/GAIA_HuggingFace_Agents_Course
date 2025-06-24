# all llms are instantiated here
from dotenv import load_dotenv
import os
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI  # For OpenRouter (using OpenAI compatible API)
from langchain_groq import ChatGroq  # For Groq

from langchain_core.language_models import BaseChatModel
from langchain_core.utils.utils import secret_from_env
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from pydantic import SecretStr, Field


load_dotenv()

# Custom ChatOpenRouter class
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("OPENROUTER_API_KEY", default=None)
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1", openai_api_key=openai_api_key, **kwargs)


# --- Helper Functions to Instantiate LLMs from Providers ---
def _try_init_llm(provider: str, model_id: str, **kwargs) -> BaseChatModel | None:
    """
    Attempts to instantiate an LLM using init_chat_model for a given provider and model.
    Handles API key retrieval and prints success/failure messages.
    """
    print(f"using init llm method: {model_id} @ {provider}")
    api_key_env_var = ""
    if provider == "huggingface": #This doesnt work
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


def _create_hf_llm(model_id: str, use_init_llm: bool = False) -> BaseChatModel | None:
    """Attempts to instantiate a ChatHuggingFace LLM that supports bind_tools."""
    if use_init_llm:
        # this doesnt work!
        return _try_init_llm("huggingface", model_id)

    hf_token = os.getenv("HF_TOKEN")  # Or HUGGINGFACEHUB_API_TOKEN
    if not hf_token:
        print("HF_TOKEN not found for HuggingFace LLM.")
        return None
    try:
        llm_hub = HuggingFaceHub(
            repo_id=model_id,
            model_kwargs={"temperature": 0.3, "max_new_tokens": 512},
            huggingfacehub_api_token=hf_token
        )

        llm = ChatHuggingFace(llm=llm_hub, verbose=True)

        print(f"Successfully instantiated HuggingFace LLM: {model_id}")
        return llm
    except Exception as e:
        print(f"Failed to instantiate HuggingFace LLM {model_id}: {e}")
        return None


def _create_openrouter_llm(model_id: str, use_init_llm: bool = False) -> BaseChatModel | None:
    """Attempts to instantiate an OpenRouter LLM using ChatOpenAI."""
    if use_init_llm:
        return _try_init_llm("openrouter", model_id)

    # alternative method
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("OPENROUTER_API_KEY not found for OpenRouter LLM.")
        return None
    try:
        llm = ChatOpenRouter(
            model_name=model_id,
            temperature=0.3,
            max_tokens=512
        )
        print(f"Successfully instantiated OpenRouter LLM: {model_id}")
        return llm
    except Exception as e:
        print(f"Failed to instantiate OpenRouter LLM {model_id}: {e}")
        return None


def _create_groq_llm(model_id: str, use_init_llm: bool = True) -> BaseChatModel | None:
    if use_init_llm:
        return _try_init_llm("groq", model_id)

    # alternative method
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
def create_orchestrator_llm(use_hf: bool = False, use_or: bool = True, use_groq: bool = False):
    # Try HuggingFace first
    if use_hf:
        llm = _create_hf_llm("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        if llm: return llm
        print("~"*60)

    # Then OpenRouter
    if use_or:
        llm = _create_openrouter_llm("deepseek/deepseek-chat-v3-0324:free")
        if llm: return llm
        print("~"*60)

    # Finally Groq
    if use_groq:
        llm = _create_groq_llm("deepseek-r1-distill-llama-70b")
        if llm: return llm
        print("~"*60)

    raise ValueError("Failed to instantiate Orchestrator LLM from any provider.")


def create_generic_llm(use_hf: bool = False, use_or: bool = True, use_groq: bool = False):
    """Generic: Some normal free LLM."""
    # HuggingFace
    if use_hf:
        llm = _create_hf_llm("Qwen/QwQ-32B")  # A good general-purpose model
        if llm: return llm
        print("~"*60)

    # OpenRouter
    if use_or:
        llm = _create_openrouter_llm("deepseek/deepseek-chat-v3-0324:free")
        if llm: return llm
        print("~"*60)

    # Groq
    if use_groq:
        llm = _create_groq_llm("qwen/qwen3-32b")  # Groq's fast Llama3
        if llm: return llm
        print("~"*60)

    raise ValueError("Failed to instantiate Audio LLM from any provider.")


def create_researcher_llm(use_hf: bool = False, use_or: bool = True, use_groq: bool = False):
    """Researcher: Some normal LLM (can be the same as audio/generic)."""
    # HuggingFace
    if use_hf:
        llm = _create_hf_llm("Qwen/QwQ-32B")  # A good general-purpose model
        if llm: return llm
        print("~"*60)

    # OpenRouter
    if use_or:
        llm = _create_openrouter_llm("deepseek/deepseek-chat-v3-0324:free")
        if llm: return llm
        print("~"*60)

    # Groq
    if use_groq:
        llm = _create_groq_llm("qwen/qwen3-32b")  # Groq's fast Llama3
        if llm: return llm
        print("~"*60)

    raise ValueError("Failed to instantiate Audio LLM from any provider.")


def create_audio_llm(use_hf: bool = False, use_or: bool = True, use_groq: bool = False):
    # HuggingFace
    if use_hf:
        llm = _create_hf_llm("Qwen/QwQ-32B")  # A good general-purpose model
        if llm: return llm
        print("~"*60)

    # OpenRouter
    if use_or:
        llm = _create_openrouter_llm("deepseek/deepseek-chat-v3-0324:free")
        if llm: return llm
        print("~"*60)

    # Groq
    if use_groq:
        llm = _create_groq_llm("qwen/qwen3-32b")  # Groq's fast Llama3
        if llm: return llm
        print("~"*60)

    raise ValueError("Failed to instantiate Audio LLM from any provider.")


def create_visual_llm(use_hf: bool = True, use_or: bool = False, use_groq: bool = False):
    # HuggingFace (some multi-modal models like Llava might be available as endpoints)
    if use_hf:
        llm = _create_hf_llm("meta-llama/Llama-3.2-11B-Vision-Instruct")
        if llm: return llm
        print("~"*60)

    # OpenRouter (often has access to multi-modal models)
    if use_or:
        llm = _create_openrouter_llm("mistralai/mistral-small-3.2-24b-instruct:free") #works
        if llm: return llm
        print("~"*60)

    # Fallback if no multi-modal found: a generic LLM (won't handle images directly)
    print("Warning: No multi-modal LLM found for Visual Agent. Falling back to generic LLM.")
    return create_generic_llm()  # Fallback to a generic text-only LLM


def create_interpreter_llm(use_hf: bool = False, use_or: bool = True, use_groq: bool = False):
    """Code: Some LLM for coding tasks."""
    # Prioritize HuggingFace for code models
    if use_hf:
        llm = _create_hf_llm("Qwen/Qwen2.5-Coder-32B-Instruct")
        if llm: return llm
        print("~"*60)

    # Then try OpenRouter
    if use_or:
        llm = _create_openrouter_llm("mistralai/devstral-small:free")
        if llm: return llm
        print("~"*60)

    # Finally Groq
    if use_groq:
        llm = _create_groq_llm("qwen-qwq-32b")
        if llm: return llm
        print("~"*60)

    raise ValueError("Failed to instantiate Code Interpreter LLM from any provider.")

