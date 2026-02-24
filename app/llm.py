import os
from typing import List, Dict, Any
from litellm import completion


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gemini/gemini-2.5-flash",
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """
    Minimal LLM wrapper via LiteLLM.
    Uses Gemini API key from env: GEMINI_API_KEY
    NOTE: model must be prefixed with 'gemini/' to use Gemini API key auth.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it in the same terminal that runs uvicorn.")

    resp: Any = completion(
        model=model,              # e.g. "gemini/gemini-2.0-flash"
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    return resp["choices"][0]["message"]["content"]
