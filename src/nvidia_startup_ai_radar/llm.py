"""Optional LLM adapter.

The graph is designed to run without API keys. When keys are present, nodes can
call this adapter to replace deterministic fallbacks with structured LLM output.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from nvidia_startup_ai_radar.config import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def build_chat_model(settings: Settings):
    """Build the preferred chat model, or return None in offline mode."""

    if settings.nvidia_api_key:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            return ChatNVIDIA(model=settings.nvidia_model, api_key=settings.nvidia_api_key)
        except Exception:
            return None

    if settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
        except Exception:
            return None

    return None


def invoke_structured(
    llm,
    schema: type[SchemaT],
    system_prompt: str,
    user_prompt: str,
) -> SchemaT | None:
    """Call a structured-output model if available.

    Any provider/runtime error returns None so the caller can use the local
    deterministic path.
    """

    if llm is None:
        return None
    try:
        structured = llm.with_structured_output(schema)
        result = structured.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        return result if isinstance(result, schema) else schema.model_validate(result)
    except Exception:
        return None


def invoke_text(llm, system_prompt: str, user_prompt: str) -> str | None:
    if llm is None:
        return None
    try:
        result = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        return getattr(result, "content", str(result))
    except Exception:
        return None
