"""Optional LLM adapter.

The graph is designed to run without API keys. When keys are present, nodes can
call this adapter to replace deterministic fallbacks with structured LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel

from nvidia_startup_ai_radar.config import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuredResult:
    value: BaseModel | None
    errors: list[str]


def build_chat_model(settings: Settings):
    """Build the preferred chat model, or return None in offline mode."""

    provider = settings.llm_provider.strip().lower()
    if provider in {"none", "offline", "disabled", ""}:
        return None

    if provider in {"nvidia_nim", "nvidia"} and settings.nvidia_api_key:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            return ChatNVIDIA(model=settings.nvidia_model, api_key=settings.nvidia_api_key)
        except Exception as exc:
            logger.warning("Falha ao inicializar ChatNVIDIA; usando fallback local: %s", exc)
            return None

    if provider == "groq" and settings.groq_api_key:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
                metadata={"radar_provider": "groq"},
            ).bind(response_format={"type": "json_object"})
        except Exception as exc:
            logger.warning("Falha ao inicializar Groq via ChatOpenAI; usando fallback local: %s", exc)
            return None

    if provider == "anthropic" and settings.anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key)
        except Exception as exc:
            logger.warning("Falha ao inicializar ChatAnthropic; usando fallback local: %s", exc)
            return None

    if provider == "openai" and settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
        except Exception as exc:
            logger.warning("Falha ao inicializar ChatOpenAI; usando fallback local: %s", exc)
            return None

    return None


def _extract_json_payload(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned
    start_candidates = [index for index in [cleaned.find("{"), cleaned.find("[")] if index >= 0]
    if not start_candidates:
        raise ValueError("Resposta nao contem JSON.")
    start = min(start_candidates)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end < start:
        raise ValueError("JSON incompleto na resposta.")
    return cleaned[start : end + 1]


def _coerce_to_schema(schema: type[SchemaT], result) -> SchemaT:
    if isinstance(result, schema):
        return result
    if isinstance(result, str):
        return schema.model_validate(json.loads(_extract_json_payload(result)))
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return schema.model_validate(json.loads(_extract_json_payload(content)))
    return schema.model_validate(result)


def _provider_name(llm) -> str:
    metadata = getattr(llm, "metadata", None) or {}
    if isinstance(metadata, dict) and metadata.get("radar_provider"):
        return str(metadata["radar_provider"])
    bound = getattr(llm, "bound", None)
    bound_metadata = getattr(bound, "metadata", None) or {}
    if isinstance(bound_metadata, dict) and bound_metadata.get("radar_provider"):
        return str(bound_metadata["radar_provider"])
    return ""


def _invoke_structured_once(llm, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT:
    if hasattr(llm, "with_structured_output") and _provider_name(llm) != "groq":
        structured = llm.with_structured_output(schema)
        result = structured.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        return _coerce_to_schema(schema, result)
    result = llm.invoke(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )
    return _coerce_to_schema(schema, result)


def invoke_structured_with_retry(
    llm,
    schema: type[SchemaT],
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
) -> tuple[SchemaT | None, list[str]]:
    """Call an LLM for strict Pydantic output, retrying validation failures."""

    if llm is None:
        return None, ["llm_unavailable"]

    errors: list[str] = []
    schema_payload = schema.model_json_schema()
    schema_instruction = (
        "\n\nRetorne apenas JSON valido, sem Markdown, obedecendo exatamente este JSON Schema:\n"
        f"{json.dumps(schema_payload, ensure_ascii=False)}"
    )
    current_user_prompt = f"{user_prompt}{schema_instruction}"
    for attempt in range(max_retries + 1):
        try:
            return _invoke_structured_once(llm, schema, system_prompt, current_user_prompt), errors
        except Exception as exc:
            error = f"tentativa {attempt + 1}: {type(exc).__name__}: {exc}"
            errors.append(error)
            if "rate_limit" in error.lower() or "429" in error:
                break
            current_user_prompt = (
                f"{user_prompt}\n\n"
                "A tentativa anterior retornou uma saida invalida para o schema Pydantic.\n"
                f"Erro de validacao: {error}\n\n"
                f"{schema_instruction.strip()}"
            )
    return None, errors


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

    result, _errors = invoke_structured_with_retry(llm, schema, system_prompt, user_prompt)
    return result


def invoke_text(llm, system_prompt: str, user_prompt: str) -> str | None:
    if llm is None:
        return None
    try:
        result = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        return getattr(result, "content", str(result))
    except Exception as exc:
        logger.warning("Falha em chamada de texto ao LLM; usando fallback local: %s", exc)
        return None
