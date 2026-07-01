"""Runtime configuration for the NVIDIA Startup AI Radar."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os


logger = logging.getLogger(__name__)
_DOTENV_LOADED = False


def load_environment() -> None:
    """Load local .env once, without making python-dotenv mandatory at runtime."""

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception as exc:
        logger.debug("python-dotenv indisponivel; usando apenas variaveis de ambiente: %s", exc)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """Environment-backed settings.

    The default mode is intentionally local/offline so the graph can be
    demonstrated without paid services. Set API keys to enable LLM calls.
    """

    llm_provider: str = "none"
    nvidia_api_key: str | None = None
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    output_language: str = "pt"
    enable_web_fetch: bool = False


def get_settings() -> Settings:
    load_environment()
    return Settings(
        llm_provider=(os.getenv("LLM_PROVIDER", "none").strip().lower() or "none"),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY") or None,
        nvidia_model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        output_language=os.getenv("RADAR_OUTPUT_LANGUAGE", "pt"),
        enable_web_fetch=_truthy(os.getenv("RADAR_ENABLE_WEB_FETCH")),
    )
