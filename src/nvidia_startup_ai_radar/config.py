"""Runtime configuration for the NVIDIA Startup AI Radar."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """Environment-backed settings.

    The default mode is intentionally local/offline so the graph can be
    demonstrated without paid services. Set API keys to enable LLM calls.
    """

    nvidia_api_key: str | None = os.getenv("NVIDIA_API_KEY") or None
    nvidia_model: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    output_language: str = os.getenv("RADAR_OUTPUT_LANGUAGE", "pt")
    enable_web_fetch: bool = _truthy(os.getenv("RADAR_ENABLE_WEB_FETCH"))


def get_settings() -> Settings:
    return Settings()
