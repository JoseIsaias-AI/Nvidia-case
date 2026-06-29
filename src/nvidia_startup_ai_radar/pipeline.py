"""Shared execution helpers for CLI and dashboard entry points."""

from __future__ import annotations

from typing import Any, Literal

from nvidia_startup_ai_radar.graph import build_graph
from nvidia_startup_ai_radar.schemas import AgentState


def run_radar(
    *,
    query: str = "",
    inbound_profile: dict[str, Any] | None = None,
    output_language: Literal["pt", "en", "both"] = "pt",
) -> AgentState:
    """Run the LangGraph workflow from either outbound text or inbound JSON."""

    inbound_profile = inbound_profile or {}
    if not query and not inbound_profile:
        raise ValueError("Provide query or inbound_profile.")

    graph = build_graph()
    return graph.invoke(
        {
            "query": query,
            "inbound_profile": inbound_profile,
            "output_language": output_language,
            "errors": [],
        }
    )
