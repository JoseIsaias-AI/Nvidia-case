"""Shared execution helpers for CLI and dashboard entry points."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from nvidia_startup_ai_radar.graph import build_graph
from nvidia_startup_ai_radar.schemas import AgentState


def run_radar(
    *,
    query: str = "",
    inbound_profile: dict[str, Any] | None = None,
    output_language: Literal["pt", "en", "both"] = "pt",
    rag_db_path: str | None = None,
    discover_candidates: bool = False,
    discovery_limit: int = 15,
    discovery_results_per_query: int = 4,
    discovery_fetch_pages: bool = False,
    discovery_delay_seconds: float = 0.0,
    discovery_search_workers: int = 6,
) -> AgentState:
    """Run the LangGraph workflow from either outbound text or inbound JSON."""

    inbound_profile = inbound_profile or {}
    if not query and not inbound_profile:
        raise ValueError("Provide query or inbound_profile.")

    graph = build_graph()
    state: AgentState = {
        "execution_id": str(uuid4()),
        "query": query,
        "inbound_profile": inbound_profile,
        "output_language": output_language,
        "discover_candidates": discover_candidates,
        "discovery_limit": discovery_limit,
        "discovery_results_per_query": discovery_results_per_query,
        "discovery_fetch_pages": discovery_fetch_pages,
        "discovery_delay_seconds": discovery_delay_seconds,
        "discovery_search_workers": discovery_search_workers,
        "agent_execution_modes": {},
        "agent_execution_log": [],
        "agent_traces": [],
        "errors": [],
    }
    if rag_db_path:
        state["rag_db_path"] = rag_db_path
    return graph.invoke(state)
