"""LangGraph assembly for the NVIDIA Startup AI Radar."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Literal

from nvidia_startup_ai_radar.agents import (
    briefing_agent,
    classifier_agent,
    economic_estimator_agent,
    evidence_validator_agent,
    extractor_agent,
    judge_agent,
    nvidia_rag_agent,
    recommendation_agent,
    scraper_agent,
    search_planner_agent,
    translation_agent,
)
from nvidia_startup_ai_radar.schemas import AgentState, utc_now_iso


LLM_CAPABLE_AGENTS = {
    "extractor",
    "startup_classifier",
    "evidence_validator",
    "recommendation",
    "llm_as_judge",
    "briefing",
}


def route_after_briefing(state: AgentState) -> Literal["translate", "__end__"]:
    return "translate" if state.get("output_language") in {"en", "both"} else "__end__"


def _extract_tokens_used(result: AgentState, agent_name: str) -> int | None:
    logs = result.get("agent_execution_log") or []
    for item in reversed(logs):
        if item.get("agent") != agent_name:
            continue
        for key in ("tokens_used", "tokens_usados", "total_tokens"):
            value = item.get(key)
            if isinstance(value, int):
                return value
        usage = item.get("usage") or item.get("usage_metadata")
        if isinstance(usage, dict):
            for key in ("total_tokens", "tokens_used", "input_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    return value
    return None


def _extract_estimated_cost_usd(result: AgentState, agent_name: str) -> float | None:
    logs = result.get("agent_execution_log") or []
    for item in reversed(logs):
        if item.get("agent") != agent_name:
            continue
        for key in ("estimated_cost_usd", "cost_usd", "custo_estimado_usd"):
            value = item.get(key)
            if isinstance(value, int | float):
                return float(value)
    return None


def _resolve_execution_mode(agent_name: str, state: AgentState, result: AgentState) -> str:
    modes = dict(state.get("agent_execution_modes", {}))
    modes.update(result.get("agent_execution_modes", {}))
    if modes.get(agent_name):
        return str(modes[agent_name])
    if agent_name in LLM_CAPABLE_AGENTS:
        return "fallback_apos_falha"
    return "deterministic"


def _instrument_agent(agent_name: str, agent: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    """Wrap a graph node with local observability without changing node behavior."""

    def wrapped(state: AgentState) -> AgentState:
        started_at = utc_now_iso()
        started = time.perf_counter()
        traces = list(state.get("agent_traces", []))
        execution_id = state.get("execution_id")
        try:
            result = agent(state) or {}
            success = True
            error_message = None
        except Exception as exc:  # pragma: no cover - defensive path for production resiliency
            result = {"errors": [*state.get("errors", []), f"{agent_name}: {type(exc).__name__}: {exc}"]}
            success = False
            error_message = f"{type(exc).__name__}: {exc}"
        ended_at = utc_now_iso()
        trace = {
            "agent": agent_name,
            "run_id": execution_id,
            "execution_id": execution_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "success": success,
            "error": error_message,
            "modo_execucao": _resolve_execution_mode(agent_name, state, result),
            "tokens_used": _extract_tokens_used(result, agent_name),
            "estimated_cost_usd": _extract_estimated_cost_usd(result, agent_name),
        }
        return {**result, "agent_traces": [*traces, trace]}

    return wrapped


def build_graph():
    """Build and compile the LangGraph workflow."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Run `pip install -e .` before executing the graph."
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("search_planner", _instrument_agent("search_planner", search_planner_agent))
    graph.add_node("scraper", _instrument_agent("scraper", scraper_agent))
    graph.add_node("extractor", _instrument_agent("extractor", extractor_agent))
    graph.add_node("startup_classifier", _instrument_agent("startup_classifier", classifier_agent))
    graph.add_node("evidence_validator", _instrument_agent("evidence_validator", evidence_validator_agent))
    graph.add_node("nvidia_rag", _instrument_agent("nvidia_rag", nvidia_rag_agent))
    graph.add_node("recommendation", _instrument_agent("recommendation", recommendation_agent))
    graph.add_node("economic_estimator", _instrument_agent("economic_estimator", economic_estimator_agent))
    graph.add_node("llm_as_judge", _instrument_agent("llm_as_judge", judge_agent))
    graph.add_node("briefing", _instrument_agent("briefing", briefing_agent))
    graph.add_node("technical_translation", _instrument_agent("technical_translation", translation_agent))

    graph.add_edge(START, "search_planner")
    graph.add_edge("search_planner", "scraper")
    graph.add_edge("scraper", "extractor")
    graph.add_edge("extractor", "startup_classifier")
    graph.add_edge("startup_classifier", "evidence_validator")
    graph.add_edge("evidence_validator", "nvidia_rag")
    graph.add_edge("nvidia_rag", "recommendation")
    graph.add_edge("recommendation", "economic_estimator")
    graph.add_edge("economic_estimator", "llm_as_judge")
    graph.add_edge("llm_as_judge", "briefing")
    graph.add_conditional_edges(
        "briefing",
        route_after_briefing,
        {"translate": "technical_translation", "__end__": END},
    )
    graph.add_edge("technical_translation", END)
    return graph.compile()
