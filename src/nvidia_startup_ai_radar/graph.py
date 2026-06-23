"""LangGraph assembly for the NVIDIA Startup AI Radar."""

from __future__ import annotations

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
from nvidia_startup_ai_radar.schemas import AgentState


def route_after_briefing(state: AgentState) -> Literal["translate", "__end__"]:
    return "translate" if state.get("output_language") in {"en", "both"} else "__end__"


def build_graph():
    """Build and compile the LangGraph workflow."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Run `pip install -e .` before executing the graph."
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("search_planner", search_planner_agent)
    graph.add_node("scraper", scraper_agent)
    graph.add_node("extractor", extractor_agent)
    graph.add_node("startup_classifier", classifier_agent)
    graph.add_node("evidence_validator", evidence_validator_agent)
    graph.add_node("nvidia_rag", nvidia_rag_agent)
    graph.add_node("recommendation", recommendation_agent)
    graph.add_node("economic_estimator", economic_estimator_agent)
    graph.add_node("llm_as_judge", judge_agent)
    graph.add_node("briefing", briefing_agent)
    graph.add_node("technical_translation", translation_agent)

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
