"""NVIDIA Startup AI Radar LangGraph package."""

from nvidia_startup_ai_radar.config import load_environment

load_environment()

from nvidia_startup_ai_radar.graph import build_graph

__all__ = ["build_graph"]
