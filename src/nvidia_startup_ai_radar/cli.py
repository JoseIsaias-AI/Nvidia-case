"""Command-line entry point for the LangGraph pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nvidia_startup_ai_radar.graph import build_graph


def _load_inbound(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NVIDIA Startup AI Radar LangGraph.")
    parser.add_argument("--query", help="Outbound search query or pasted startup description.")
    parser.add_argument("--inbound-json", help="Path to an inbound StartupProfile-like JSON.")
    parser.add_argument(
        "--output-language",
        choices=["pt", "en", "both"],
        default="pt",
        help="Briefing language. pt is canonical.",
    )
    parser.add_argument("--json", action="store_true", help="Print full final state as JSON.")
    args = parser.parse_args()

    inbound_profile = _load_inbound(args.inbound_json)
    if not args.query and not inbound_profile:
        parser.error("Provide --query or --inbound-json.")

    graph = build_graph()
    final_state = graph.invoke(
        {
            "query": args.query or "",
            "inbound_profile": inbound_profile or {},
            "output_language": args.output_language,
            "errors": [],
        }
    )
    if args.json:
        print(json.dumps(final_state, ensure_ascii=False, indent=2))
    else:
        print(final_state.get("briefing_en") or final_state.get("briefing_pt", ""))


if __name__ == "__main__":
    main()
