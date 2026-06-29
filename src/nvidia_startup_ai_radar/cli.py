"""Command-line entry point for the LangGraph pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nvidia_startup_ai_radar.exporting import DEFAULT_EXPORT_DIR, export_run
from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.storage import DEFAULT_DB_PATH, get_run, list_recent_runs, save_run


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
    parser.add_argument(
        "--save-profile",
        action="store_true",
        help="Persist the final StartupProfile and briefing in a local SQLite store.",
    )
    parser.add_argument(
        "--profile-db",
        default=str(DEFAULT_DB_PATH),
        help="SQLite path used by --save-profile or --list-runs.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List recent persisted profile runs and exit.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max rows for --list-runs.")
    parser.add_argument(
        "--classification",
        choices=["AI-native", "AI-enabled", "non-AI", "indeterminado"],
        help="Filter --list-runs by classification.",
    )
    parser.add_argument("--sector", help="Filter --list-runs by sector.")
    parser.add_argument(
        "--human-review-only",
        action="store_true",
        help="Only show persisted runs that require human review.",
    )
    parser.add_argument("--export-run", type=int, help="Export a persisted run by run_id and exit.")
    parser.add_argument(
        "--export-format",
        choices=["markdown", "pdf"],
        default="pdf",
        help="Format used with --export-run.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory used with --export-run.",
    )
    args = parser.parse_args()

    if args.export_run:
        run = get_run(args.export_run, args.profile_db)
        if run is None:
            parser.error(f"Run {args.export_run} not found in {args.profile_db}.")
        path = export_run(run, args.export_dir, args.export_format)
        print(str(path))
        return

    if args.list_runs:
        print(
            json.dumps(
                list_recent_runs(
                    args.profile_db,
                    limit=args.limit,
                    classificacao=args.classification,
                    setor=args.sector,
                    human_review_required=True if args.human_review_only else None,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    inbound_profile = _load_inbound(args.inbound_json)
    if not args.query and not inbound_profile:
        parser.error("Provide --query or --inbound-json.")

    final_state = run_radar(
        query=args.query or "",
        inbound_profile=inbound_profile,
        output_language=args.output_language,
    )
    saved_run_id = save_run(final_state, args.profile_db) if args.save_profile else None
    if args.json:
        if saved_run_id is not None:
            final_state["saved_run_id"] = saved_run_id
            final_state["profile_db"] = args.profile_db
        print(json.dumps(final_state, ensure_ascii=False, indent=2))
    else:
        print(final_state.get("briefing_en") or final_state.get("briefing_pt", ""))
        if saved_run_id is not None:
            print(f"\nPerfil salvo no SQLite em {args.profile_db} (run_id={saved_run_id}).")


if __name__ == "__main__":
    main()
