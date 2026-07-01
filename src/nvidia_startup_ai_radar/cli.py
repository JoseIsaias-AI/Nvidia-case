"""Command-line entry point for the LangGraph pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nvidia_startup_ai_radar.batch import run_batch_query
from nvidia_startup_ai_radar.discovery import (
    DEFAULT_DISCOVERY_DB_PATH,
    DEFAULT_DISCOVERY_OUTPUT,
    DISCOVERY_CAMPAIGNS,
    candidates_as_dicts,
    discover_startups,
    list_discovery_candidates,
    save_candidates,
    save_candidates_sqlite,
)
from nvidia_startup_ai_radar.exporting import DEFAULT_EXPORT_DIR, export_run
from nvidia_startup_ai_radar.golden_set_eval import (
    DEFAULT_GOLDEN_SET_FIXTURE,
    DEFAULT_GOLDEN_SET_REPORT,
    evaluate_pipeline_golden_set,
)
from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.rag import (
    DEFAULT_RAG_DB_PATH,
    evaluate_rag_golden_set,
    rag_index_stats,
    rag_search,
    rebuild_rag_index,
)
from nvidia_startup_ai_radar.rag_ingestion import DEFAULT_RAW_SOURCE_DIR, ingest_rag_sources
from nvidia_startup_ai_radar.storage import DEFAULT_DB_PATH, get_run, list_recent_runs, save_run


def _load_inbound(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

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
    parser.add_argument(
        "--rag-db",
        default=str(DEFAULT_RAG_DB_PATH),
        help="SQLite path used by the local RAG index.",
    )
    parser.add_argument(
        "--rag-source-dir",
        default=str(DEFAULT_RAW_SOURCE_DIR),
        help="Directory used to store raw RAG source snapshots.",
    )
    parser.add_argument(
        "--rag-ingest",
        action="store_true",
        help="Ingest curated RAG sources, rebuild the local RAG index and exit.",
    )
    parser.add_argument(
        "--rag-fetch-official",
        action="store_true",
        help="Compatibility alias for --rag-ingest.",
    )
    parser.add_argument("--rag-rebuild", action="store_true", help="Rebuild the local RAG index and exit.")
    parser.add_argument("--rag-stats", action="store_true", help="Show local RAG index stats and exit.")
    parser.add_argument("--rag-search", help="Search the local RAG index and exit.")
    parser.add_argument("--rag-eval", action="store_true", help="Evaluate RAG retrieval against the golden set.")
    parser.add_argument("--rag-limit", type=int, default=7, help="Max rows for --rag-search.")
    parser.add_argument(
        "--eval-golden-set",
        action="store_true",
        help="Run the full pipeline against the planning golden set and write a Markdown report.",
    )
    parser.add_argument(
        "--eval-golden-fixture",
        default=str(DEFAULT_GOLDEN_SET_FIXTURE),
        help="Fixture path used by --eval-golden-set.",
    )
    parser.add_argument(
        "--eval-golden-report",
        default=str(DEFAULT_GOLDEN_SET_REPORT),
        help="Markdown report path written by --eval-golden-set.",
    )
    parser.add_argument(
        "--discover-startups",
        action="store_true",
        help="Search public web sources for high-fit AI startup candidates and exit.",
    )
    parser.add_argument(
        "--discover-query",
        action="append",
        help="Advanced: custom discovery query. Repeat to use multiple queries.",
    )
    parser.add_argument(
        "--discover-campaign",
        choices=sorted(DISCOVERY_CAMPAIGNS),
        default="full",
        help="Internal discovery campaign. Defaults to full product radar.",
    )
    parser.add_argument("--discover-limit", type=int, default=20, help="Max ranked candidates for discovery.")
    parser.add_argument(
        "--discover-results-per-query",
        type=int,
        default=6,
        help="Search results fetched for each discovery query.",
    )
    parser.add_argument(
        "--discover-no-page-fetch",
        action="store_true",
        help="Only use search-result titles/snippets, without fetching candidate pages.",
    )
    parser.add_argument(
        "--discover-output",
        default=str(DEFAULT_DISCOVERY_OUTPUT),
        help="JSONL output path for --discover-startups.",
    )
    parser.add_argument(
        "--discover-db",
        default=str(DEFAULT_DISCOVERY_DB_PATH),
        help="SQLite path used to persist discovered candidates.",
    )
    parser.add_argument(
        "--discover-list",
        action="store_true",
        help="List persisted discovery candidates and exit.",
    )
    parser.add_argument(
        "--discover-quality",
        choices=["alta", "media", "baixa", "triagem"],
        help="Filter --discover-list by quality tier.",
    )
    parser.add_argument(
        "--batch-query",
        help="Discover outbound candidates for a theme and run the full pipeline for each one.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=15,
        help="Max candidates to process with --batch-query.",
    )
    parser.add_argument(
        "--batch-concurrency",
        type=int,
        default=2,
        help="Max full-pipeline candidate runs in parallel for --batch-query.",
    )
    parser.add_argument(
        "--batch-rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between starting candidate runs in --batch-query.",
    )
    parser.add_argument(
        "--batch-results-per-query",
        type=int,
        default=4,
        help="Search depth per source query during --batch-query discovery.",
    )
    parser.add_argument(
        "--batch-no-page-fetch",
        action="store_true",
        help="Only use search-result snippets during batch discovery.",
    )
    args = parser.parse_args()

    if args.discover_list:
        print(
            json.dumps(
                list_discovery_candidates(args.discover_db, limit=args.discover_limit, quality_tier=args.discover_quality),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.discover_startups:
        candidates = discover_startups(
            queries=args.discover_query,
            campaign=args.discover_campaign,
            limit=args.discover_limit,
            results_per_query=args.discover_results_per_query,
            fetch_pages=not args.discover_no_page_fetch,
        )
        output_path = save_candidates(candidates, args.discover_output)
        saved_count = save_candidates_sqlite(candidates, args.discover_db, replace=True)
        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "db_path": args.discover_db,
                    "campaign": args.discover_campaign,
                    "saved_count": saved_count,
                    "count": len(candidates),
                    "candidates": candidates_as_dicts(candidates),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.batch_query:
        print(
            json.dumps(
                run_batch_query(
                    query=args.batch_query,
                    max_results=args.max_results,
                    concurrency=args.batch_concurrency,
                    rate_limit_seconds=args.batch_rate_limit,
                    results_per_query=args.batch_results_per_query,
                    fetch_pages=not args.batch_no_page_fetch,
                    profile_db_path=args.profile_db,
                    discovery_db_path=args.discover_db,
                    rag_db_path=args.rag_db,
                    output_language=args.output_language,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.rag_ingest or args.rag_fetch_official:
        manifest = ingest_rag_sources(args.rag_source_dir)
        index = rebuild_rag_index(args.rag_db, source_dir=args.rag_source_dir)
        print(json.dumps({"ingest": manifest, "index": index}, ensure_ascii=False, indent=2))
        return

    if args.rag_rebuild:
        print(json.dumps(rebuild_rag_index(args.rag_db, source_dir=args.rag_source_dir), ensure_ascii=False, indent=2))
        return

    if args.rag_stats:
        print(json.dumps(rag_index_stats(args.rag_db), ensure_ascii=False, indent=2))
        return

    if args.rag_search:
        print(
            json.dumps(
                rag_search(args.rag_search, db_path=args.rag_db, limit=args.rag_limit),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.rag_eval:
        print(json.dumps(evaluate_rag_golden_set(args.rag_db), ensure_ascii=False, indent=2))
        return

    if args.eval_golden_set:
        print(
            json.dumps(
                evaluate_pipeline_golden_set(
                    fixture_path=args.eval_golden_fixture,
                    report_path=args.eval_golden_report,
                    rag_db_path=args.rag_db,
                    output_language=args.output_language,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.export_run:
        run = get_run(args.export_run, args.profile_db)
        if run is None:
            parser.error(f"Run {args.export_run} not found in {args.profile_db}.")
        if run.get("review_status", "aprovado") != "aprovado":
            parser.error(
                f"Run {args.export_run} is not approved for export "
                f"(review_status={run.get('review_status')})."
            )
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
        rag_db_path=args.rag_db,
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
