"""Outbound batch execution with SQLite checkpointing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import time
from pathlib import Path
from typing import Any, Literal

from nvidia_startup_ai_radar.discovery import (
    DEFAULT_DISCOVERY_DB_PATH,
    DiscoveryCandidate,
    candidate_key,
    discover_startups_for_theme,
    save_candidates_sqlite,
)
from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.schemas import utc_now_iso
from nvidia_startup_ai_radar.storage import (
    DEFAULT_DB_PATH,
    finish_batch_run,
    get_or_create_batch_run,
    save_run,
    successful_startup_ids,
    upsert_batch_item,
)


def _batch_key(query: str, max_results: int) -> str:
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(f"{normalized}|{max_results}".encode("utf-8")).hexdigest()[:24]


def _candidate_id(candidate: DiscoveryCandidate) -> str:
    raw_key = candidate_key(candidate)
    return f"disc-{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:24]}"


def _candidate_profile(candidate: DiscoveryCandidate, topic: str) -> dict[str, Any]:
    site = candidate.company_website or candidate.url
    evidence = candidate.evidence_excerpt or candidate.snippet or candidate.title
    description_parts = [
        f"Tema outbound: {topic}.",
        candidate.analysis_query,
        f"Fonte: {candidate.url}.",
        f"Evidencia: {evidence}",
    ]
    return {
        "id": _candidate_id(candidate),
        "nome": candidate.name,
        "site": site,
        "origem": "outbound",
        "produto_descricao": " ".join(part for part in description_parts if part),
        "evidencias": [
            {
                "fonte_url": candidate.url,
                "trecho_resumido": evidence or candidate.name,
                "data_coleta": candidate.collected_at,
            }
        ],
    }


def _process_candidate(
    *,
    candidate: DiscoveryCandidate,
    topic: str,
    batch_id: int,
    profile_db_path: str | Path,
    rag_db_path: str | Path | None,
    output_language: Literal["pt", "en", "both"],
) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    upsert_batch_item(
        batch_id=batch_id,
        candidate_key=candidate_id,
        name=candidate.name,
        url=candidate.url,
        status="running",
        db_path=profile_db_path,
    )
    try:
        inbound_profile = _candidate_profile(candidate, topic)
        final_state = run_radar(
            query=candidate.analysis_query or inbound_profile["produto_descricao"],
            inbound_profile=inbound_profile,
            output_language=output_language,
            rag_db_path=str(rag_db_path) if rag_db_path else None,
        )
        profile = dict(final_state.get("profile") or {})
        profile["id"] = candidate_id
        profile["nome"] = profile.get("nome") or candidate.name
        profile["site"] = profile.get("site") or candidate.company_website or candidate.url
        profile["origem"] = "outbound"
        final_state["profile"] = profile
        run_id = save_run(final_state, profile_db_path)
        upsert_batch_item(
            batch_id=batch_id,
            candidate_key=candidate_id,
            name=candidate.name,
            url=candidate.url,
            status="success",
            run_id=run_id,
            db_path=profile_db_path,
        )
        return {
            "candidate_key": candidate_id,
            "name": candidate.name,
            "url": candidate.url,
            "status": "success",
            "run_id": run_id,
        }
    except Exception as exc:
        upsert_batch_item(
            batch_id=batch_id,
            candidate_key=candidate_id,
            name=candidate.name,
            url=candidate.url,
            status="failed",
            error_message=str(exc),
            db_path=profile_db_path,
        )
        return {
            "candidate_key": candidate_id,
            "name": candidate.name,
            "url": candidate.url,
            "status": "failed",
            "error": str(exc),
        }


def run_batch_query(
    *,
    query: str,
    max_results: int = 15,
    concurrency: int = 2,
    rate_limit_seconds: float = 1.0,
    results_per_query: int = 4,
    fetch_pages: bool = True,
    profile_db_path: str | Path = DEFAULT_DB_PATH,
    discovery_db_path: str | Path = DEFAULT_DISCOVERY_DB_PATH,
    rag_db_path: str | Path | None = None,
    output_language: Literal["pt", "en", "both"] = "pt",
) -> dict[str, Any]:
    """Discover candidates for a topic and process them incrementally."""

    if not query.strip():
        raise ValueError("batch query cannot be empty")

    candidates = discover_startups_for_theme(
        query,
        limit=max_results,
        results_per_query=results_per_query,
        fetch_pages=fetch_pages,
        search_workers=8,
    )
    save_candidates_sqlite(candidates, discovery_db_path, replace=False)
    batch_id = get_or_create_batch_run(
        batch_key=_batch_key(query, max_results),
        query=query,
        max_results=max_results,
        db_path=profile_db_path,
    )
    candidate_ids = [_candidate_id(candidate) for candidate in candidates]
    already_successful = successful_startup_ids(candidate_ids, profile_db_path)
    results: list[dict[str, Any]] = []
    pending: list[DiscoveryCandidate] = []
    for candidate in candidates:
        candidate_id = _candidate_id(candidate)
        if candidate_id in already_successful:
            upsert_batch_item(
                batch_id=batch_id,
                candidate_key=candidate_id,
                name=candidate.name,
                url=candidate.url,
                status="skipped_existing",
                db_path=profile_db_path,
            )
            results.append(
                {
                    "candidate_key": candidate_id,
                    "name": candidate.name,
                    "url": candidate.url,
                    "status": "skipped_existing",
                }
            )
        else:
            upsert_batch_item(
                batch_id=batch_id,
                candidate_key=candidate_id,
                name=candidate.name,
                url=candidate.url,
                status="pending",
                db_path=profile_db_path,
            )
            pending.append(candidate)

    worker_count = max(1, min(concurrency, len(pending) or 1))
    if pending:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for candidate in pending:
                futures.append(
                    executor.submit(
                        _process_candidate,
                        candidate=candidate,
                        topic=query,
                        batch_id=batch_id,
                        profile_db_path=profile_db_path,
                        rag_db_path=rag_db_path,
                        output_language=output_language,
                    )
                )
                if rate_limit_seconds > 0:
                    time.sleep(rate_limit_seconds)
            for future in as_completed(futures):
                results.append(future.result())

    failed = sum(1 for result in results if result["status"] == "failed")
    finish_batch_run(
        batch_id=batch_id,
        status="failed" if failed else "completed",
        db_path=profile_db_path,
    )
    return {
        "batch_id": batch_id,
        "query": query,
        "evaluated_at": utc_now_iso(),
        "max_results": max_results,
        "discovered_count": len(candidates),
        "processed_count": sum(1 for result in results if result["status"] == "success"),
        "skipped_existing_count": sum(1 for result in results if result["status"] == "skipped_existing"),
        "failed_count": failed,
        "profile_db_path": str(profile_db_path),
        "discovery_db_path": str(discovery_db_path),
        "candidates": [asdict(candidate) for candidate in candidates],
        "results": sorted(results, key=lambda item: (item["status"], item["name"])),
    }
