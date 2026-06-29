"""Local persistence for structured radar runs.

The project roadmap points to PostgreSQL for production. This module keeps the
MVP durable with SQLite first, using a table shape that can be migrated to
Postgres without changing the agent contracts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

from nvidia_startup_ai_radar.schemas import AgentState, StartupProfile, utc_now_iso


DEFAULT_DB_PATH = Path("data") / "radar_profiles.sqlite"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_store(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the profile store if it does not exist yet."""

    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS startup_profile_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                startup_id TEXT,
                nome TEXT NOT NULL,
                setor TEXT,
                origem TEXT NOT NULL,
                classificacao TEXT NOT NULL,
                score_maturidade_ia REAL NOT NULL,
                score_wrapper_risco REAL NOT NULL,
                human_review_required INTEGER NOT NULL,
                output_language TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                briefing_pt TEXT,
                briefing_en TEXT,
                errors_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_profile_runs_lookup
            ON startup_profile_runs (classificacao, setor, score_maturidade_ia)
            """
        )


def save_run(
    state: AgentState,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persist the final graph state and return the inserted run id."""

    profile = StartupProfile.model_validate(state.get("profile") or {})
    initialize_store(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO startup_profile_runs (
                startup_id,
                nome,
                setor,
                origem,
                classificacao,
                score_maturidade_ia,
                score_wrapper_risco,
                human_review_required,
                output_language,
                profile_json,
                briefing_pt,
                briefing_en,
                errors_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.id,
                profile.nome,
                profile.setor,
                profile.origem,
                profile.classificacao,
                profile.score_maturidade_ia,
                profile.score_wrapper_risco,
                int(bool(state.get("human_review_required"))),
                state.get("output_language", "pt"),
                _json_dumps(profile.model_dump()),
                state.get("briefing_pt"),
                state.get("briefing_en"),
                _json_dumps(state.get("errors", [])),
                utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)


def list_recent_runs(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 20,
    classificacao: str | None = None,
    setor: str | None = None,
    human_review_required: bool | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent persisted runs for dashboards, audits or quick inspection."""

    initialize_store(db_path)
    filters: list[str] = []
    params: list[Any] = []
    if classificacao:
        filters.append("classificacao = ?")
        params.append(classificacao)
    if setor:
        filters.append("setor = ?")
        params.append(setor)
    if human_review_required is not None:
        filters.append("human_review_required = ?")
        params.append(int(human_review_required))
    if search:
        filters.append("(LOWER(nome) LIKE ? OR LOWER(COALESCE(setor, '')) LIKE ?)")
        search_term = f"%{search.lower()}%"
        params.extend([search_term, search_term])

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                run_id,
                startup_id,
                nome,
                setor,
                origem,
                classificacao,
                score_maturidade_ia,
                score_wrapper_risco,
                human_review_required,
                output_language,
                created_at
            FROM startup_profile_runs
            {where_clause}
            ORDER BY run_id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_run(
    run_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Return one persisted run with parsed profile and error payloads."""

    initialize_store(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                run_id,
                startup_id,
                nome,
                setor,
                origem,
                classificacao,
                score_maturidade_ia,
                score_wrapper_risco,
                human_review_required,
                output_language,
                profile_json,
                briefing_pt,
                briefing_en,
                errors_json,
                created_at
            FROM startup_profile_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["profile"] = _json_loads(result.pop("profile_json", None), {})
    result["errors"] = _json_loads(result.pop("errors_json", None), [])
    result["human_review_required"] = bool(result["human_review_required"])
    return result


def list_distinct_values(
    column: Literal["classificacao", "setor", "origem"],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[str]:
    """List stored values for dashboard filters."""

    initialize_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT {column}
            FROM startup_profile_runs
            WHERE {column} IS NOT NULL AND TRIM({column}) != ''
            ORDER BY {column}
            """
        ).fetchall()
    return [str(row[column]) for row in rows]
