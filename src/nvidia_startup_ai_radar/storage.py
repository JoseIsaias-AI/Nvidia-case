"""Local persistence for structured radar runs.

The project roadmap points to PostgreSQL for production. This module keeps the
MVP durable with SQLite first, using a table shape that can be migrated to
Postgres without changing the agent contracts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from nvidia_startup_ai_radar.schemas import AgentState, StartupProfile, utc_now_iso


DEFAULT_DB_PATH = Path("data") / "radar_profiles.sqlite"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


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
) -> list[dict[str, Any]]:
    """Return recent persisted runs for dashboards, audits or quick inspection."""

    initialize_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
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
                created_at
            FROM startup_profile_runs
            ORDER BY run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
