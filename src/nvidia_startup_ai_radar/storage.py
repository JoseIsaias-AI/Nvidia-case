"""Local persistence for structured radar runs.

The project roadmap points to PostgreSQL for production. This module keeps the
MVP durable with SQLite first, using a table shape that can be migrated to
Postgres without changing the agent contracts.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, Protocol

from nvidia_startup_ai_radar.schemas import AgentState, StartupProfile, utc_now_iso


DEFAULT_DB_PATH = Path("data") / "radar_profiles.sqlite"
REVIEW_STATUSES = {"pendente", "aprovado", "rejeitado"}


class ProfileRepository(Protocol):
    """Persistence contract for profile runs.

    A future Postgres repository should implement this interface; the rest of
    the product can keep depending on the public functions in this module.
    """

    def initialize(self) -> None: ...

    def initialize_batch_store(self) -> None: ...

    def save(self, state: AgentState) -> int: ...

    def get(self, run_id: int) -> dict[str, Any] | None: ...

    def list(
        self,
        *,
        limit: int = 20,
        classificacao: str | None = None,
        setor: str | None = None,
        human_review_required: bool | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def filter(
        self,
        *,
        limit: int = 20,
        classificacao: str | None = None,
        setor: str | None = None,
        human_review_required: bool | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def list_profile_records(self, limit: int = 500) -> list[dict[str, Any]]: ...

    def list_review_queue(
        self,
        *,
        limit: int = 100,
        review_status: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def update_review_status(
        self,
        run_id: int,
        review_status: str,
        review_nota: str | None = None,
    ) -> None: ...

    def get_agent_traces(self, run_id: int) -> list[dict[str, Any]]: ...

    def agent_trace_summary(self) -> list[dict[str, Any]]: ...

    def list_distinct_values(self, column: Literal["classificacao", "setor", "origem"]) -> list[str]: ...

    def successful_startup_ids(self, startup_ids: list[str]) -> set[str]: ...

    def get_or_create_batch_run(self, *, batch_key: str, query: str, max_results: int) -> int: ...

    def upsert_batch_item(
        self,
        *,
        batch_id: int,
        candidate_key: str,
        name: str,
        url: str,
        status: str,
        run_id: int | None = None,
        error_message: str | None = None,
    ) -> None: ...

    def finish_batch_run(self, *, batch_id: int, status: str) -> None: ...


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


def _sqlite_initialize_store(db_path: str | Path = DEFAULT_DB_PATH) -> None:
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
                judge_json TEXT NOT NULL DEFAULT '{}',
                agent_execution_log_json TEXT NOT NULL DEFAULT '[]',
                errors_json TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'aprovado',
                review_nota TEXT,
                reviewed_at TEXT,
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
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(startup_profile_runs)").fetchall()
        }
        if "agent_execution_log_json" not in columns:
            connection.execute(
                "ALTER TABLE startup_profile_runs ADD COLUMN agent_execution_log_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "judge_json" not in columns:
            connection.execute("ALTER TABLE startup_profile_runs ADD COLUMN judge_json TEXT NOT NULL DEFAULT '{}'")
        if "review_status" not in columns:
            connection.execute(
                "ALTER TABLE startup_profile_runs ADD COLUMN review_status TEXT NOT NULL DEFAULT 'aprovado'"
            )
            connection.execute(
                "UPDATE startup_profile_runs SET review_status = 'pendente' WHERE human_review_required = 1"
            )
        if "review_nota" not in columns:
            connection.execute("ALTER TABLE startup_profile_runs ADD COLUMN review_nota TEXT")
        if "reviewed_at" not in columns:
            connection.execute("ALTER TABLE startup_profile_runs ADD COLUMN reviewed_at TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_run_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                execution_id TEXT,
                agent TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                modo_execucao TEXT NOT NULL,
                tokens_used INTEGER,
                estimated_cost_usd REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES startup_profile_runs(run_id)
            )
            """
        )
        trace_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(agent_run_traces)").fetchall()
        }
        if "estimated_cost_usd" not in trace_columns:
            connection.execute("ALTER TABLE agent_run_traces ADD COLUMN estimated_cost_usd REAL")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_traces_run
            ON agent_run_traces (run_id, started_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_traces_agent
            ON agent_run_traces (agent, modo_execucao, success)
            """
        )


def _sqlite_initialize_batch_store(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create checkpoint tables used by outbound batch execution."""

    _sqlite_initialize_store(db_path)
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS outbound_batch_runs (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_key TEXT NOT NULL UNIQUE,
                query TEXT NOT NULL,
                max_results INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS outbound_batch_items (
                batch_id INTEGER NOT NULL,
                candidate_key TEXT NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                status TEXT NOT NULL,
                run_id INTEGER,
                error_message TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (batch_id, candidate_key)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_batch_items_status
            ON outbound_batch_items (batch_id, status)
            """
        )


def _sqlite_get_or_create_batch_run(
    *,
    batch_key: str,
    query: str,
    max_results: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    _sqlite_initialize_batch_store(db_path)
    now = utc_now_iso()
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO outbound_batch_runs (batch_key, query, max_results, status, created_at, updated_at)
            VALUES (?, ?, ?, 'running', ?, ?)
            ON CONFLICT(batch_key) DO UPDATE SET
                query=excluded.query,
                max_results=excluded.max_results,
                status='running',
                updated_at=excluded.updated_at,
                completed_at=NULL
            """,
            (batch_key, query, max_results, now, now),
        )
        row = connection.execute(
            "SELECT batch_id FROM outbound_batch_runs WHERE batch_key = ?",
            (batch_key,),
        ).fetchone()
    return int(row["batch_id"])


def _sqlite_upsert_batch_item(
    *,
    batch_id: int,
    candidate_key: str,
    name: str,
    url: str,
    status: str,
    run_id: int | None = None,
    error_message: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    _sqlite_initialize_batch_store(db_path)
    now = utc_now_iso()
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO outbound_batch_items (
                batch_id, candidate_key, name, url, status, run_id, error_message, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_id, candidate_key) DO UPDATE SET
                name=excluded.name,
                url=excluded.url,
                status=excluded.status,
                run_id=excluded.run_id,
                error_message=excluded.error_message,
                updated_at=excluded.updated_at
            """,
            (batch_id, candidate_key, name, url, status, run_id, error_message, now),
        )


def _sqlite_finish_batch_run(
    *,
    batch_id: int,
    status: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    _sqlite_initialize_batch_store(db_path)
    now = utc_now_iso()
    with _connect(db_path) as connection:
        connection.execute(
            """
            UPDATE outbound_batch_runs
            SET status = ?, updated_at = ?, completed_at = ?
            WHERE batch_id = ?
            """,
            (status, now, now if status in {"completed", "failed"} else None, batch_id),
        )


def _sqlite_successful_startup_ids(
    startup_ids: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> set[str]:
    """Return startup IDs that already have a successfully persisted profile."""

    if not startup_ids:
        return set()
    _sqlite_initialize_store(db_path)
    placeholders = ",".join("?" for _ in startup_ids)
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT startup_id
            FROM startup_profile_runs
            WHERE startup_id IN ({placeholders})
            """,
            tuple(startup_ids),
        ).fetchall()
    return {str(row["startup_id"]) for row in rows if row["startup_id"]}


def _coerce_tokens_used(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _insert_agent_traces(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    traces: list[dict[str, Any]],
) -> None:
    if not traces:
        return
    now = utc_now_iso()
    connection.executemany(
        """
        INSERT INTO agent_run_traces (
            run_id,
            execution_id,
            agent,
            started_at,
            ended_at,
            latency_ms,
            success,
            error_message,
            modo_execucao,
            tokens_used,
            estimated_cost_usd,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                trace.get("execution_id") or trace.get("run_id"),
                str(trace.get("agent") or "unknown"),
                str(trace.get("started_at") or now),
                str(trace.get("ended_at") or now),
                float(trace.get("latency_ms") or 0.0),
                int(bool(trace.get("success", True))),
                trace.get("error") or trace.get("error_message"),
                str(trace.get("modo_execucao") or "deterministic"),
                _coerce_tokens_used(trace.get("tokens_used")),
                _coerce_float(trace.get("estimated_cost_usd") or trace.get("cost_usd")),
                now,
            )
            for trace in traces
        ],
    )


def _sqlite_save_run(
    state: AgentState,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persist the final graph state and return the inserted run id."""

    profile = StartupProfile.model_validate(state.get("profile") or {})
    _sqlite_initialize_store(db_path)
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
                judge_json,
                agent_execution_log_json,
                errors_json,
                review_status,
                review_nota,
                reviewed_at,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _json_dumps(state.get("judge", {})),
                _json_dumps(state.get("agent_execution_log", [])),
                _json_dumps(state.get("errors", [])),
                "pendente" if state.get("human_review_required") else "aprovado",
                None,
                None,
                utc_now_iso(),
            ),
        )
        run_id = int(cursor.lastrowid)
        _insert_agent_traces(connection, run_id=run_id, traces=state.get("agent_traces", []))
        return run_id


def _sqlite_get_agent_traces(
    run_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return the per-agent timeline for one persisted run."""

    _sqlite_initialize_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                run_id,
                execution_id,
                agent,
                started_at,
                ended_at,
                latency_ms,
                success,
                error_message,
                modo_execucao,
                tokens_used,
                estimated_cost_usd,
                created_at
            FROM agent_run_traces
            WHERE run_id = ?
            ORDER BY started_at, id
            """,
            (run_id,),
        ).fetchall()
    traces = [dict(row) for row in rows]
    for trace in traces:
        trace["success"] = bool(trace["success"])
    return traces


def _sqlite_agent_trace_summary(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Aggregate latency, error rate and execution mode by agent."""

    _sqlite_initialize_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                agent,
                COUNT(*) AS total_execucoes,
                ROUND(AVG(latency_ms), 3) AS latencia_media_ms,
                ROUND(MAX(latency_ms), 3) AS latencia_max_ms,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS erros,
                ROUND(
                    100.0 * SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) / COUNT(*),
                    2
                ) AS taxa_erro_pct,
                SUM(CASE WHEN modo_execucao = 'llm' THEN 1 ELSE 0 END) AS llm,
                SUM(CASE WHEN modo_execucao LIKE 'fallback%' THEN 1 ELSE 0 END) AS fallback,
                SUM(CASE WHEN modo_execucao = 'deterministic' THEN 1 ELSE 0 END) AS deterministic,
                SUM(COALESCE(tokens_used, 0)) AS tokens_usados,
                ROUND(SUM(COALESCE(estimated_cost_usd, 0)), 6) AS custo_estimado_usd
            FROM agent_run_traces
            GROUP BY agent
            ORDER BY agent
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _sqlite_list_recent_runs(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 20,
    classificacao: str | None = None,
    setor: str | None = None,
    human_review_required: bool | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent persisted runs for dashboards, audits or quick inspection."""

    _sqlite_initialize_store(db_path)
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
                profile_json,
                review_status,
                review_nota,
                reviewed_at,
                created_at
            FROM startup_profile_runs
            {where_clause}
            ORDER BY run_id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    runs: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["profile"] = _json_loads(item.pop("profile_json", None), {})
        runs.append(item)
    return runs


def _sqlite_get_run(
    run_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Return one persisted run with parsed profile and error payloads."""

    _sqlite_initialize_store(db_path)
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
                judge_json,
                agent_execution_log_json,
                errors_json,
                review_status,
                review_nota,
                reviewed_at,
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
    result["judge"] = _json_loads(result.pop("judge_json", None), {})
    result["agent_execution_log"] = _json_loads(result.pop("agent_execution_log_json", None), [])
    result["agent_traces"] = _sqlite_get_agent_traces(run_id, db_path)
    result["errors"] = _json_loads(result.pop("errors_json", None), [])
    result["human_review_required"] = bool(result["human_review_required"])
    result["review_motivos"] = review_reasons(result)
    return result


def _sqlite_list_profile_records(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return persisted runs with parsed StartupProfile payloads."""

    _sqlite_initialize_store(db_path)
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
                profile_json,
                briefing_pt,
                briefing_en,
                judge_json,
                agent_execution_log_json,
                errors_json,
                review_status,
                review_nota,
                reviewed_at,
                created_at
            FROM startup_profile_runs
            ORDER BY run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["profile"] = _json_loads(record.pop("profile_json", None), {})
        record["judge"] = _json_loads(record.pop("judge_json", None), {})
        record["agent_execution_log"] = _json_loads(record.pop("agent_execution_log_json", None), [])
        record["errors"] = _json_loads(record.pop("errors_json", None), [])
        record["human_review_required"] = bool(record["human_review_required"])
        record["review_motivos"] = review_reasons(record)
        records.append(record)
    return records


def review_reasons(record: dict[str, Any]) -> list[str]:
    """Summarize why a run was sent to human review."""

    reasons: list[str] = []
    for error in record.get("errors") or []:
        if error:
            reasons.append(f"Evidence Validator: {error}")
    judge = record.get("judge") or {}
    for motivo in judge.get("motivos") or []:
        if motivo:
            reasons.append(f"Judge: {motivo}")
    for divergencia in judge.get("divergencias_golden_set") or []:
        if divergencia:
            reasons.append(f"Golden set: {divergencia}")
    if record.get("human_review_required") and not reasons:
        reasons.append("Sinalizado para revisao humana pelo pipeline sem motivo estruturado salvo.")
    return reasons


def _sqlite_list_review_queue(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 100,
    review_status: str | None = None,
) -> list[dict[str, Any]]:
    """List runs that require human review, including approved/rejected decisions."""

    _sqlite_initialize_store(db_path)
    filters = ["human_review_required = 1"]
    params: list[Any] = []
    if review_status:
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"Invalid review_status: {review_status}")
        filters.append("review_status = ?")
        params.append(review_status)
    where_clause = " AND ".join(filters)
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
                profile_json,
                briefing_pt,
                briefing_en,
                judge_json,
                agent_execution_log_json,
                errors_json,
                review_status,
                review_nota,
                reviewed_at,
                created_at
            FROM startup_profile_runs
            WHERE {where_clause}
            ORDER BY
                CASE review_status
                    WHEN 'pendente' THEN 0
                    WHEN 'rejeitado' THEN 1
                    ELSE 2
                END,
                run_id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    queue: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["profile"] = _json_loads(item.pop("profile_json", None), {})
        item["judge"] = _json_loads(item.pop("judge_json", None), {})
        item["agent_execution_log"] = _json_loads(item.pop("agent_execution_log_json", None), [])
        item["errors"] = _json_loads(item.pop("errors_json", None), [])
        item["human_review_required"] = bool(item["human_review_required"])
        item["review_motivos"] = review_reasons(item)
        queue.append(item)
    return queue


def _sqlite_update_review_status(
    run_id: int,
    review_status: str,
    review_nota: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Approve or reject a human-reviewed run."""

    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review_status: {review_status}")
    _sqlite_initialize_store(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE startup_profile_runs
            SET review_status = ?, review_nota = ?, reviewed_at = ?
            WHERE run_id = ?
            """,
            (review_status, review_nota or None, utc_now_iso(), run_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Run {run_id} not found")


def _sqlite_list_distinct_values(
    column: Literal["classificacao", "setor", "origem"],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[str]:
    """List stored values for dashboard filters."""

    _sqlite_initialize_store(db_path)
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


class SQLiteProfileRepository:
    """SQLite implementation of the profile repository contract."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        _sqlite_initialize_store(self.db_path)

    def initialize_batch_store(self) -> None:
        _sqlite_initialize_batch_store(self.db_path)

    def save(self, state: AgentState) -> int:
        return _sqlite_save_run(state, self.db_path)

    def save_run(self, state: AgentState) -> int:
        return self.save(state)

    def get(self, run_id: int) -> dict[str, Any] | None:
        return _sqlite_get_run(run_id, self.db_path)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        return self.get(run_id)

    def list(
        self,
        *,
        limit: int = 20,
        classificacao: str | None = None,
        setor: str | None = None,
        human_review_required: bool | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        return _sqlite_list_recent_runs(
            self.db_path,
            limit=limit,
            classificacao=classificacao,
            setor=setor,
            human_review_required=human_review_required,
            search=search,
        )

    def filter(
        self,
        *,
        limit: int = 20,
        classificacao: str | None = None,
        setor: str | None = None,
        human_review_required: bool | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.list(
            limit=limit,
            classificacao=classificacao,
            setor=setor,
            human_review_required=human_review_required,
            search=search,
        )

    def list_recent_runs(
        self,
        *,
        limit: int = 20,
        classificacao: str | None = None,
        setor: str | None = None,
        human_review_required: bool | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.list(
            limit=limit,
            classificacao=classificacao,
            setor=setor,
            human_review_required=human_review_required,
            search=search,
        )

    def list_profile_records(self, limit: int = 500) -> list[dict[str, Any]]:
        return _sqlite_list_profile_records(self.db_path, limit=limit)

    def list_review_queue(
        self,
        *,
        limit: int = 100,
        review_status: str | None = None,
    ) -> list[dict[str, Any]]:
        return _sqlite_list_review_queue(self.db_path, limit=limit, review_status=review_status)

    def update_review_status(
        self,
        run_id: int,
        review_status: str,
        review_nota: str | None = None,
    ) -> None:
        _sqlite_update_review_status(run_id, review_status, review_nota, self.db_path)

    def get_agent_traces(self, run_id: int) -> list[dict[str, Any]]:
        return _sqlite_get_agent_traces(run_id, self.db_path)

    def agent_trace_summary(self) -> list[dict[str, Any]]:
        return _sqlite_agent_trace_summary(self.db_path)

    def list_distinct_values(self, column: Literal["classificacao", "setor", "origem"]) -> list[str]:
        return _sqlite_list_distinct_values(column, self.db_path)

    def successful_startup_ids(self, startup_ids: list[str]) -> set[str]:
        return _sqlite_successful_startup_ids(startup_ids, self.db_path)

    def get_or_create_batch_run(self, *, batch_key: str, query: str, max_results: int) -> int:
        return _sqlite_get_or_create_batch_run(
            batch_key=batch_key,
            query=query,
            max_results=max_results,
            db_path=self.db_path,
        )

    def upsert_batch_item(
        self,
        *,
        batch_id: int,
        candidate_key: str,
        name: str,
        url: str,
        status: str,
        run_id: int | None = None,
        error_message: str | None = None,
    ) -> None:
        _sqlite_upsert_batch_item(
            batch_id=batch_id,
            candidate_key=candidate_key,
            name=name,
            url=url,
            status=status,
            run_id=run_id,
            error_message=error_message,
            db_path=self.db_path,
        )

    def finish_batch_run(self, *, batch_id: int, status: str) -> None:
        _sqlite_finish_batch_run(batch_id=batch_id, status=status, db_path=self.db_path)


def get_profile_repository(db_path: str | Path = DEFAULT_DB_PATH) -> ProfileRepository:
    backend = os.getenv("PROFILE_REPOSITORY_BACKEND", "sqlite").strip().lower() or "sqlite"
    if backend != "sqlite":
        raise ValueError(
            "PROFILE_REPOSITORY_BACKEND suporta apenas 'sqlite' neste MVP. "
            "Implemente ProfileRepository para habilitar outro backend."
        )
    return SQLiteProfileRepository(db_path)


def initialize_store(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    get_profile_repository(db_path).initialize()


def initialize_batch_store(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    get_profile_repository(db_path).initialize_batch_store()


def get_or_create_batch_run(
    *,
    batch_key: str,
    query: str,
    max_results: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    return get_profile_repository(db_path).get_or_create_batch_run(
        batch_key=batch_key,
        query=query,
        max_results=max_results,
    )


def upsert_batch_item(
    *,
    batch_id: int,
    candidate_key: str,
    name: str,
    url: str,
    status: str,
    run_id: int | None = None,
    error_message: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    get_profile_repository(db_path).upsert_batch_item(
        batch_id=batch_id,
        candidate_key=candidate_key,
        name=name,
        url=url,
        status=status,
        run_id=run_id,
        error_message=error_message,
    )


def finish_batch_run(
    *,
    batch_id: int,
    status: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    get_profile_repository(db_path).finish_batch_run(batch_id=batch_id, status=status)


def successful_startup_ids(
    startup_ids: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> set[str]:
    return get_profile_repository(db_path).successful_startup_ids(startup_ids)


def save_run(
    state: AgentState,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    return get_profile_repository(db_path).save(state)


def get_agent_traces(
    run_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    return get_profile_repository(db_path).get_agent_traces(run_id)


def agent_trace_summary(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    return get_profile_repository(db_path).agent_trace_summary()


def list_recent_runs(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 20,
    classificacao: str | None = None,
    setor: str | None = None,
    human_review_required: bool | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    return get_profile_repository(db_path).list(
        limit=limit,
        classificacao=classificacao,
        setor=setor,
        human_review_required=human_review_required,
        search=search,
    )


def get_run(
    run_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    return get_profile_repository(db_path).get(run_id)


def list_profile_records(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 500,
) -> list[dict[str, Any]]:
    return get_profile_repository(db_path).list_profile_records(limit=limit)


def list_review_queue(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 100,
    review_status: str | None = None,
) -> list[dict[str, Any]]:
    return get_profile_repository(db_path).list_review_queue(limit=limit, review_status=review_status)


def update_review_status(
    run_id: int,
    review_status: str,
    review_nota: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    get_profile_repository(db_path).update_review_status(run_id, review_status, review_nota)


def list_distinct_values(
    column: Literal["classificacao", "setor", "origem"],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[str]:
    return get_profile_repository(db_path).list_distinct_values(column)
