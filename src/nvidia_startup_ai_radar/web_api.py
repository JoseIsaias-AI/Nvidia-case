"""FastAPI layer for the product web interface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nvidia_startup_ai_radar.base_query import query_startup_base
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
from nvidia_startup_ai_radar.exporting import briefing_markdown, markdown_to_pdf_bytes, slugify
from nvidia_startup_ai_radar.golden_set_eval import evaluate_pipeline_golden_set
from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.product_metrics import (
    build_opportunity_matrix,
    build_overview,
    build_profile_radar,
    build_quality_summary,
    build_readiness,
    claim_quality_for_run,
    infer_needs_for_profile,
    is_displayable_profile_record,
    load_json_resource,
)
from nvidia_startup_ai_radar.rag import (
    DEFAULT_RAG_DB_PATH,
    evaluate_rag_golden_set,
    rag_index_stats,
    rag_search,
    rebuild_rag_index,
)
from nvidia_startup_ai_radar.rag_ingestion import DEFAULT_RAW_SOURCE_DIR, ingest_rag_sources
from nvidia_startup_ai_radar.storage import (
    DEFAULT_DB_PATH,
    agent_trace_summary,
    get_agent_traces,
    get_run,
    list_distinct_values,
    list_recent_runs,
    list_review_queue,
    save_run,
    update_review_status,
)


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT / "frontend" / "dist"
SOURCE_REGISTRY = ROOT / "config" / "source_registry.json"
ACTIVATION_PLAYBOOKS = ROOT / "config" / "activation_playbooks.json"


class AnalyzeRequest(BaseModel):
    query: str = ""
    inbound_profile: dict[str, Any] | None = None
    output_language: Literal["pt", "en", "both"] = "pt"
    save_profile: bool = True
    rag_db_path: str | None = None


class ReviewRequest(BaseModel):
    review_status: Literal["pendente", "aprovado", "rejeitado"]
    review_nota: str | None = None


class DiscoveryRequest(BaseModel):
    campaign: str = "full"
    limit: int = Field(default=20, ge=1, le=100)
    results_per_query: int = Field(default=6, ge=1, le=12)
    fetch_pages: bool = True
    queries: list[str] | None = None


class DiscoveryAnalyzeRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)
    quality: Literal["todos", "alta", "media", "baixa", "triagem"] = "todos"
    output_language: Literal["pt", "en", "both"] = "pt"
    rag_db_path: str | None = None
    candidates: list[dict[str, Any]] | None = Field(default=None, max_length=100)


class AskRequest(BaseModel):
    question: str
    limit: int = Field(default=10, ge=1, le=50)
    scan_limit: int = Field(default=500, ge=10, le=2000)


def _candidate_query(candidate: dict[str, Any]) -> str:
    signals: list[str] = []
    for key in [
        "nvidia_signals",
        "ai_framework_signals",
        "competitor_stack_signals",
        "maturity_signals",
        "wrapper_risk_signals",
    ]:
        values = candidate.get(key) or []
        if isinstance(values, list):
            signals.extend(str(item) for item in values if item)
    parts = [
        str(candidate.get("analysis_query") or "").strip(),
        f"Startup candidata: {candidate.get('name') or candidate.get('nome') or candidate.get('title') or 'Startup'}",
        f"Fonte principal: {candidate.get('url')}" if candidate.get("url") else "",
        str(candidate.get("evidence_excerpt") or candidate.get("snippet") or candidate.get("recommended_action") or ""),
        f"Sinais detectados: {', '.join(signals)}" if signals else "",
    ]
    return "\n".join(part for part in parts if part).strip()


def _candidate_name(candidate: dict[str, Any]) -> str:
    for key in ["name", "nome", "company_name"]:
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    title = str(candidate.get("title") or "").strip()
    if title:
        for separator in [" | ", " - ", " : "]:
            if separator in title:
                return title.split(separator, 1)[0].strip()
        return title
    return "Startup candidata"


def _looks_generic_startup_name(value: str | None) -> bool:
    normalized = " ".join(str(value or "").lower().split())
    return normalized in {
        "",
        "startup",
        "startup candidata",
        "startup nao identificada",
        "startup não identificada",
        "nao identificado",
        "não identificado",
    }


def _preserve_candidate_identity(final_state: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep visible discovery candidates visible after the analysis pipeline.

    Discovery already knows the candidate card the user clicked. If the
    extraction fallback cannot infer a proper company name, keep the discovery
    identity so the saved run does not disappear from the Radar filters.
    """

    profile = final_state.get("profile")
    if not isinstance(profile, dict):
        profile = {}
        final_state["profile"] = profile
    name = _candidate_name(candidate)
    if name and (_looks_generic_startup_name(profile.get("nome")) or profile.get("nome") != name):
        profile["nome"] = name
    if candidate.get("url") and not profile.get("site"):
        profile["site"] = candidate.get("url")
    profile.setdefault("origem", "outbound")
    if not profile.get("produto_descricao"):
        description = candidate.get("evidence_excerpt") or candidate.get("snippet") or candidate.get("recommended_action")
        if description:
            profile["produto_descricao"] = str(description)[:800]
    if not profile.get("setor"):
        profile["setor"] = "Nao identificado"
    return final_state


def create_app() -> FastAPI:
    app = FastAPI(
        title="NVIDIA Startup AI Radar Web API",
        version="0.1.0",
        description="Thin product API over the existing radar pipeline and storage.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "readiness": build_readiness()}

    @app.get("/api/readiness")
    def readiness(
        db_path: str = str(DEFAULT_DB_PATH),
        knowledge_db_path: str = str(DEFAULT_RAG_DB_PATH),
    ) -> dict[str, Any]:
        return build_readiness(db_path=db_path, rag_db_path=knowledge_db_path)

    @app.get("/api/features")
    def features() -> dict[str, str]:
        path = ROOT / "docs" / "guia-completo-do-case.md"
        return {"markdown": path.read_text(encoding="utf-8") if path.exists() else ""}

    @app.get("/api/runs")
    def runs(
        limit: int = Query(default=200, ge=1, le=1000),
        classificacao: str | None = None,
        setor: str | None = None,
        review_only: bool = False,
        search: str | None = None,
        db_path: str = str(DEFAULT_DB_PATH),
    ) -> dict[str, Any]:
        fetch_limit = limit if review_only else min(2000, max(limit * 5, limit))
        rows = list_recent_runs(
            db_path=db_path,
            limit=fetch_limit,
            classificacao=classificacao or None,
            setor=setor or None,
            human_review_required=True if review_only else None,
            search=search or None,
        )
        if not review_only:
            rows = [row for row in rows if is_displayable_profile_record(row)]
            rows = rows[:limit]
        return {
            "items": rows,
            "filters": {
                "classificacoes": list_distinct_values("classificacao", db_path),
                "setores": list_distinct_values("setor", db_path),
                "origens": list_distinct_values("origem", db_path),
            },
        }

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: int, db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        run = get_run(run_id, db_path)
        if run is None:
            raise HTTPException(status_code=404, detail="Execucao nao encontrada.")
        profile = run.get("profile") or {}
        return {
            "run": run,
            "quality": claim_quality_for_run(run),
            "profile_radar": build_profile_radar(run_id, db_path),
            "nvidia_needs": infer_needs_for_profile(profile),
        }

    @app.post("/api/analyze")
    def analyze(request: AnalyzeRequest, db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        if not request.query.strip() and not request.inbound_profile:
            raise HTTPException(status_code=400, detail="Informe um texto ou perfil estruturado.")
        final_state = run_radar(
            query=request.query.strip(),
            inbound_profile=request.inbound_profile or {},
            output_language=request.output_language,
            rag_db_path=request.rag_db_path or str(DEFAULT_RAG_DB_PATH),
        )
        saved_run_id = save_run(final_state, db_path) if request.save_profile else None
        return {"state": final_state, "saved_run_id": saved_run_id}

    @app.get("/api/overview")
    def overview(db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        return build_overview(db_path)

    @app.get("/api/opportunities")
    def opportunities(db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        return build_opportunity_matrix(db_path)

    @app.get("/api/runs/{run_id}/profile-radar")
    def profile_radar(run_id: int, db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        try:
            return build_profile_radar(run_id, db_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/review")
    def review_queue(
        status: Literal["todos", "pendente", "aprovado", "rejeitado"] = "pendente",
        limit: int = Query(default=100, ge=1, le=500),
        db_path: str = str(DEFAULT_DB_PATH),
    ) -> dict[str, Any]:
        rows = list_review_queue(db_path, limit=limit, review_status=None if status == "todos" else status)
        return {"items": rows}

    @app.patch("/api/review/{run_id}")
    def review_update(run_id: int, request: ReviewRequest, db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        try:
            update_review_status(run_id, request.review_status, request.review_nota, db_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/ask")
    def ask_base(request: AskRequest, db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        return query_startup_base(
            request.question,
            db_path=db_path,
            limit=request.limit,
            scan_limit=request.scan_limit,
        )

    @app.get("/api/knowledge/stats")
    def knowledge_stats(knowledge_db_path: str = str(DEFAULT_RAG_DB_PATH)) -> dict[str, Any]:
        return rag_index_stats(knowledge_db_path)

    @app.get("/api/knowledge/search")
    def knowledge_search(
        q: str,
        limit: int = Query(default=7, ge=1, le=20),
        knowledge_db_path: str = str(DEFAULT_RAG_DB_PATH),
    ) -> dict[str, Any]:
        return {"items": rag_search(q, db_path=knowledge_db_path, limit=limit)}

    @app.post("/api/knowledge/ingest")
    def knowledge_ingest(
        knowledge_db_path: str = str(DEFAULT_RAG_DB_PATH),
        source_dir: str = str(DEFAULT_RAW_SOURCE_DIR),
    ) -> dict[str, Any]:
        manifest = ingest_rag_sources(source_dir)
        index = rebuild_rag_index(knowledge_db_path, source_dir=source_dir)
        return {"ingest": manifest, "index": index}

    @app.post("/api/knowledge/rebuild")
    def knowledge_rebuild(
        knowledge_db_path: str = str(DEFAULT_RAG_DB_PATH),
        source_dir: str = str(DEFAULT_RAW_SOURCE_DIR),
    ) -> dict[str, Any]:
        return rebuild_rag_index(knowledge_db_path, source_dir=source_dir)

    @app.get("/api/knowledge/evaluation")
    def knowledge_evaluation(knowledge_db_path: str = str(DEFAULT_RAG_DB_PATH)) -> dict[str, Any]:
        return evaluate_rag_golden_set(knowledge_db_path)

    @app.post("/api/discovery")
    def discovery_run(
        request: DiscoveryRequest,
        discovery_db_path: str = str(DEFAULT_DISCOVERY_DB_PATH),
    ) -> dict[str, Any]:
        campaign_aliases = {
            "competitor_stack": "competitors",
            "sector_health": "sectors",
            "sector_finance": "sectors",
        }
        campaign = campaign_aliases.get(request.campaign, request.campaign)
        if campaign not in DISCOVERY_CAMPAIGNS:
            raise HTTPException(status_code=400, detail="Campanha invalida.")
        candidates = discover_startups(
            queries=request.queries,
            campaign=campaign,
            limit=request.limit,
            results_per_query=request.results_per_query,
            fetch_pages=request.fetch_pages,
        )
        output_path = save_candidates(candidates, DEFAULT_DISCOVERY_OUTPUT)
        saved_count = save_candidates_sqlite(candidates, discovery_db_path, replace=True)
        return {
            "output_path": str(output_path),
            "saved_count": saved_count,
            "items": candidates_as_dicts(candidates),
        }

    @app.get("/api/discovery")
    def discovery_list(
        quality: Literal["todos", "alta", "media", "baixa", "triagem"] = "todos",
        limit: int = Query(default=100, ge=1, le=500),
        discovery_db_path: str = str(DEFAULT_DISCOVERY_DB_PATH),
    ) -> dict[str, Any]:
        return {
            "items": list_discovery_candidates(
                discovery_db_path,
                limit=limit,
                quality_tier=None if quality == "todos" else quality,
            )
        }

    @app.post("/api/discovery/analyze")
    def discovery_analyze(
        request: DiscoveryAnalyzeRequest,
        db_path: str = str(DEFAULT_DB_PATH),
        discovery_db_path: str = str(DEFAULT_DISCOVERY_DB_PATH),
    ) -> dict[str, Any]:
        candidates = (
            request.candidates[: request.limit]
            if request.candidates
            else list_discovery_candidates(
                discovery_db_path,
                limit=request.limit,
                quality_tier=None if request.quality == "todos" else request.quality,
            )
        )
        saved: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for candidate in candidates:
            query = _candidate_query(candidate)
            if not query:
                failed.append({"candidate": candidate.get("name") or candidate.get("title"), "error": "Sem contexto para analise."})
                continue
            try:
                final_state = run_radar(
                    query=query,
                    output_language=request.output_language,
                    rag_db_path=request.rag_db_path or str(DEFAULT_RAG_DB_PATH),
                )
                final_state = _preserve_candidate_identity(final_state, candidate)
                run_id = save_run(final_state, db_path)
                saved.append(
                    {
                        "run_id": run_id,
                        "candidate": _candidate_name(candidate),
                        "profile_name": (final_state.get("profile") or {}).get("nome"),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive web boundary
                failed.append({"candidate": _candidate_name(candidate), "error": str(exc)})
        return {
            "requested_count": len(candidates),
            "saved_count": len(saved),
            "failed_count": len(failed),
            "saved": saved,
            "failed": failed,
        }

    @app.get("/api/observability/{run_id}")
    def run_observability(run_id: int, db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        traces = get_agent_traces(run_id, db_path)
        timeline = [
            {
                "step": trace.get("agent"),
                "started_at": trace.get("started_at"),
                "ended_at": trace.get("ended_at"),
                "latency_ms": trace.get("latency_ms"),
                "success": trace.get("success"),
                "mode": trace.get("modo_execucao"),
                "units": trace.get("tokens_used"),
                "estimated_cost_usd": trace.get("estimated_cost_usd"),
                "error": trace.get("error_message"),
            }
            for trace in traces
        ]
        return {"items": timeline}

    @app.get("/api/observability")
    def observability_summary(db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        summary = []
        for row in agent_trace_summary(db_path):
            total = int(row.get("total_execucoes") or 0)
            error_rate = float(row.get("taxa_erro_pct") or 0.0) / 100.0
            mode_counts = {
                "llm": int(row.get("llm") or 0),
                "fallback": int(row.get("fallback") or 0),
                "deterministic": int(row.get("deterministic") or 0),
            }
            dominant_mode = max(mode_counts, key=mode_counts.get) if total else "local"
            summary.append(
                {
                    "step": row.get("agent"),
                    "mode": dominant_mode,
                    "success_rate": max(0.0, 1.0 - error_rate),
                    "avg_latency_ms": row.get("latencia_media_ms"),
                    "count": total,
                    "error_count": int(row.get("erros") or 0),
                    "mode_counts": mode_counts,
                }
            )
        return {"items": summary}

    @app.get("/api/export/{run_id}")
    def export(
        run_id: int,
        format: Literal["markdown", "pdf"] = "pdf",
        db_path: str = str(DEFAULT_DB_PATH),
    ) -> Response:
        run = get_run(run_id, db_path)
        if run is None:
            raise HTTPException(status_code=404, detail="Execucao nao encontrada.")
        if run.get("review_status", "aprovado") != "aprovado":
            raise HTTPException(status_code=409, detail="Aprovacao necessaria antes da exportacao.")
        markdown = briefing_markdown(run)
        filename = f"run-{run_id}-{slugify(run.get('nome') or 'startup')}"
        if format == "markdown":
            return Response(
                markdown,
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{filename}.md"'},
            )
        return Response(
            markdown_to_pdf_bytes(markdown, title=str(run.get("nome") or filename)),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
        )

    @app.get("/api/quality")
    def quality_summary(db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        return build_quality_summary(db_path)

    @app.get("/api/evaluation/pipeline")
    def pipeline_evaluation() -> dict[str, Any]:
        return evaluate_pipeline_golden_set()

    @app.get("/api/sources")
    def sources() -> dict[str, Any]:
        return {"items": load_json_resource(SOURCE_REGISTRY) if SOURCE_REGISTRY.exists() else []}

    @app.get("/api/playbooks")
    def playbooks() -> dict[str, Any]:
        return {"items": load_json_resource(ACTIVATION_PLAYBOOKS) if ACTIVATION_PLAYBOOKS.exists() else []}

    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            target = FRONTEND_DIST / full_path
            if full_path and target.exists() and target.is_file():
                return FileResponse(target)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()


def launch() -> None:
    host = os.getenv("RADAR_WEB_HOST", "127.0.0.1")
    port = os.getenv("RADAR_WEB_PORT", "8000")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "nvidia_startup_ai_radar.web_api:app",
            "--reload",
            "--host",
            host,
            "--port",
            port,
        ],
        check=False,
    )
