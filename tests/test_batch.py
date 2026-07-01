from nvidia_startup_ai_radar.agents import search_planner_agent
from nvidia_startup_ai_radar.batch import run_batch_query
from nvidia_startup_ai_radar.discovery import DiscoveryCandidate
from nvidia_startup_ai_radar.storage import list_recent_runs


def _candidate(name: str, url: str) -> DiscoveryCandidate:
    return DiscoveryCandidate(
        name=name,
        url=url,
        source_domain=url.split("/")[2],
        source_query="site:startse.com healthtech IA",
        title=name,
        snippet=f"{name} usa IA em produto vertical.",
        evidence_excerpt=f"{name} usa IA proprietaria com dados do setor.",
        score=72,
        nvidia_signals=["gpu"],
        ai_framework_signals=["machine learning"],
        competitor_stack_signals=[],
        maturity_signals=["model deployment"],
        wrapper_risk_signals=[],
        collected_at="2026-06-30T00:00:00+00:00",
        source_type="web_search",
        quality_tier="alta",
        recommended_action="Rodar analise profunda.",
        analysis_query=f"Startup candidata: {name}. Fonte principal: {url}. Evidencia coletada.",
    )


def test_search_planner_discovers_candidate_leads_from_theme(monkeypatch):
    candidates = [_candidate("Laura", "https://startse.com/laura-health-ai")]
    monkeypatch.setattr("nvidia_startup_ai_radar.agents.discover_startups_for_theme", lambda *args, **kwargs: candidates)

    result = search_planner_agent(
        {
            "query": "healthtechs com IA no Brasil",
            "discover_candidates": True,
            "discovery_limit": 1,
            "discovery_results_per_query": 2,
            "discovery_fetch_pages": False,
            "discovery_delay_seconds": 0,
            "discovery_search_workers": 1,
        }
    )

    assert result["candidate_leads"][0]["name"] == "Laura"
    assert result["urls"] == ["https://startse.com/laura-health-ai"]
    assert any("site:startse.com" in query for query in result["planned_searches"])
    assert any("site:neofeed.com.br" in query for query in result["planned_searches"])


def test_batch_query_saves_incrementally_and_skips_successful_on_resume(tmp_path, monkeypatch):
    candidates = [
        _candidate("Laura", "https://startse.com/laura-health-ai"),
        _candidate("OncoAI", "https://neofeed.com.br/oncoai"),
    ]
    calls = {"run_radar": 0}

    monkeypatch.setattr("nvidia_startup_ai_radar.batch.discover_startups_for_theme", lambda *args, **kwargs: candidates)

    def fake_run_radar(**kwargs):
        calls["run_radar"] += 1
        inbound = kwargs["inbound_profile"]
        return {
            "output_language": kwargs.get("output_language", "pt"),
            "human_review_required": False,
            "errors": [],
            "agent_execution_log": [],
            "briefing_pt": f"# Briefing NVIDIA Startup AI Radar: {inbound['nome']}",
            "profile": {
                "nome": inbound["nome"],
                "site": inbound["site"],
                "origem": "outbound",
                "classificacao": "AI-enabled",
                "score_maturidade_ia": 50,
                "score_wrapper_risco": 0,
                "produto_descricao": inbound["produto_descricao"],
                "evidencias": inbound["evidencias"],
            },
        }

    monkeypatch.setattr("nvidia_startup_ai_radar.batch.run_radar", fake_run_radar)

    profile_db = tmp_path / "profiles.sqlite"
    discovery_db = tmp_path / "discovery.sqlite"
    first = run_batch_query(
        query="healthtechs com IA no Brasil",
        max_results=2,
        concurrency=1,
        rate_limit_seconds=0,
        profile_db_path=profile_db,
        discovery_db_path=discovery_db,
    )
    second = run_batch_query(
        query="healthtechs com IA no Brasil",
        max_results=2,
        concurrency=1,
        rate_limit_seconds=0,
        profile_db_path=profile_db,
        discovery_db_path=discovery_db,
    )

    runs = list_recent_runs(profile_db, limit=10)

    assert first["processed_count"] == 2
    assert first["skipped_existing_count"] == 0
    assert second["processed_count"] == 0
    assert second["skipped_existing_count"] == 2
    assert calls["run_radar"] == 2
    assert len(runs) == 2
    assert {run["nome"] for run in runs} == {"Laura", "OncoAI"}
