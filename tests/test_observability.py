from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.storage import agent_trace_summary, get_agent_traces, save_run


def test_pipeline_records_and_persists_agent_traces(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("RADAR_ENABLE_WEB_FETCH", "false")
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "deterministic")
    monkeypatch.setenv("RAG_RERANKER", "heuristic")

    state = run_radar(
        query=(
            "Noleak usa NVIDIA GPUs, TensorRT e Triton Inference Server para "
            "visao computacional em cameras de seguranca."
        ),
        rag_db_path=str(tmp_path / "rag.sqlite"),
    )

    traces = state["agent_traces"]
    assert [trace["agent"] for trace in traces] == [
        "search_planner",
        "scraper",
        "extractor",
        "startup_classifier",
        "evidence_validator",
        "nvidia_rag",
        "recommendation",
        "economic_estimator",
        "llm_as_judge",
        "briefing",
    ]
    assert all(trace["execution_id"] == state["execution_id"] for trace in traces)
    assert all(trace["latency_ms"] >= 0 for trace in traces)
    assert all(trace["success"] for trace in traces)
    assert all("estimated_cost_usd" in trace for trace in traces)
    assert any(trace["modo_execucao"] == "fallback_sem_chave" for trace in traces)

    db_path = tmp_path / "profiles.sqlite"
    run_id = save_run(state, db_path)
    persisted = get_agent_traces(run_id, db_path)
    summary = agent_trace_summary(db_path)

    assert len(persisted) == len(traces)
    assert all(trace["run_id"] == run_id for trace in persisted)
    assert all("estimated_cost_usd" in trace for trace in persisted)
    assert any(row["agent"] == "startup_classifier" for row in summary)
    assert any(row["fallback"] >= 1 for row in summary)
    assert all("custo_estimado_usd" in row for row in summary)
