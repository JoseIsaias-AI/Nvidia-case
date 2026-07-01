from fastapi.testclient import TestClient

from nvidia_startup_ai_radar.product_metrics import (
    build_opportunity_matrix,
    build_overview,
    build_profile_radar,
    build_readiness,
    is_displayable_profile_record,
    visible_text_guard,
)
from nvidia_startup_ai_radar.storage import save_run
from nvidia_startup_ai_radar.web_api import app


def _state(
    name: str,
    *,
    sector: str = "Fintech",
    classification: str = "AI-native",
    score: float = 82,
    technology: str = "NVIDIA NIM",
    competitor_stack: list[str] | None = None,
) -> dict:
    return {
        "output_language": "pt",
        "human_review_required": False,
        "errors": [],
        "briefing_pt": f"# Briefing: {name}\n- Recomendacao com evidencia.",
        "judge": {"status": "aprovado", "motivos": []},
        "profile": {
            "id": name.lower().replace(" ", "-"),
            "nome": name,
            "setor": sector,
            "origem": "outbound",
            "produto_descricao": f"{name} usa dados proprietarios e inferencia em producao.",
            "classificacao": classification,
            "score_maturidade_ia": score,
            "score_wrapper_risco": 12 if classification != "AI-native" else 3,
            "stack_tecnica_detectada": ["NVIDIA GPU", "Triton", "TensorRT"],
            "stack_concorrente_detectada": competitor_stack or [],
            "stack_concorrente_evidencias": [
                {
                    "sinal": "AWS Bedrock",
                    "evidencia_trecho": "Vaga cita AWS Bedrock",
                    "fonte_url": "https://example.com/jobs",
                }
            ]
            if competitor_stack
            else [],
            "sinais_ai_native": [
                {
                    "sinal": "infraestrutura propria",
                    "evidencia_trecho": "Opera inferencia propria com GPU",
                    "fonte_url": "https://example.com",
                },
                {
                    "sinal": "dados proprietarios",
                    "evidencia_trecho": "Mantem dataset proprietario",
                    "fonte_url": "https://example.com/data",
                },
            ],
            "sinais_wrapper_risco": [],
            "evidencias": [{"fonte_url": "https://example.com", "trecho_resumido": "Fonte publica"}],
            "recomendacoes_nvidia": [
                {
                    "tecnologia": technology,
                    "justificativa_tecnica": "Aderente ao uso de inferencia em producao.",
                    "justificativa_negocio": "Reduz latencia e melhora controle operacional.",
                    "prioridade": "alta",
                    "complexidade": "media",
                    "proxima_acao": "Medir baseline e preparar prova tecnica.",
                    "evidencias": [{"fonte_url": "https://docs.nvidia.com", "trecho_resumido": "Docs NVIDIA"}],
                }
            ],
        },
    }


def test_overview_uses_persisted_records_and_empty_threshold(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    save_run(_state("Alpha", sector="Fintech", technology="NVIDIA NIM"), db_path)
    save_run(_state("Beta", sector="Healthtech", technology="NVIDIA Clara"), db_path)

    early = build_overview(db_path)
    assert early["record_count"] == 2
    assert early["classification_by_sector"]["ready"] is False
    assert early["technology_ranking"]["items"] == []

    save_run(_state("Gamma", sector="Fintech", classification="AI-enabled", score=64, technology="NVIDIA NIM"), db_path)
    overview = build_overview(db_path)

    assert overview["classification_by_sector"]["ready"] is True
    assert {item["name"]: item["count"] for item in overview["classification_by_sector"]["classification_counts"]} == {
        "AI-native": 2,
        "AI-enabled": 1,
    }
    assert overview["technology_ranking"]["items"][0]["technology"] == "NVIDIA NIM"
    assert overview["technology_ranking"]["items"][0]["count"] == 2
    assert sum(bucket["count"] for bucket in overview["maturity_distribution"]["bins"]) == 3


def test_profile_radar_reference_is_calculated_from_native_records(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    save_run(_state("Native One", sector="Fintech", score=90), db_path)
    save_run(_state("Native Two", sector="Healthtech", score=84), db_path)
    target_id = save_run(
        _state(
            "Target",
            sector="Retail",
            classification="AI-enabled",
            score=56,
            technology="NeMo Guardrails",
            competitor_stack=["AWS Bedrock"],
        ),
        db_path,
    )

    radar = build_profile_radar(target_id, db_path)

    assert radar["reference_available"] is True
    assert radar["reference_count"] == 2
    assert all(item["reference"] is not None for item in radar["axes"])
    assert {item["axis"] for item in radar["axes"]} >= {"Infra propria", "Dados exclusivos", "Baixo risco"}


def test_opportunity_matrix_maps_startups_to_nvidia_needs(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    save_run(_state("FinOps AI", sector="Fintech", technology="NVIDIA NIM"), db_path)
    save_run(_state("Clinical AI", sector="Healthtech", technology="NVIDIA Clara"), db_path)

    matrix = build_opportunity_matrix(db_path)
    client = TestClient(app)
    response = client.get("/api/opportunities", params={"db_path": str(db_path)})

    assert matrix["summary"]["startup_count"] == 2
    assert matrix["items"][0]["recommendations"]
    assert any(item["technology"] == "NVIDIA NIM" for item in matrix["technology_counts"])
    assert matrix["items"][0]["decision_bucket"]
    assert matrix["items"][0]["recommended_stack"]
    assert matrix["items"][0]["source_urls"] == ["https://docs.nvidia.com"]
    assert "priority_count" in matrix["summary"]
    assert response.status_code == 200
    assert response.json()["summary"]["startup_count"] == 2


def test_discovery_analyze_persists_candidates_in_radar(monkeypatch, tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    discovery_db_path = tmp_path / "discovery.sqlite"
    client = TestClient(app)

    def fake_list_discovery_candidates(_db_path, *, limit=100, quality_tier=None):
        assert limit == 1
        assert quality_tier is None
        return [
            {
                "name": "Candidate AI",
                "url": "https://example.com/candidate-ai",
                "analysis_query": "Candidate AI usa inferencia em tempo real e precisa reduzir latencia com NVIDIA NIM.",
                "evidence_excerpt": "Opera modelos proprietarios com GPU e inferencia em producao.",
                "nvidia_signals": ["NVIDIA GPU"],
                "ai_framework_signals": ["LLM"],
                "competitor_stack_signals": ["Azure OpenAI"],
                "maturity_signals": ["real-time inference"],
                "wrapper_risk_signals": [],
            }
        ]

    def fake_run_radar(*, query, output_language, rag_db_path):
        assert "Candidate AI" in query
        assert output_language == "pt"
        assert rag_db_path
        return _state("Candidate AI", sector="Fintech", technology="NVIDIA NIM", competitor_stack=["Azure OpenAI"])

    monkeypatch.setattr("nvidia_startup_ai_radar.web_api.list_discovery_candidates", fake_list_discovery_candidates)
    monkeypatch.setattr("nvidia_startup_ai_radar.web_api.run_radar", fake_run_radar)

    response = client.post(
        "/api/discovery/analyze",
        params={"db_path": str(db_path), "discovery_db_path": str(discovery_db_path)},
        json={"limit": 1, "quality": "todos", "output_language": "pt"},
    )
    assert response.status_code == 200
    assert response.json()["saved_count"] == 1

    runs = client.get("/api/runs", params={"db_path": str(db_path)}).json()["items"]
    opportunities = client.get("/api/opportunities", params={"db_path": str(db_path)}).json()
    assert runs[0]["nome"] == "Candidate AI"
    assert opportunities["summary"]["startup_count"] == 1
    assert opportunities["items"][0]["competitor_stack"] == ["Azure OpenAI"]


def test_discovery_analyze_processes_exact_visible_candidate_list(monkeypatch, tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    discovery_db_path = tmp_path / "discovery.sqlite"
    client = TestClient(app)
    candidates = [
        {
            "name": f"Visible AI {index}",
            "url": f"https://example.com/visible-{index}",
            "analysis_query": f"Visible AI {index} usa modelos proprietarios e precisa avaliar NVIDIA NIM.",
            "evidence_excerpt": "Produto com inferencia em producao e demanda por governanca.",
            "nvidia_signals": ["NVIDIA NIM"],
            "ai_framework_signals": ["LLM"],
            "competitor_stack_signals": [],
            "maturity_signals": ["inference"],
            "wrapper_risk_signals": [],
        }
        for index in range(1, 21)
    ]

    def fail_if_db_list_is_used(*args, **kwargs):
        raise AssertionError("A analise em lote deve usar a lista visivel enviada pela interface.")

    def fake_run_radar(*, query, output_language, rag_db_path):
        assert "Visible AI" in query
        return _state("Startup nao identificada", sector="Fintech", technology="NVIDIA NIM")

    monkeypatch.setattr("nvidia_startup_ai_radar.web_api.list_discovery_candidates", fail_if_db_list_is_used)
    monkeypatch.setattr("nvidia_startup_ai_radar.web_api.run_radar", fake_run_radar)

    response = client.post(
        "/api/discovery/analyze",
        params={"db_path": str(db_path), "discovery_db_path": str(discovery_db_path)},
        json={"limit": 20, "quality": "todos", "output_language": "pt", "candidates": candidates},
    )

    body = response.json()
    runs = client.get("/api/runs", params={"db_path": str(db_path), "limit": 50}).json()["items"]
    names = {run["nome"] for run in runs}

    assert response.status_code == 200
    assert body["requested_count"] == 20
    assert body["saved_count"] == 20
    assert len(runs) == 20
    assert {f"Visible AI {index}" for index in range(1, 21)} == names


def test_web_api_serves_runs_and_frontend_copy_guard(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    run_id = save_run(_state("API Ready"), db_path)
    client = TestClient(app)

    response = client.get("/api/runs", params={"db_path": str(db_path)})
    assert response.status_code == 200
    assert response.json()["items"][0]["run_id"] == run_id

    detail = client.get(f"/api/runs/{run_id}", params={"db_path": str(db_path)})
    assert detail.status_code == 200
    assert detail.json()["profile_radar"]["run_id"] == run_id

    guard = visible_text_guard("frontend")
    assert guard["hit_count"] == 0


def test_radar_filters_failed_or_educational_profile_records(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    save_run(_state("CloudWalk", sector="Fintech"), db_path)
    save_run(_state("QI Tech", sector="Fintech"), db_path)
    save_run(_state("Laura", sector="Healthtech"), db_path)
    save_run(
        _state(
            "TensorFlow - Wikipedia",
            sector="Nao identificado",
            classification="indeterminado",
            score=0,
        ),
        db_path,
    )
    save_run(
        _state(
            "Healthtech : o que sao e como inovam na Medicina? - Amigo Tech",
            sector="Healthtech",
            classification="AI-enabled",
            score=10,
        ),
        db_path,
    )
    broken_url_only = _state(
        "Broken Candidate",
        sector="Nao identificado",
        classification="indeterminado",
        score=0,
    )
    broken_url_only["profile"]["produto_descricao"] = "URL planejada para coleta futura: https://example.com/broken"
    broken_url_only["profile"]["evidencias"] = [
        {
            "fonte_url": "https://example.com/broken",
            "trecho_resumido": "URL planejada para coleta futura: https://example.com/broken",
        }
    ]
    save_run(broken_url_only, db_path)

    overview = build_overview(db_path)
    client = TestClient(app)
    response = client.get("/api/runs", params={"db_path": str(db_path), "limit": 10})
    names = {item["nome"] for item in response.json()["items"]}

    assert overview["record_count"] == 3
    assert "TensorFlow - Wikipedia" not in names
    assert "Healthtech : o que sao e como inovam na Medicina? - Amigo Tech" not in names
    assert "Broken Candidate" not in names
    assert is_displayable_profile_record(response.json()["items"][0])

    limited = client.get("/api/runs", params={"db_path": str(db_path), "limit": 3}).json()["items"]
    assert len(limited) == 3
    assert {item["nome"] for item in limited} == {"CloudWalk", "QI Tech", "Laura"}


def test_readiness_counts_displayable_startups_not_internal_limit(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    for index in range(7):
        save_run(_state(f"Visible {index}", sector="Fintech"), db_path)
    hidden = _state("Hidden Url Only", sector="Nao identificado", classification="indeterminado", score=0)
    hidden["profile"]["produto_descricao"] = "URL planejada para coleta futura: https://example.com/hidden"
    save_run(hidden, db_path)

    readiness = build_readiness(db_path=db_path, rag_db_path="data/radar_rag.sqlite")
    database_check = next(item for item in readiness["checks"] if item["id"] == "database")

    assert database_check["detail"] == "7 registros carregados"
