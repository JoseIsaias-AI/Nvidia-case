from nvidia_startup_ai_radar.agents import (
    briefing_agent,
    classifier_agent,
    economic_estimator_agent,
    evidence_validator_agent,
    extractor_agent,
    judge_agent,
    nvidia_rag_agent,
    recommendation_agent,
    scraper_agent,
    search_planner_agent,
)
from nvidia_startup_ai_radar.storage import get_run, list_recent_runs, save_run


def test_offline_agent_sequence_generates_briefing():
    state = {
        "query": (
            "Noleak usa NVIDIA GPUs, TensorRT e Triton Inference Server para "
            "visao computacional em cameras de seguranca, com P&D em universidades."
        ),
        "output_language": "pt",
        "errors": [],
    }
    for node in [
        search_planner_agent,
        scraper_agent,
        extractor_agent,
        classifier_agent,
        evidence_validator_agent,
        nvidia_rag_agent,
        recommendation_agent,
        economic_estimator_agent,
        judge_agent,
        briefing_agent,
    ]:
        state.update(node(state))

    assert state["profile"]["classificacao"] in {"AI-native", "AI-enabled"}
    assert state["profile"]["score_componentes"]
    assert state["profile"]["recomendacoes_nvidia"]
    assert "Briefing NVIDIA Startup AI Radar" in state["briefing_pt"]


def test_classifier_flags_thin_wrapper_risk():
    state = {
        "profile": {
            "nome": "PDFBuddy",
            "setor": "IA generativa",
            "produto_descricao": (
                "Produto de chat com PDF powered by GPT-4, construido como wrapper "
                "sobre OpenAI API, sem vagas tecnicas e apenas vendas/growth."
            ),
            "evidencias": [
                {
                    "fonte_url": "local://test",
                    "trecho_resumido": (
                        "Produto de chat com PDF powered by GPT-4, construido como wrapper "
                        "sobre OpenAI API, sem vagas tecnicas e apenas vendas/growth."
                    ),
                }
            ],
        }
    }

    result = classifier_agent(state)
    profile = result["profile"]

    assert profile["classificacao"] == "non-AI"
    assert profile["score_wrapper_risco"] >= 35
    assert profile["sinais_wrapper_risco"]
    assert any(component["tipo"] == "negativo" for component in profile["score_componentes"])


def test_save_run_persists_structured_profile(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    state = {
        "output_language": "pt",
        "human_review_required": False,
        "errors": [],
        "briefing_pt": "# Briefing NVIDIA Startup AI Radar: Noleak",
        "profile": {
            "id": "noleak",
            "nome": "Noleak",
            "setor": "Seguranca",
            "origem": "outbound",
            "classificacao": "AI-native",
            "score_maturidade_ia": 72,
            "score_wrapper_risco": 0,
            "evidencias": [
                {
                    "fonte_url": "local://test",
                    "trecho_resumido": "Noleak usa NVIDIA GPUs e Triton.",
                }
            ],
        },
    }

    run_id = save_run(state, db_path)
    recent = list_recent_runs(db_path)

    assert run_id == 1
    assert recent[0]["nome"] == "Noleak"
    assert recent[0]["classificacao"] == "AI-native"
    assert recent[0]["score_maturidade_ia"] == 72

    detailed = get_run(run_id, db_path)
    assert detailed is not None
    assert detailed["profile"]["id"] == "noleak"
    assert detailed["briefing_pt"].startswith("# Briefing NVIDIA")


def test_list_recent_runs_filters_by_classification_and_review(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    base_state = {
        "output_language": "pt",
        "errors": [],
        "briefing_pt": "# Briefing NVIDIA Startup AI Radar",
        "profile": {
            "nome": "PDFBuddy",
            "setor": "IA generativa",
            "origem": "outbound",
            "classificacao": "non-AI",
            "score_maturidade_ia": 0,
            "score_wrapper_risco": 66,
            "evidencias": [{"fonte_url": "local://test", "trecho_resumido": "wrapper"}],
        },
    }
    save_run({**base_state, "human_review_required": True}, db_path)
    save_run(
        {
            **base_state,
            "human_review_required": False,
            "profile": {
                **base_state["profile"],
                "nome": "Noleak",
                "setor": "Seguranca",
                "classificacao": "AI-enabled",
                "score_maturidade_ia": 52,
                "score_wrapper_risco": 0,
            },
        },
        db_path,
    )

    review_runs = list_recent_runs(db_path, human_review_required=True)
    ai_enabled_runs = list_recent_runs(db_path, classificacao="AI-enabled")
    search_runs = list_recent_runs(db_path, search="noleak")

    assert [run["nome"] for run in review_runs] == ["PDFBuddy"]
    assert [run["nome"] for run in ai_enabled_runs] == ["Noleak"]
    assert [run["nome"] for run in search_runs] == ["Noleak"]
