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
    assert state["profile"]["recomendacoes_nvidia"]
    assert "Briefing NVIDIA Startup AI Radar" in state["briefing_pt"]
