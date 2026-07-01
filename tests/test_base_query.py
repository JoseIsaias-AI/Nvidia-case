from nvidia_startup_ai_radar.base_query import query_startup_base, translate_question_to_filter
from nvidia_startup_ai_radar.storage import save_run


def _save_profile(db_path, *, name, sector, classification, description, competitors=None, recommendations=None):
    state = {
        "output_language": "pt",
        "human_review_required": False,
        "errors": [],
        "briefing_pt": f"# Briefing NVIDIA Startup AI Radar: {name}",
        "profile": {
            "id": name.lower().replace(" ", "-"),
            "nome": name,
            "setor": sector,
            "origem": "outbound",
            "classificacao": classification,
            "score_maturidade_ia": 70,
            "score_wrapper_risco": 0,
            "produto_descricao": description,
            "stack_concorrente_detectada": competitors or [],
            "recomendacoes_nvidia": [
                {
                    "tecnologia": technology,
                    "justificativa_tecnica": "Teste",
                    "justificativa_negocio": "Teste",
                    "prioridade": "media",
                    "complexidade": "media",
                    "proxima_acao": "Teste",
                }
                for technology in (recommendations or [])
            ],
            "evidencias": [{"fonte_url": "local://test", "trecho_resumido": description}],
        },
    }
    return save_run(state, db_path)


def test_translate_question_to_filter_detects_guardrails_negation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")

    query_filter = translate_question_to_filter("quais startups de saude usam LLM mas nao tem guardrails")

    assert query_filter.setor == "Healthtech"
    assert "llm" in query_filter.required_terms
    assert "guardrails" in query_filter.missing_technologies


def test_query_startup_base_answers_three_fallback_questions(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    db_path = tmp_path / "profiles.sqlite"
    _save_profile(
        db_path,
        name="Health LLM",
        sector="Healthtech",
        classification="AI-enabled",
        description="Healthtech usa LLM para prontuario clinico e triagem medica.",
        recommendations=["NVIDIA Clara"],
    )
    _save_profile(
        db_path,
        name="Health Guard",
        sector="Healthtech",
        classification="AI-native",
        description="Healthtech usa LLM para prontuario clinico com governanca.",
        recommendations=["NeMo Guardrails", "NVIDIA Clara"],
    )
    _save_profile(
        db_path,
        name="Fin Bedrock",
        sector="Fintech",
        classification="AI-enabled",
        description="Fintech usa AWS Bedrock para analise de credito com modelos generativos.",
        competitors=["AWS Bedrock"],
        recommendations=["RAPIDS"],
    )
    _save_profile(
        db_path,
        name="Vision Triton",
        sector="Seguranca",
        classification="AI-native",
        description="Startup de visao computacional roda GPU e Triton em producao.",
        recommendations=["Triton Inference Server"],
    )

    health_without_guardrails = query_startup_base(
        "quais startups de saude usam LLM mas nao tem guardrails",
        db_path=db_path,
    )
    fintech_bedrock = query_startup_base("quais fintechs usam Bedrock", db_path=db_path)
    triton_native = query_startup_base("startups AI-native com Triton", db_path=db_path)

    assert [row["nome"] for row in health_without_guardrails["results"]] == ["Health LLM"]
    assert [row["nome"] for row in fintech_bedrock["results"]] == ["Fin Bedrock"]
    assert [row["nome"] for row in triton_native["results"]] == ["Vision Triton"]
