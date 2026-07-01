import json
from dataclasses import asdict

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
from nvidia_startup_ai_radar.discovery import (
    DISCOVERY_CAMPAIGNS,
    DiscoveryCandidate,
    SearchResult,
    build_candidate,
    campaign_queries,
    dedupe_candidates,
    extract_aws_startup_name,
    extract_candidate_name,
    find_signals,
    list_discovery_candidates,
    save_candidates_sqlite,
    score_discovery_candidate,
)
from nvidia_startup_ai_radar.exporting import export_run, markdown_to_pdf_bytes
from nvidia_startup_ai_radar.golden_set_eval import evaluate_pipeline_golden_set, load_golden_set
from nvidia_startup_ai_radar import rag as rag_module
from nvidia_startup_ai_radar.rag import (
    SourceDocument,
    build_chunks,
    embed_text,
    embedding_backend_info,
    evaluate_rag_golden_set,
    rag_search,
    rebuild_rag_index,
)
from nvidia_startup_ai_radar.rag_ingestion import ingest_rag_sources, load_ingested_source_payloads
from nvidia_startup_ai_radar.storage import get_run, list_recent_runs, save_run
from nvidia_startup_ai_radar.storage import list_review_queue, update_review_status


def test_golden_set_fixture_covers_planning_cases():
    cases = load_golden_set()
    ids = {case.id for case in cases}

    assert len(cases) == 10
    assert ids == {
        "cloudwalk-infinitepay",
        "qi-tech",
        "laura",
        "oncoai",
        "noleak",
        "mr-turing",
        "stark-bank",
        "wuri",
        "codewhisper",
        "cydoc",
    }


def test_golden_set_evaluation_writes_case_report(tmp_path, monkeypatch):
    fixture_path = tmp_path / "golden_set.json"
    report_path = tmp_path / "golden_set_eval.md"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": "pipeline_golden_set_v1",
                "cases": [
                    {
                        "id": "acme-fintech",
                        "question_id": "P-test",
                        "startup_name": "Acme Fintech",
                        "sector": "Fintech",
                        "input_text": "Fintech usa IA para credito e dados tabulares.",
                        "expected_classification": "AI-native",
                        "expected_priority": "alta",
                        "expected_technologies": ["RAPIDS"],
                        "expected_human_review": False,
                        "expected_rationale": "Caso sintetico para testar avaliacao.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_radar(**kwargs):
        inbound_profile = kwargs["inbound_profile"]
        return {
            "profile": {
                "nome": inbound_profile["nome"],
                "setor": inbound_profile["setor"],
                "origem": "inbound",
                "produto_descricao": inbound_profile["produto_descricao"],
                "classificacao": "AI-native",
                "score_maturidade_ia": 82,
                "score_wrapper_risco": 0,
                "recomendacoes_nvidia": [
                    {
                        "tecnologia": "RAPIDS",
                        "justificativa_tecnica": "Acelerar pipelines tabulares.",
                        "justificativa_negocio": "Reduzir tempo de analise de credito.",
                        "prioridade": "alta",
                        "complexidade": "media",
                        "proxima_acao": "Validar volume e SLA.",
                    }
                ],
            },
            "human_review_required": False,
            "agent_execution_modes": {"startup_classifier": "fallback_sem_chave"},
        }

    monkeypatch.setattr("nvidia_startup_ai_radar.golden_set_eval.run_radar", fake_run_radar)

    evaluation = evaluate_pipeline_golden_set(fixture_path=fixture_path, report_path=report_path)

    assert evaluation["metrics"]["classification_accuracy"] == 1.0
    assert evaluation["metrics"]["technology_hit_rate"] == 1.0
    assert evaluation["metrics"]["failed_case_count"] == 0
    assert report_path.exists()
    assert "Acme Fintech" in report_path.read_text(encoding="utf-8")


def test_offline_agent_sequence_generates_briefing(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    state = {
        "query": (
            "Noleak usa NVIDIA GPUs, TensorRT e Triton Inference Server para "
            "visao computacional em cameras de seguranca, com P&D em universidades."
        ),
        "output_language": "pt",
        "rag_db_path": str(tmp_path / "rag.sqlite"),
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
    assert state["retrieved_entries"]
    assert any(entry["document_type"] == "knowledge_entry" for entry in state["retrieved_entries"])
    assert "Briefing NVIDIA Startup AI Radar" in state["briefing_pt"]


def test_scraper_keeps_discovery_query_before_disabled_url_placeholders(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("RADAR_ENABLE_WEB_FETCH", "false")
    query = (
        "Startup candidata: Boosted.ai\n"
        "Fonte principal: https://aws.amazon.com/blogs/startups/boosted-ai\n"
        "Evidencia coletada: Boosted.ai uses generative AI, LLMs, proprietary dataset, "
        "model training, GPU inference, latency optimization and Amazon Bedrock."
    )

    result = scraper_agent(
        {
            "query": query,
            "urls": ["https://aws.amazon.com/blogs/startups/boosted-ai"],
            "errors": [],
        }
    )
    raw_pages = result["raw_pages"]

    assert raw_pages[0]["url"] == "local://query"
    assert "Boosted.ai" in raw_pages[0]["text"]
    assert raw_pages[1]["scrape_method"] == "disabled"
    assert "URL planejada para coleta futura" in raw_pages[1]["text"]


def test_discovery_candidate_fallback_scores_with_saved_signals(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("RADAR_ENABLE_WEB_FETCH", "false")
    query = (
        "Startup candidata: Boosted.ai\n"
        "Fonte principal: https://aws.amazon.com/blogs/startups/boosted-ai\n"
        "Evidencia coletada: Boosted.ai uses generative AI portfolio insights with LLM, "
        "proprietary dataset, model training, GPU inference, production latency work "
        "and Amazon Bedrock.\n"
        "Sinais detectados: gpu, llm, dataset, training, inference, amazon bedrock"
    )
    state = {
        "query": query,
        "urls": ["https://aws.amazon.com/blogs/startups/boosted-ai"],
        "run_mode": "outbound",
        "agent_execution_modes": {},
        "agent_execution_log": [],
        "errors": [],
    }

    state.update(scraper_agent(state))
    state.update(extractor_agent(state))
    state.update(classifier_agent(state))
    profile = state["profile"]

    assert profile["nome"] == "Boosted.ai"
    assert profile["site"] == "https://aws.amazon.com/blogs/startups/boosted-ai"
    assert profile["score_maturidade_ia"] > 0
    assert profile["classificacao"] in {"AI-native", "AI-enabled"}
    assert profile["sinais_ai_native"]
    assert "AWS Bedrock" in profile["stack_concorrente_detectada"]


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


def test_extractor_detects_competitor_stack_with_evidence_and_briefing_action(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    raw_pages = [
        {
            "url": "https://example.com/careers",
            "title": "Careers",
            "text": (
                "HealthAI esta contratando ML engineer para produto clinico com LLM, "
                "AWS Bedrock em producao e avaliacao de Azure OpenAI para novos fluxos."
            ),
        }
    ]

    extracted = extractor_agent(
        {
            "query": "HealthAI",
            "run_mode": "outbound",
            "raw_pages": raw_pages,
            "agent_execution_modes": {},
            "agent_execution_log": [],
            "errors": [],
        }
    )
    profile = extracted["profile"]

    assert "AWS Bedrock" in profile["stack_concorrente_detectada"]
    assert "Azure OpenAI" in profile["stack_concorrente_detectada"]
    assert profile["stack_concorrente_evidencias"]
    assert profile["stack_concorrente_evidencias"][0]["fonte_url"] == "https://example.com/careers"

    briefing = briefing_agent(
        {
            "profile": {
                **profile,
                "classificacao": "AI-enabled",
                "score_maturidade_ia": 42,
                "score_wrapper_risco": 0,
            },
            "judge": {"status": "aprovado", "confianca": 0.7, "motivos": []},
            "human_review_required": False,
            "agent_execution_modes": {},
            "agent_execution_log": [],
            "errors": [],
        }
    )

    assert "Stack concorrente detectada: AWS Bedrock, Azure OpenAI" in briefing["briefing_pt"]
    assert "substituicao/migracao" in briefing["briefing_pt"]


def test_classifier_falls_back_after_invalid_llm_output(monkeypatch):
    class InvalidStructuredLLM:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return {
                "score_maturidade_ia": 101,
                "score_wrapper_risco": 0,
                "explicacao_classificacao": "saida invalida proposital",
                "classificacao": "banana",
            }

    monkeypatch.setenv("LLM_PROVIDER", "nvidia_nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key")
    monkeypatch.setattr("nvidia_startup_ai_radar.agents.build_chat_model", lambda settings: InvalidStructuredLLM())

    state = {
        "agent_execution_modes": {},
        "agent_execution_log": [],
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
        },
    }

    result = classifier_agent(state)

    assert result["agent_execution_modes"]["startup_classifier"] == "fallback_apos_falha"
    assert result["agent_execution_log"][-1]["modo_execucao"] == "fallback_apos_falha"
    assert result["profile"]["classificacao"] == "non-AI"
    assert result["profile"]["score_wrapper_risco"] >= 35


def test_rate_limited_llm_opens_local_fallback_circuit(monkeypatch):
    calls = []

    class RateLimitedLLM:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            raise RuntimeError("429 rate_limit_exceeded")

    def fake_build_chat_model(settings):
        calls.append(settings.llm_provider)
        return RateLimitedLLM()

    monkeypatch.setenv("LLM_PROVIDER", "nvidia_nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key")
    monkeypatch.setattr("nvidia_startup_ai_radar.agents.build_chat_model", fake_build_chat_model)

    state = {
        "agent_execution_modes": {},
        "agent_execution_log": [],
        "errors": [],
        "profile": {
            "nome": "Boosted.ai",
            "setor": "Fintech",
            "produto_descricao": "Produto usa LLM, GPU inference, proprietary dataset e Amazon Bedrock.",
            "stack_concorrente_detectada": ["AWS Bedrock"],
            "evidencias": [{"fonte_url": "local://test", "trecho_resumido": "LLM GPU inference Amazon Bedrock"}],
        },
    }

    classified = classifier_agent(state)
    validated = evidence_validator_agent({**state, **classified})

    assert classified["llm_provider_unavailable"] is True
    assert classified["agent_execution_modes"]["startup_classifier"] == "fallback_apos_falha"
    assert validated["agent_execution_modes"]["evidence_validator"] == "fallback_apos_falha"
    assert len(calls) == 1


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


def test_review_queue_approval_and_rejection_persist(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    review_state = {
        "output_language": "pt",
        "human_review_required": True,
        "errors": ["Perfil sem evidencias rastreaveis; revisao humana recomendada."],
        "judge": {
            "status": "revisao_humana",
            "confianca": 0.4,
            "motivos": ["Evidencia insuficiente para recomendacao automatica."],
        },
        "briefing_pt": "# Briefing NVIDIA Startup AI Radar: ReviewMe",
        "profile": {
            "nome": "ReviewMe",
            "setor": "Healthtech",
            "origem": "outbound",
            "classificacao": "indeterminado",
            "score_maturidade_ia": 0,
            "score_wrapper_risco": 0,
            "evidencias": [{"fonte_url": "local://test", "trecho_resumido": "incompleto"}],
        },
    }
    approved_id = save_run(review_state, db_path)
    rejected_id = save_run(
        {
            **review_state,
            "profile": {**review_state["profile"], "nome": "RejectMe"},
            "briefing_pt": "# Briefing NVIDIA Startup AI Radar: RejectMe",
        },
        db_path,
    )

    pending = list_review_queue(db_path)
    assert {item["review_status"] for item in pending} == {"pendente"}
    assert any("Evidence Validator" in reason for reason in pending[0]["review_motivos"])
    assert any("Judge" in reason for reason in pending[0]["review_motivos"])

    update_review_status(approved_id, "aprovado", "OK para envio", db_path)
    update_review_status(rejected_id, "rejeitado", "Faltam fontes confiaveis", db_path)

    approved = get_run(approved_id, db_path)
    rejected = get_run(rejected_id, db_path)

    assert approved["review_status"] == "aprovado"
    assert approved["review_nota"] == "OK para envio"
    assert rejected["review_status"] == "rejeitado"
    assert rejected["review_nota"] == "Faltam fontes confiaveis"

    export_run(approved, tmp_path, "markdown")
    try:
        export_run(rejected, tmp_path, "markdown")
    except ValueError as exc:
        assert "nao aprovado" in str(exc)
    else:
        raise AssertionError("Rejected briefing should not export.")


def test_export_run_writes_markdown_and_pdf(tmp_path):
    db_path = tmp_path / "profiles.sqlite"
    state = {
        "output_language": "pt",
        "human_review_required": False,
        "errors": [],
        "briefing_pt": "# Briefing NVIDIA Startup AI Radar: Noleak\n\n## Diagnostico\n- OK",
        "profile": {
            "nome": "Noleak",
            "setor": "Seguranca",
            "origem": "outbound",
            "classificacao": "AI-enabled",
            "score_maturidade_ia": 52,
            "score_wrapper_risco": 0,
            "evidencias": [{"fonte_url": "local://test", "trecho_resumido": "nvidia"}],
        },
    }
    run_id = save_run(state, db_path)
    run = get_run(run_id, db_path)

    markdown_path = export_run(run, tmp_path, "markdown")
    pdf_path = export_run(run, tmp_path, "pdf")
    pdf_bytes = markdown_to_pdf_bytes(markdown_path.read_text(encoding="utf-8"))

    assert markdown_path.read_text(encoding="utf-8").startswith("<!-- NVIDIA Startup AI Radar")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_bytes.startswith(b"%PDF")


def test_rag_chunks_are_semantic_and_metadata_rich():
    chunks = build_chunks()

    assert chunks
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.source_url for chunk in chunks)
    assert all(chunk.token_count <= 320 for chunk in chunks)
    assert any(chunk.document_type == "knowledge_entry" for chunk in chunks)
    assert any(chunk.document_type == "historical_case" for chunk in chunks)
    assert any(chunk.metadata.get("chunk_strategy") == "semantic_section_paragraph_overlap" for chunk in chunks)
    assert all(chunk.metadata.get("caminho_secao") for chunk in chunks)
    assert all(chunk.metadata.get("fonte_url") for chunk in chunks)
    assert all(chunk.metadata.get("data_ultima_verificacao") for chunk in chunks)


def test_rag_embedding_falls_back_and_uses_cache(tmp_path, monkeypatch):
    def fail_neural_embed(text: str, model_name: str):
        raise RuntimeError("offline model cache missing")

    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "neural")
    monkeypatch.setenv("RAG_EMBEDDING_CACHE_DB", str(tmp_path / "embeddings.sqlite"))
    monkeypatch.setattr(rag_module, "_neural_embed_text", fail_neural_embed)

    vector = embed_text("startup usa GPU, Triton e TensorRT para inferencia")
    cached_vector = embed_text("startup usa GPU, Triton e TensorRT para inferencia")
    info = embedding_backend_info()

    assert len(vector) == 384
    assert vector == cached_vector
    assert info["backend"] == "deterministic"
    assert "offline model cache missing" in info["fallback_reason"]


def test_rag_reranker_cohere_without_key_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "deterministic")
    monkeypatch.setenv("RAG_EMBEDDING_CACHE_DB", str(tmp_path / "embeddings.sqlite"))
    monkeypatch.setenv("RAG_RERANKER", "cohere")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("RAG_COHERE_API_KEY", raising=False)

    rag_db = tmp_path / "rag.sqlite"
    rebuild_rag_index(rag_db)
    results = rag_search("wrapper OpenAI API externa pivots risco", db_path=rag_db, limit=5)

    assert results
    assert results[0]["reranker_backend"] in {"heuristic", "flashrank"}


def test_rag_ingestion_manual_sources_and_inline_cases(tmp_path):
    manual_dir = tmp_path / "nim"
    manual_dir.mkdir()
    (manual_dir / "nim-notes.md").write_text(
        "# NIM notas manuais\n\nNVIDIA NIM reduz dependencia de API externa e melhora inferencia.",
        encoding="utf-8",
    )

    manifest = ingest_rag_sources(tmp_path, network_enabled=False)
    payloads = load_ingested_source_payloads(tmp_path)

    assert manifest["manual_count"] == 1
    assert any(payload["status"] == "manual" for payload in payloads)
    assert any(payload["document_type"] == "historical_case" for payload in payloads)
    assert any(payload["document_type"] == "golden_case" for payload in payloads)


def test_rag_deduplicates_near_identical_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "deterministic")
    monkeypatch.setenv("RAG_EMBEDDING_CACHE_DB", str(tmp_path / "embeddings.sqlite"))
    text = (
        "# Triton duplicate\n\n"
        "## Inference\n"
        "NVIDIA Triton Inference Server acelera inferencia em GPU com batching e MLOps."
    )
    metadata = {
        "tecnologia": "Triton Inference Server",
        "categoria": "Inference serving",
        "problema_que_resolve": "Serving de modelos.",
        "sinais_de_gatilho": ["gpu", "triton", "inferencia"],
        "fonte_url": "local://dup",
        "data_ultima_verificacao": "2026-06-30",
    }
    doc_a = SourceDocument("dup-a", "manual_source", "Triton duplicate", "Inference", "local://dup", text, metadata)
    doc_b = SourceDocument("dup-b", "manual_source", "Triton duplicate", "Inference", "local://dup", text, metadata)

    chunks = build_chunks([doc_a, doc_b])
    indexed, deduplicated_count = rag_module._embed_and_deduplicate_chunks(chunks)

    assert len(chunks) >= 2
    assert deduplicated_count >= 1
    assert len(indexed) < len(chunks)


def test_rag_hybrid_search_retrieves_expected_technology(tmp_path):
    rag_db = tmp_path / "rag.sqlite"
    stats = rebuild_rag_index(rag_db)
    results = rag_search(
        "visao computacional cameras GPU Triton TensorRT NVIDIA",
        db_path=rag_db,
        limit=5,
    )
    evaluation = evaluate_rag_golden_set(rag_db)

    assert stats["chunk_count"] > 0
    assert any(
        expected in str(results[0]["tecnologia"])
        for expected in {"Triton Inference Server", "TensorRT-LLM", "NVIDIA Inception"}
    )
    assert any(result["tecnologia"] == "Triton Inference Server" for result in results)
    assert results[0]["semantic_score"] >= 0
    assert results[0]["bm25_score"] >= 0
    assert evaluation["precision_at_5_proxy"] >= 0.8


def test_discovery_scoring_detects_nvidia_fit_and_competitor_stack():
    text = (
        "AcmeAI uses PyTorch and TensorFlow for computer vision inference on GPU. "
        "The team is evaluating Amazon Bedrock and Azure OpenAI, while hiring ML Engineer "
        "roles for model serving and production latency work."
    )

    signals = find_signals(text)
    score = score_discovery_candidate(text, "https://aws.amazon.com/blogs/startups/acmeai")

    assert "pytorch" in signals["ai_framework"]
    assert "tensorflow" in signals["ai_framework"]
    assert "amazon bedrock" in signals["competitor_stack"]
    assert "azure openai" in signals["competitor_stack"]
    assert "ml engineer" in signals["maturity"]
    assert score >= 70
    assert extract_aws_startup_name("Boosted.ai’s generative AI portfolio manager surfaces insights") == "Boosted.ai"
    assert extract_aws_startup_name("How Patronus AI helps enterprises boost confidence") == "Patronus AI"
    assert extract_aws_startup_name("AWS Activate credits now accepted for Amazon Bedrock") is None


def test_discovery_campaigns_cover_planning_signals():
    full_queries = " ".join(campaign_queries("full")).lower()

    assert len(campaign_queries("full")) >= 40
    assert set(DISCOVERY_CAMPAIGNS) >= {
        "full",
        "competitors",
        "nvidia_fit",
        "frameworks",
        "careers",
        "sectors",
        "ai_native",
        "wrapper_risk",
    }
    for expected in [
        "amazon bedrock",
        "vertex ai",
        "azure openai",
        "pytorch",
        "tensorflow",
        "tensorrt",
        "triton",
        "ml engineer",
        "healthtech",
        "fintech",
        "gpt wrapper",
    ]:
        assert expected in full_queries


def test_discovery_candidate_extraction_and_dedupe(monkeypatch):
    def fake_fetch_page_text(url: str, timeout: int = 20) -> str:
        return (
            "NVIDIA GPU inference with Triton Inference Server, TensorRT, "
            "PyTorch and production throughput for enterprise AI."
        )

    monkeypatch.setattr("nvidia_startup_ai_radar.discovery.fetch_page_text", fake_fetch_page_text)
    result = SearchResult(
        query='site:ycombinator.com/companies "Artificial Intelligence"',
        title="acme-ai | Y Combinator",
        url="https://www.ycombinator.com/companies/acme-ai",
        snippet="AI infrastructure startup.",
    )

    candidate = build_candidate(result)
    duplicate = DiscoveryCandidate(
        **{
            **asdict(candidate),
            "score": candidate.score - 5,
        }
    )
    deduped = dedupe_candidates([duplicate, candidate])

    assert extract_candidate_name(result.title, result.url) == "Acme Ai"
    assert extract_candidate_name("Theorem", "https://www.ycombinator.com/companies/theorem-2") == "Theorem"
    assert extract_candidate_name(
        "Founding Applied AI Engineer at Kastle - Y Combinator",
        "https://www.ycombinator.com/companies/kastle/jobs/XSq5nJT-founding-applied-ai-engineer-at-kastle",
    ) == "Kastle"
    assert candidate.name == "Acme Ai"
    assert "triton inference server" in candidate.nvidia_signals
    assert "pytorch" in candidate.ai_framework_signals
    assert candidate.quality_tier == "alta"
    assert "Startup candidata: Acme Ai" in candidate.analysis_query
    assert len(deduped) == 1
    assert deduped[0].score == candidate.score


def test_discovery_rejects_tutorials_docs_and_generic_framework_pages(monkeypatch):
    from nvidia_startup_ai_radar.discovery import (
        SearchResult,
        build_candidate,
        is_accepted_startup_candidate,
        is_non_startup_search_result,
    )

    monkeypatch.setattr(
        "nvidia_startup_ai_radar.discovery.fetch_page_text",
        lambda url, timeout=20: "PyTorch tutorial, TensorFlow documentation and MLOps explanation.",
    )
    bad_results = [
        SearchResult(
            query='"PyTorch" "GPU" "startup" "AI"',
            title="Tutorial PyTorch : um guia rapido para voce entender agora",
            url="https://www.ufc.br/tutorial-pytorch",
            snippet="Guia educacional de PyTorch.",
        ),
        SearchResult(
            query='"TensorFlow" "computer vision" "startup"',
            title="TensorFlow - Wikipedia",
            url="https://pt.wikipedia.org/wiki/TensorFlow",
            snippet="Pagina enciclopedica sobre TensorFlow.",
        ),
        SearchResult(
            query='"PyTorch" "GPU" "startup" "AI"',
            title="PyTorch documentation - PyTorch main documentation",
            url="https://pytorch.org/docs/stable/index.html",
            snippet="Documentacao oficial da biblioteca.",
        ),
    ]

    for result in bad_results:
        candidate = build_candidate(result, fetch_pages=True)
        assert is_non_startup_search_result(result, candidate.evidence_excerpt)
        assert not is_accepted_startup_candidate(candidate)


def test_discovery_candidates_are_persisted_in_sqlite(tmp_path, monkeypatch):
    def fake_fetch_page_text(url: str, timeout: int = 20) -> str:
        return "NVIDIA GPU inference, PyTorch, model serving and latency optimization."

    monkeypatch.setattr("nvidia_startup_ai_radar.discovery.fetch_page_text", fake_fetch_page_text)
    candidate = build_candidate(
        SearchResult(
            query="test",
            title="RunLocal AI",
            url="https://www.ycombinator.com/companies/runlocal-ai",
            snippet="AI agent for optimizing ML model inference on edge hardware.",
            source_type="yc_directory",
            company_website="https://runlocal.ai",
            location="London, UK",
            team_size=3,
        )
    )
    db_path = tmp_path / "discovery.sqlite"

    saved_count = save_candidates_sqlite([candidate], db_path)
    rows = list_discovery_candidates(db_path, limit=10)

    assert saved_count == 1
    assert rows[0]["name"] == "RunLocal AI"
    assert rows[0]["quality_tier"] == "alta"
    assert rows[0]["company_website"] == "https://runlocal.ai"
    assert rows[0]["team_size"] == 3
    assert "gpu" in rows[0]["nvidia_signals"]
