"""LangGraph node implementations for the NVIDIA Startup AI Radar."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from nvidia_startup_ai_radar.config import Settings, get_settings
from nvidia_startup_ai_radar.discovery import (
    build_theme_discovery_queries,
    candidates_as_dicts,
    discover_startups_for_theme,
)
from nvidia_startup_ai_radar.knowledge_base import similar_cases
from nvidia_startup_ai_radar.llm import build_chat_model, invoke_structured_with_retry, invoke_text
from nvidia_startup_ai_radar.prompts import PROMPTS
from nvidia_startup_ai_radar.rag import DEFAULT_RAG_DB_PATH, normalize_text, rag_search
from nvidia_startup_ai_radar.schemas import (
    AgentState,
    EconomicEstimate,
    Evidence,
    JudgeResult,
    RawPage,
    Recommendation,
    ScoreComponent,
    SignalEvidence,
    StartupProfile,
    utc_now_iso,
)
from nvidia_startup_ai_radar.scraping import fetch_public_page, is_probably_url


logger = logging.getLogger(__name__)

COMPETITOR_STACK_VARIANTS = {
    "AWS Bedrock": [
        "aws bedrock",
        "amazon bedrock",
        "bedrock",
    ],
    "Google Vertex AI": [
        "google vertex ai",
        "vertex ai",
        "google cloud vertex",
        "vertexai",
    ],
    "Azure OpenAI": [
        "azure openai",
        "azure open ai",
        "microsoft azure openai",
        "azure ai studio",
    ],
}


class ClassifierOutput(BaseModel):
    sinais_ai_native: list[SignalEvidence] = Field(default_factory=list)
    sinais_wrapper_risco: list[SignalEvidence] = Field(default_factory=list)
    score_maturidade_ia: float
    score_componentes: list[ScoreComponent] = Field(default_factory=list)
    score_wrapper_risco: float
    explicacao_classificacao: str
    classificacao: Literal["AI-native", "AI-enabled", "non-AI", "indeterminado"]


class EvidenceValidationOutput(BaseModel):
    sinais_ai_native: list[SignalEvidence] = Field(default_factory=list)
    sinais_wrapper_risco: list[SignalEvidence] = Field(default_factory=list)
    evidencias: list[Evidence] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    human_review_required: bool = False


class RecommendationsOutput(BaseModel):
    recomendacoes_nvidia: list[Recommendation] = Field(default_factory=list)


class BriefingOutput(BaseModel):
    briefing_pt: str


def _settings(state: AgentState) -> Settings:
    configured = get_settings()
    return Settings(
        llm_provider=configured.llm_provider,
        nvidia_api_key=configured.nvidia_api_key,
        nvidia_model=configured.nvidia_model,
        groq_api_key=configured.groq_api_key,
        groq_model=configured.groq_model,
        groq_base_url=configured.groq_base_url,
        anthropic_api_key=configured.anthropic_api_key,
        anthropic_model=configured.anthropic_model,
        openai_api_key=configured.openai_api_key,
        openai_model=configured.openai_model,
        output_language=state.get("output_language", configured.output_language),
        enable_web_fetch=configured.enable_web_fetch,
    )


def _provider_key_missing(settings: Settings) -> bool:
    provider = settings.llm_provider
    if provider in {"none", "offline", "disabled", ""}:
        return True
    if provider in {"nvidia_nim", "nvidia"}:
        return not bool(settings.nvidia_api_key)
    if provider == "groq":
        return not bool(settings.groq_api_key)
    if provider == "anthropic":
        return not bool(settings.anthropic_api_key)
    if provider == "openai":
        return not bool(settings.openai_api_key)
    return True


def _execution_update(
    state: AgentState,
    *,
    agent: str,
    mode: Literal["llm", "fallback_apos_falha", "fallback_sem_chave"],
    settings: Settings,
    reason: str,
) -> AgentState:
    modes = dict(state.get("agent_execution_modes", {}))
    modes[agent] = mode
    log = [
        *state.get("agent_execution_log", []),
        {
            "agent": agent,
            "modo_execucao": mode,
            "provider": settings.llm_provider,
            "motivo": reason,
            "timestamp": utc_now_iso(),
        },
    ]
    if mode == "llm":
        logger.info("%s executado com LLM provider=%s.", agent, settings.llm_provider)
    else:
        logger.warning("%s usando %s: %s", agent, mode, reason)
    return {"agent_execution_modes": modes, "agent_execution_log": log}


def _structured_llm_output(
    state: AgentState,
    *,
    agent: str,
    schema: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
) -> tuple[BaseModel | None, AgentState]:
    settings = _settings(state)
    if _provider_key_missing(settings):
        reason = f"LLM_PROVIDER={settings.llm_provider} sem chave configurada."
        return None, _execution_update(
            state,
            agent=agent,
            mode="fallback_sem_chave",
            settings=settings,
            reason=reason,
        )
    if state.get("llm_provider_unavailable"):
        return None, _execution_update(
            state,
            agent=agent,
            mode="fallback_apos_falha",
            settings=settings,
            reason=str(state.get("llm_provider_unavailable_reason") or "LLM indisponivel nesta execucao."),
        )

    llm = build_chat_model(settings)
    if llm is None:
        return None, _execution_update(
            state,
            agent=agent,
            mode="fallback_apos_falha",
            settings=settings,
            reason="Provider configurado, mas o modelo nao pode ser inicializado.",
        )

    result, errors = invoke_structured_with_retry(llm, schema, system_prompt, user_prompt, max_retries=2)
    if result is None:
        update = _execution_update(
            state,
            agent=agent,
            mode="fallback_apos_falha",
            settings=settings,
            reason="; ".join(errors) or "saida LLM invalida",
        )
        if any("rate_limit" in error.lower() or "429" in error for error in errors):
            update["llm_provider_unavailable"] = True
            update["llm_provider_unavailable_reason"] = "Groq/LLM retornou rate limit; usando fallback local nesta execucao."
        return None, update
    return result, _execution_update(
        state,
        agent=agent,
        mode="llm",
        settings=settings,
        reason="saida validada contra schema Pydantic",
    )


def _append_error(state: AgentState, message: str) -> list[str]:
    return [*state.get("errors", []), message]


def _json(data: Any) -> str:
    if isinstance(data, BaseException):
        return str(data)
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except TypeError:
        return str(data)


def _profile_from_state(state: AgentState) -> StartupProfile:
    profile = state.get("profile") or state.get("inbound_profile") or {}
    return StartupProfile.model_validate(profile)


def _all_text(raw_pages: list[dict[str, Any]]) -> str:
    return "\n\n".join(page.get("text", "") for page in raw_pages)


def _fallback_evidence_from_pages(raw_pages: list[dict[str, Any]]) -> list[Evidence]:
    return [
        Evidence(
            fonte_url=page.get("url", "local"),
            trecho_resumido=(page.get("text", "")[:320] or "Sem trecho textual disponivel."),
            data_coleta=page.get("collected_at", utc_now_iso()),
        )
        for page in raw_pages[:6]
    ]


def _detect_competitor_stack(raw_pages: list[dict[str, Any]]) -> tuple[list[str], list[SignalEvidence]]:
    detected: list[str] = []
    evidences: list[SignalEvidence] = []
    for page in raw_pages:
        text = page.get("text", "") or ""
        source = page.get("url", "local")
        lower = text.lower()
        for canonical, variants in COMPETITOR_STACK_VARIANTS.items():
            if canonical in detected:
                continue
            if any(variant in lower for variant in variants):
                detected.append(canonical)
                evidences.append(
                    _signal(
                        source,
                        f"stack concorrente detectada: {canonical}",
                        text,
                        variants,
                    )
                )
    return detected, evidences


def _urls_from_text(text: str) -> list[str]:
    return sorted({url.rstrip(".,;:") for url in re.findall(r"https?://[^\s\]\)>,]+", text)})


def search_planner_agent(state: AgentState) -> AgentState:
    inbound = state.get("inbound_profile")
    query = state.get("query") or ""
    mode = "inbound" if inbound else "outbound"
    source_text = _json(inbound) if inbound else query
    urls = _urls_from_text(source_text)
    if inbound and inbound.get("site"):
        urls.append(str(inbound["site"]))
    if inbound:
        planned = [
            source_text,
            f"{source_text} site oficial produto IA",
            f"{source_text} careers ML engineer GPU MLOps",
            f"{source_text} funding investidores",
            f"{source_text} NVIDIA GPU Triton TensorRT",
        ]
    else:
        planned = build_theme_discovery_queries(source_text)

    candidate_leads = []
    if state.get("discover_candidates") and source_text and not inbound:
        candidates = discover_startups_for_theme(
            source_text,
            limit=int(state.get("discovery_limit", 15)),
            results_per_query=int(state.get("discovery_results_per_query", 4)),
            fetch_pages=bool(state.get("discovery_fetch_pages", False)),
            delay_seconds=float(state.get("discovery_delay_seconds", 0.0)),
            search_workers=int(state.get("discovery_search_workers", 6)),
        )
        candidate_leads = candidates_as_dicts(candidates)
        urls.extend(candidate.url for candidate in candidates)

    return {
        "run_mode": mode,
        "planned_searches": list(dict.fromkeys(planned)),
        "urls": list(dict.fromkeys(urls)),
        "candidate_leads": candidate_leads,
    }


def scraper_agent(state: AgentState) -> AgentState:
    settings = _settings(state)
    raw_pages: list[RawPage] = []
    errors = state.get("errors", [])
    seed_text = state.get("query") or _json(state.get("inbound_profile", {}))
    for url in state.get("urls", []):
        if not is_probably_url(url):
            continue
        if not settings.enable_web_fetch:
            raw_pages.append(
                RawPage(
                    url=url,
                    title="web fetch disabled",
                    text=f"URL planejada para coleta futura: {url}",
                    scrape_method="disabled",
                    scrape_success=False,
                    failure_reason="RADAR_ENABLE_WEB_FETCH=false",
                )
            )
            continue
        try:
            page = fetch_public_page(url)
            raw_pages.append(page)
            if not page.scrape_success:
                errors.append(f"Falha ao coletar {url}: {page.failure_reason or 'fonte nao coletada'}")
        except Exception as exc:
            errors.append(f"Falha ao coletar {url}: {exc}")

    if seed_text and not any(page.url == "local://query" for page in raw_pages):
        raw_pages.insert(0, RawPage(url="local://query", title="Consulta do usuario", text=seed_text))

    return {"raw_pages": [page.model_dump() for page in raw_pages], "errors": errors}


def extractor_agent(state: AgentState) -> AgentState:
    inbound = state.get("inbound_profile") or {}
    raw_pages = state.get("raw_pages", [])
    text = _all_text(raw_pages)
    prompt = PROMPTS["extractor"]
    llm_profile, mode_update = _structured_llm_output(
        state,
        agent="extractor",
        schema=StartupProfile,
        system_prompt=(
            f"{prompt.system} Retorne somente JSON valido para o schema StartupProfile. "
            "Para cada sinal extraido, inclua evidencia_trecho e fonte_url. "
            "Nao invente fatos ausentes nas fontes."
        ),
        user_prompt=prompt.user_template.format(
            raw_pages=_json(raw_pages),
            inbound_profile=_json(inbound),
        ),
    )
    if isinstance(llm_profile, StartupProfile):
        llm_profile.origem = state.get("run_mode", llm_profile.origem)
        if not llm_profile.evidencias:
            llm_profile.evidencias = _fallback_evidence_from_pages(raw_pages)
        detected_competitors, competitor_evidences = _detect_competitor_stack(raw_pages)
        llm_profile.stack_concorrente_detectada = list(
            dict.fromkeys([*llm_profile.stack_concorrente_detectada, *detected_competitors])
        )
        if competitor_evidences:
            llm_profile.stack_concorrente_evidencias = [
                *llm_profile.stack_concorrente_evidencias,
                *competitor_evidences,
            ]
        llm_profile.ultima_atualizacao = utc_now_iso()
        return {"profile": llm_profile.model_dump(), **mode_update}

    lower = text.lower()

    name = inbound.get("nome") or inbound.get("name") or "Startup nao identificada"
    if name == "Startup nao identificada":
        candidate_name = re.search(
            r"(?:Startup candidata|Startup candidate|Company|Nome):\s*"
            r"([A-Z][A-Za-z0-9&.\- ]{1,80}?)(?=(?:\.\s+Fonte principal|\s+Fonte principal|\n|$))",
            text,
        )
        if candidate_name:
            name = candidate_name.group(1).strip(" .:-")
    if name == "Startup nao identificada":
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        name_match = re.search(
            r"\b([A-Z][A-Za-z0-9&.\- ]{2,40}?)(?:\s+usa|\s+aplica|\s+oferece|\s+combina|\s+e\s|\s+-|$)",
            first_line,
        )
        if name_match:
            name = name_match.group(1).strip()

    site = inbound.get("site") or (state.get("urls") or [None])[0]
    setor = inbound.get("setor") or inbound.get("sector")
    if not setor:
        if any(word in lower for word in ["saude", "health", "hospital", "clinico", "oncologia"]):
            setor = "Healthtech"
        elif any(word in lower for word in ["fintech", "credito", "pagamento", "banco", "pix"]):
            setor = "Fintech"
        elif any(word in lower for word in ["seguranca", "security", "camera", "video"]):
            setor = "Seguranca"
        elif any(word in lower for word in ["developer", "codigo", "debug", "dev tools"]):
            setor = "Dev tools"
        else:
            setor = "Nao identificado"

    stack_terms = [
        "NVIDIA",
        "GPU",
        "Triton",
        "TensorRT",
        "TensorRT-LLM",
        "CUDA",
        "RAPIDS",
        "Riva",
        "Clara",
        "NeMo",
        "NeMo Guardrails",
        "NVIDIA NIM",
        "NIM",
        "Jetson",
        "NVIDIA Orin",
        "Model serving",
        "Inference",
        "MLOps",
        "Kubernetes",
    ]
    detected_competitors, competitor_evidences = _detect_competitor_stack(raw_pages)
    stack = [term for term in stack_terms if term.lower() in lower]

    evidence = _fallback_evidence_from_pages(raw_pages)

    product_description = inbound.get("produto_descricao") or inbound.get("description")
    if not product_description:
        product_description = text[:500] if text else None

    profile = StartupProfile(
        id=inbound.get("id"),
        nome=name,
        site=site,
        ano_fundacao=inbound.get("ano_fundacao"),
        setor=setor,
        subsetor=inbound.get("subsetor"),
        estagio_funding=inbound.get("estagio_funding"),
        valor_captado_total=inbound.get("valor_captado_total"),
        investidores=inbound.get("investidores", []),
        headcount_estimado=inbound.get("headcount_estimado"),
        origem=state.get("run_mode", "outbound"),
        produto_descricao=product_description,
        publico_alvo=inbound.get("publico_alvo"),
        stack_tecnica_detectada=stack,
        stack_concorrente_detectada=detected_competitors,
        stack_concorrente_evidencias=competitor_evidences,
        evidencias=evidence,
    )
    return {"profile": profile.model_dump(), **mode_update}


def _evidence_snippet(text: str, keywords: list[str], fallback: str) -> str:
    if not text:
        return fallback
    lower = text.lower()
    positions = [lower.find(keyword.lower()) for keyword in keywords if keyword.lower() in lower]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text[:240]
    start = max(0, min(positions) - 90)
    end = min(len(text), min(positions) + 220)
    return text[start:end].strip()


def _signal(source: str, sinal: str, text: str, keywords: list[str]) -> SignalEvidence:
    snippet = _evidence_snippet(text, keywords, sinal)
    return SignalEvidence(sinal=sinal, evidencia_trecho=snippet, fonte_url=source)


def classifier_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    text = " ".join(
        [
            profile.nome,
            profile.setor or "",
            profile.produto_descricao or "",
            " ".join(profile.stack_tecnica_detectada),
            " ".join(profile.stack_concorrente_detectada),
            " ".join(ev.trecho_resumido for ev in profile.evidencias),
        ]
    )
    lower = text.lower()
    source = profile.evidencias[0].fonte_url if profile.evidencias else "local"
    prompt = PROMPTS["startup_classifier"]
    llm_classification, mode_update = _structured_llm_output(
        state,
        agent="startup_classifier",
        schema=ClassifierOutput,
        system_prompt=(
            f"{prompt.system} Retorne somente JSON valido. "
            "Use os sinais AI-native e wrapper do planejamento como rubrica. "
            "Cada score_componentes deve explicar pontos positivos, negativos ou neutros e citar evidencias."
        ),
        user_prompt=prompt.user_template.format(profile=_json(profile.model_dump())),
    )
    if isinstance(llm_classification, ClassifierOutput):
        profile.sinais_ai_native = llm_classification.sinais_ai_native
        profile.sinais_wrapper_risco = llm_classification.sinais_wrapper_risco
        profile.score_maturidade_ia = max(0.0, min(100.0, float(llm_classification.score_maturidade_ia)))
        profile.score_componentes = llm_classification.score_componentes
        profile.score_wrapper_risco = max(0.0, min(100.0, float(llm_classification.score_wrapper_risco)))
        profile.explicacao_classificacao = llm_classification.explicacao_classificacao
        profile.classificacao = llm_classification.classificacao
        profile.ultima_atualizacao = utc_now_iso()
        return {"profile": profile.model_dump(), **mode_update}

    native_rules = {
        "modelo/dataset proprietario": {
            "keywords": [
                "modelo proprietario",
                "dataset proprietario",
                "proprietary dataset",
                "dataset",
                "fine-tuning",
                "fine tuning",
                "treinamos",
                "training",
                "model training",
                "foundation model",
            ],
            "points": 22,
            "why": "Indica diferenciacao tecnica defensavel, nao apenas interface sobre modelo de terceiro.",
        },
        "infraestrutura GPU ou self-hosted": {
            "keywords": [
                "gpu",
                "self-hosted",
                "triton",
                "tensorrt",
                "cuda",
                "nvidia",
                "inference",
                "model inference",
                "model serving",
                "latency",
                "throughput",
                "edge hardware",
                "jetson",
                "nvidia orin",
                "nim",
            ],
            "points": 18,
            "why": "Sinal de maturidade operacional para inferencia, custo e latencia.",
        },
        "intencao de contratacao tecnica": {
            "keywords": [
                "vaga",
                "hiring",
                "contratando",
                "ml engineer",
                "mlops",
                "data engineer",
                "platform engineer",
                "inference engineer",
                "applied ai engineer",
                "machine learning engineer",
                "research scientist",
            ],
            "points": 14,
            "why": "Vagas tecnicas sugerem investimento continuo em capacidade interna de IA.",
        },
        "setor regulado": {
            "keywords": [
                "lgpd",
                "bacen",
                "anvisa",
                "hipaa",
                "healthtech",
                "fintech",
                "saude",
                "credito",
                "financial",
                "finance",
                "clinical",
                "healthcare",
            ],
            "points": 10,
            "why": "Setores regulados tendem a exigir governanca, dados proprietarios e integracao real.",
        },
        "integracao profunda": {
            "keywords": ["erp", "crm", "ehr", "core bancario", "prontuario", "enterprise", "production"],
            "points": 14,
            "why": "Integracao com sistema critico e mais dificil de copiar que uma UI generica.",
        },
        "automacao de processo": {
            "keywords": ["automacao", "workflow", "processo", "operacoes", "workflow automation", "ai agent", "agentic"],
            "points": 10,
            "why": "Automacao de trabalho/processo aponta para IA no produto, nao so assistente lateral.",
        },
        "P&D ou parceria academica": {
            "keywords": ["p&d", "universidade", "universidades", "pesquisa"],
            "points": 14,
            "why": "P&D e parceria academica reforcam profundidade tecnica e barreira de entrada.",
        },
        "IA aplicada ao produto central": {
            "keywords": [
                "visao computacional",
                "computer vision",
                "deteccao",
                "predicao",
                "monitoramento",
                "generative ai",
                "llm",
                "machine learning",
                "deep learning",
                "artificial intelligence",
                "images and videos",
                "image",
                "video",
            ],
            "points": 12,
            "why": "A IA aparece como mecanismo central de entrega de valor.",
        },
    }
    wrapper_rules = {
        "diferencial baseado em GPT sem camada propria": {
            "keywords": ["powered by gpt", "chatgpt", "gpt-4", "wrapper"],
            "points": -22,
            "why": "Diferencial comunicado parece depender do modelo-base, sem camada propria evidente.",
        },
        "produto facilmente replicavel": {
            "keywords": ["chat com pdf", "gerador de post", "resumo de documentos"],
            "points": -18,
            "why": "Categoria vulneravel a virar recurso nativo de provedores de modelo.",
        },
        "dependencia de API externa": {
            "keywords": [
                "openai api",
                "anthropic api",
                "api externa",
                "unica api",
                "aws bedrock",
                "amazon bedrock",
                "vertex ai",
                "azure openai",
            ],
            "points": -16,
            "why": "Dependencia de uma unica API reduz defensibilidade e poder de negociacao.",
        },
        "pivots frequentes": {
            "keywords": ["pivot", "mudou de produto", "novo posicionamento"],
            "points": -12,
            "why": "Mudancas frequentes sem mercado travado aumentam risco de wrapper fino.",
        },
        "ausencia de contratacao tecnica": {
            "keywords": ["sem vagas tecnicas", "apenas vendas", "somente growth", "so marketing"],
            "points": -10,
            "why": "Ausencia explicita de contratacao tecnica enfraquece tese de capacidade interna.",
        },
    }

    native_signals: list[SignalEvidence] = []
    wrapper_signals: list[SignalEvidence] = []
    score_components: list[ScoreComponent] = []
    positive_score = 0.0
    wrapper_risk_score = 0.0
    for label, rule in native_rules.items():
        keywords = rule["keywords"]
        if any(keyword in lower for keyword in keywords):
            signal = _signal(source, label, text, keywords)
            native_signals.append(signal)
            positive_score += float(rule["points"])
            score_components.append(
                ScoreComponent(
                    componente=label,
                    tipo="positivo",
                    pontos=float(rule["points"]),
                    justificativa=rule["why"],
                    evidencias=[signal],
                )
            )
    for label, rule in wrapper_rules.items():
        keywords = rule["keywords"]
        if any(keyword in lower for keyword in keywords):
            signal = _signal(source, label, text, keywords)
            wrapper_signals.append(signal)
            wrapper_risk_score += abs(float(rule["points"]))
            score_components.append(
                ScoreComponent(
                    componente=label,
                    tipo="negativo",
                    pontos=float(rule["points"]),
                    justificativa=rule["why"],
                    evidencias=[signal],
                )
            )

    if profile.stack_tecnica_detectada:
        stack_signal = SignalEvidence(
            sinal="stack tecnica detectada",
            evidencia_trecho=", ".join(profile.stack_tecnica_detectada),
            fonte_url=source,
        )
        positive_score += 8
        score_components.append(
            ScoreComponent(
                componente="bonus por stack tecnica detectada",
                tipo="positivo",
                pontos=8,
                justificativa="Tecnologias tecnicas citadas ajudam a diferenciar maturidade real de mensagem comercial.",
                evidencias=[stack_signal],
            )
        )

    score = max(0.0, min(100.0, positive_score - wrapper_risk_score))
    if score >= 60 and len(native_signals) >= 3 and wrapper_risk_score < 35:
        classification = "AI-native"
        explanation = "Ha varios sinais tecnicos fortes e risco wrapper controlado."
    elif wrapper_risk_score >= 35 and positive_score < 35:
        classification = "non-AI"
        explanation = "Predominam sinais de wrapper/API externa sem evidencias tecnicas suficientes."
    elif score >= 30 or native_signals:
        classification = "AI-enabled"
        explanation = "Ha sinais de uso de IA, mas ainda faltam evidencias para cravar AI-native."
    else:
        classification = "indeterminado"
        explanation = "As evidencias publicas sao insuficientes para uma classificacao confiavel."

    profile.sinais_ai_native = native_signals
    profile.sinais_wrapper_risco = wrapper_signals
    profile.score_maturidade_ia = float(score)
    profile.score_componentes = score_components
    profile.score_wrapper_risco = float(wrapper_risk_score)
    profile.explicacao_classificacao = explanation
    profile.classificacao = classification
    profile.ultima_atualizacao = utc_now_iso()
    return {"profile": profile.model_dump(), **mode_update}


def evidence_validator_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    errors = state.get("errors", [])
    prompt = PROMPTS["evidence_validator"]
    llm_validation, mode_update = _structured_llm_output(
        state,
        agent="evidence_validator",
        schema=EvidenceValidationOutput,
        system_prompt=(
            f"{prompt.system} Retorne somente JSON valido. "
            "Mantenha apenas sinais com fonte e trecho rastreavel."
        ),
        user_prompt=prompt.user_template.format(profile=_json(profile.model_dump())),
    )
    if isinstance(llm_validation, EvidenceValidationOutput):
        profile.sinais_ai_native = llm_validation.sinais_ai_native
        profile.sinais_wrapper_risco = llm_validation.sinais_wrapper_risco
        profile.evidencias = llm_validation.evidencias or profile.evidencias
        errors = [*errors, *llm_validation.errors]
        return {
            "profile": profile.model_dump(),
            "errors": errors,
            "human_review_required": llm_validation.human_review_required
            or bool(errors)
            or profile.classificacao == "indeterminado",
            **mode_update,
        }

    def valid(signal: SignalEvidence) -> bool:
        return bool(signal.fonte_url and signal.evidencia_trecho)

    before = len(profile.sinais_ai_native) + len(profile.sinais_wrapper_risco)
    profile.sinais_ai_native = [signal for signal in profile.sinais_ai_native if valid(signal)]
    profile.sinais_wrapper_risco = [signal for signal in profile.sinais_wrapper_risco if valid(signal)]
    after = len(profile.sinais_ai_native) + len(profile.sinais_wrapper_risco)
    if after < before:
        errors.append("Evidence Validator removeu sinais sem fonte ou trecho.")
    if not profile.evidencias:
        errors.append("Perfil sem evidencias rastreaveis; revisao humana recomendada.")
    return {
        "profile": profile.model_dump(),
        "errors": errors,
        "human_review_required": bool(errors) or profile.classificacao == "indeterminado",
        **mode_update,
    }


def nvidia_rag_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    query = " ".join(
        [
            profile.setor or "",
            profile.classificacao,
            profile.produto_descricao or "",
            " ".join(signal.sinal for signal in profile.sinais_ai_native),
            " ".join(signal.sinal for signal in profile.sinais_wrapper_risco),
            " ".join(profile.stack_tecnica_detectada),
            " ".join(profile.stack_concorrente_detectada),
        ]
    )
    entries = rag_search(query, db_path=state.get("rag_db_path", str(DEFAULT_RAG_DB_PATH)), limit=8)
    return {"retrieved_entries": entries}


def recommendation_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    retrieved = state.get("retrieved_entries", [])
    evidence = profile.evidencias[:2]
    eligible_entries = [
        entry for entry in retrieved if entry.get("document_type") in {"knowledge_entry", "official_nvidia_doc"}
    ]
    allowed_technologies = [str(entry.get("tecnologia") or entry.get("title")) for entry in eligible_entries]
    allowed_norm = {technology.lower().strip() for technology in allowed_technologies if technology}
    prompt = PROMPTS["recommendation"]
    llm_recommendations, mode_update = _structured_llm_output(
        state,
        agent="recommendation",
        schema=RecommendationsOutput,
        system_prompt=(
            f"{prompt.system} Retorne somente JSON valido. "
            "Voce so pode recomendar tecnologias presentes em Conhecimento recuperado; "
            "nao invente tecnologia fora dos chunks."
        ),
        user_prompt=prompt.user_template.format(
            profile=_json(profile.model_dump()),
            retrieved_entries=_json(eligible_entries),
        ),
    )

    def allowed_by_rag(technology: str) -> bool:
        normalized = technology.lower().strip()
        return any(normalized == allowed or normalized in allowed or allowed in normalized for allowed in allowed_norm)

    if isinstance(llm_recommendations, RecommendationsOutput):
        invalid = [
            recommendation.tecnologia
            for recommendation in llm_recommendations.recomendacoes_nvidia
            if not allowed_by_rag(recommendation.tecnologia)
        ]
        if not invalid:
            profile.recomendacoes_nvidia = llm_recommendations.recomendacoes_nvidia[:5]
            return {"profile": profile.model_dump(), **mode_update}
        settings = _settings(state)
        mode_update = _execution_update(
            state,
            agent="recommendation",
            mode="fallback_apos_falha",
            settings=settings,
            reason=f"LLM recomendou tecnologias fora do RAG: {', '.join(invalid)}",
        )

    recommendations: list[Recommendation] = []
    seen: set[str] = set()

    for entry in eligible_entries:
        tecnologia = entry["tecnologia"]
        if tecnologia in seen:
            continue
        seen.add(tecnologia)
        priority = "alta" if profile.classificacao == "AI-native" else "media"
        if tecnologia == "NVIDIA Inception":
            priority = "alta" if profile.classificacao in {"AI-native", "AI-enabled"} else "media"
        if profile.sinais_wrapper_risco and tecnologia not in {"NVIDIA NIM", "NeMo Guardrails", "NVIDIA Inception"}:
            priority = "baixa"
        approach = "migracao/complemento competitivo" if profile.stack_concorrente_detectada else "adocao nativa"
        rag_evidence = Evidence(
            fonte_url=entry.get("fonte_url") or entry.get("source_url") or "local://rag",
            trecho_resumido=(entry.get("chunk_text") or entry.get("text") or "")[:320],
        )
        recommendations.append(
            Recommendation(
                tecnologia=tecnologia,
                justificativa_tecnica=entry["problema_que_resolve"],
                justificativa_negocio=entry["descricao_negocio"],
                prioridade=priority,
                complexidade=entry["complexidade_implementacao"],
                proxima_acao=(
                    f"Rodar discovery de {approach} com foco em {tecnologia}; "
                    "validar evidencias tecnicas antes de proposta comercial."
                ),
                evidencias=[*evidence, rag_evidence],
            )
        )

    has_inception_in_rag = any(str(entry.get("tecnologia")) == "NVIDIA Inception" for entry in eligible_entries)
    if has_inception_in_rag and not any(rec.tecnologia == "NVIDIA Inception" for rec in recommendations):
        recommendations.append(
            Recommendation(
                tecnologia="NVIDIA Inception",
                justificativa_tecnica="Entrada no ecossistema NVIDIA para validar fit tecnico e acesso a especialistas.",
                justificativa_negocio="Proxima acao de baixo atrito para startups com potencial de IA.",
                prioridade="media",
                complexidade="baixa",
                proxima_acao="Convidar para diagnostico Inception e coletar requisitos de infraestrutura.",
                evidencias=evidence,
            )
        )

    profile.recomendacoes_nvidia = recommendations[:5]
    return {"profile": profile.model_dump(), **mode_update}


def economic_estimator_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    if any(rec.tecnologia in {"NVIDIA NIM", "TensorRT-LLM", "Triton Inference Server"} for rec in profile.recomendacoes_nvidia):
        profile.estimativa_economica = EconomicEstimate(
            cenario_assumido="Startup possui workload de inferencia ou pretende levar IA para producao.",
            metodologia=(
                "Medir baseline atual e comparar com NIM/Triton/TensorRT-LLM usando GenAI-Perf: "
                "TTFT, throughput, custo por token, taxa de erro e utilizacao de GPU."
            ),
            economia_estimada_percentual=None,
            fonte_benchmark="NVIDIA GenAI-Perf e serie NVIDIA LLM Inference Benchmarking",
            nivel_confianca="projecao_com_premissas",
        )
    else:
        profile.estimativa_economica = EconomicEstimate()
    return {"profile": profile.model_dump()}


def judge_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    profile_text = _json(profile.model_dump())
    judge_query = " ".join(
        [
            profile_text,
            " ".join(recommendation.tecnologia for recommendation in profile.recomendacoes_nvidia),
            "golden set case bank sucesso fracasso wrapper AI-native",
        ]
    )
    rag_cases = rag_search(
        judge_query,
        db_path=state.get("rag_db_path", str(DEFAULT_RAG_DB_PATH)),
        limit=8,
    )
    semantic_cases = [
        result
        for result in rag_cases
        if result.get("document_type") in {"historical_case", "golden_case", "conceptual_source"}
    ]
    if semantic_cases:
        profile.casos_similares = [
            {
                "case_id": result.get("tecnologia") or result.get("title"),
                "tipo": result.get("document_type"),
                "similaridade": round(float(result.get("rerank_score", 0.0)), 3),
                "licao": result.get("descricao_negocio") or result.get("problema_que_resolve"),
                "fonte_url": result.get("fonte_url"),
            }
            for result in semantic_cases[:4]
        ]
    else:
        cases = similar_cases(profile_text, limit=4)
        profile.casos_similares = [
            {
                "case_id": case.empresa,
                "tipo": case.tipo,
                "similaridade": 0.7,
                "licao": case.licao_estruturada,
            }
            for case in cases
        ]

    prompt = PROMPTS["llm_as_judge"]
    llm_judge, mode_update = _structured_llm_output(
        state,
        agent="llm_as_judge",
        schema=JudgeResult,
        system_prompt=(
            f"{prompt.system} Retorne somente JSON valido para JudgeResult. "
            "Compare recomendacoes contra os chunks semanticos do golden set, case bank e fontes conceituais."
        ),
        user_prompt=prompt.user_template.format(
            profile=_json(profile.model_dump()),
            similar_cases=_json(semantic_cases or profile.casos_similares),
        ),
    )
    if isinstance(llm_judge, JudgeResult):
        return {
            "profile": profile.model_dump(),
            "judge": llm_judge.model_dump(),
            "human_review_required": state.get("human_review_required", False) or llm_judge.status != "aprovado",
            **mode_update,
        }

    motives: list[str] = []
    confidence = 0.72
    if profile.sinais_wrapper_risco and profile.classificacao == "AI-native":
        motives.append("Classificacao AI-native coexistindo com sinais wrapper; revisar.")
        confidence -= 0.25
    if profile.classificacao == "indeterminado":
        motives.append("Evidencia insuficiente para recomendacao automatica.")
        confidence -= 0.3
    similar_failure = any(
        str(case.get("tipo") or case.get("document_type")).lower() in {"fracasso", "alerta", "historical_case"}
        and any(word in normalize for word in ["fracasso", "alerta", "wrapper", "falhou"])
        for case in profile.casos_similares
        for normalize in [normalize_text(_json(case))]
    )
    if similar_failure and profile.sinais_wrapper_risco:
        motives.append("Perfil parecido com casos historicos de alerta/fracasso.")
        confidence -= 0.2

    status = "aprovado" if confidence >= 0.6 and not motives else "revisao_humana"
    judge = JudgeResult(status=status, confianca=max(0.0, confidence), motivos=motives)
    return {
        "profile": profile.model_dump(),
        "judge": judge.model_dump(),
        "human_review_required": state.get("human_review_required", False) or status != "aprovado",
        **mode_update,
    }


def briefing_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    judge = JudgeResult.model_validate(state.get("judge", {}))
    prompt = PROMPTS["briefing"]
    llm_briefing, mode_update = _structured_llm_output(
        state,
        agent="briefing",
        schema=BriefingOutput,
        system_prompt=(
            f"{prompt.system} Retorne somente JSON valido no formato {{\"briefing_pt\": \"...markdown...\"}}. "
            "Se stack_concorrente_detectada nao estiver vazia, a proxima acao deve indicar "
            "abordagem de substituicao/migracao ou otimizacao competitiva. Se estiver vazia, "
            "indique adocao nativa/complemento NVIDIA."
        ),
        user_prompt=prompt.user_template.format(profile=_json(profile.model_dump()), judge=_json(judge.model_dump())),
    )
    if isinstance(llm_briefing, BriefingOutput) and llm_briefing.briefing_pt.strip():
        return {"briefing_pt": llm_briefing.briefing_pt, **mode_update}

    rec_lines = "\n".join(
        [
            f"- {rec.tecnologia}: prioridade {rec.prioridade}, complexidade {rec.complexidade}. "
            f"{rec.proxima_acao}"
            for rec in profile.recomendacoes_nvidia
        ]
    )
    native_lines = "\n".join(f"- {signal.sinal}: {signal.evidencia_trecho[:180]}" for signal in profile.sinais_ai_native) or "- Sem sinais fortes validados."
    wrapper_lines = "\n".join(f"- {signal.sinal}: {signal.evidencia_trecho[:180]}" for signal in profile.sinais_wrapper_risco) or "- Nenhum sinal critico validado."
    score_lines = "\n".join(
        f"- {component.pontos:+.0f} {component.componente}: {component.justificativa}"
        for component in profile.score_componentes
    ) or "- Sem componentes suficientes para pontuar."
    cases = "\n".join(
        f"- {case['case_id']} ({case['tipo']}): {case['licao']}" for case in profile.casos_similares
    ) or "- Sem caso similar forte."
    review = "Sim" if state.get("human_review_required") else "Nao"
    competitor_stack = ", ".join(profile.stack_concorrente_detectada) or "Nao detectada"
    if profile.stack_concorrente_detectada:
        next_action = (
            "Abordagem recomendada: substituicao/migracao ou otimizacao competitiva da stack "
            f"detectada ({', '.join(profile.stack_concorrente_detectada)}). Priorizar uma conversa "
            "tecnica curta para comparar custo, latencia, governanca e caminho de migracao para NVIDIA."
        )
    else:
        next_action = (
            "Abordagem recomendada: adocao nativa/complemento NVIDIA. Priorizar uma conversa tecnica "
            "curta para validar stack, volume de inferencia, restricoes de compliance e abertura para Inception."
        )
    briefing = f"""# Briefing NVIDIA Startup AI Radar: {profile.nome}

## Diagnostico
- Classificacao: {profile.classificacao}
- Score de maturidade de IA: {profile.score_maturidade_ia:.0f}/100
- Score de risco wrapper: {profile.score_wrapper_risco:.0f}/100
- Setor: {profile.setor or "Nao identificado"}
- Stack concorrente detectada: {competitor_stack}
- Revisao humana antes do envio: {review}
- Confianca do judge: {judge.confianca:.2f}
- Explicacao: {profile.explicacao_classificacao or "Nao informada."}

## Componentes do score
{score_lines}

## Evidencias AI-native
{native_lines}

## Riscos de wrapper
{wrapper_lines}

## Casos comparaveis
{cases}

## Recomendacoes NVIDIA
{rec_lines}

## Estimativa economica
{profile.estimativa_economica.metodologia}
Nivel de confianca: {profile.estimativa_economica.nivel_confianca}.

## Proxima acao
{next_action}
"""
    return {"briefing_pt": briefing, **mode_update}


def translation_agent(state: AgentState) -> AgentState:
    settings = _settings(state)
    prompt = PROMPTS["technical_translation"]
    briefing_pt = state.get("briefing_pt", "")
    llm_text = invoke_text(
        build_chat_model(settings),
        prompt.system,
        prompt.user_template.format(briefing_pt=briefing_pt),
    )
    if llm_text:
        return {"briefing_en": llm_text}
    return {
        "briefing_en": (
            "# English briefing\n\n"
            "Automatic offline translation is disabled. Set an LLM API key to generate "
            "the English technical version from the canonical pt-BR briefing.\n\n"
            + briefing_pt
        )
    }
