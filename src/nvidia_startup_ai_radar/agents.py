"""LangGraph node implementations for the NVIDIA Startup AI Radar."""

from __future__ import annotations

import json
import re
from typing import Any

from nvidia_startup_ai_radar.config import Settings, get_settings
from nvidia_startup_ai_radar.knowledge_base import retrieve_knowledge, similar_cases
from nvidia_startup_ai_radar.llm import build_chat_model, invoke_text
from nvidia_startup_ai_radar.prompts import PROMPTS
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


def _settings(state: AgentState) -> Settings:
    configured = get_settings()
    return Settings(
        nvidia_api_key=configured.nvidia_api_key,
        nvidia_model=configured.nvidia_model,
        openai_api_key=configured.openai_api_key,
        openai_model=configured.openai_model,
        output_language=state.get("output_language", configured.output_language),
        enable_web_fetch=configured.enable_web_fetch,
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


def _urls_from_text(text: str) -> list[str]:
    return sorted(set(re.findall(r"https?://[^\s\]\)>,]+", text)))


def search_planner_agent(state: AgentState) -> AgentState:
    inbound = state.get("inbound_profile")
    query = state.get("query") or ""
    mode = "inbound" if inbound else "outbound"
    source_text = _json(inbound) if inbound else query
    urls = _urls_from_text(source_text)
    if inbound and inbound.get("site"):
        urls.append(str(inbound["site"]))
    planned = [
        source_text,
        f"{source_text} site oficial produto IA",
        f"{source_text} careers ML engineer GPU MLOps",
        f"{source_text} funding investidores",
        f"{source_text} NVIDIA GPU Triton TensorRT",
    ]
    return {
        "run_mode": mode,
        "planned_searches": list(dict.fromkeys(planned)),
        "urls": list(dict.fromkeys(urls)),
    }


def scraper_agent(state: AgentState) -> AgentState:
    settings = _settings(state)
    raw_pages: list[RawPage] = []
    errors = state.get("errors", [])
    for url in state.get("urls", []):
        if not is_probably_url(url):
            continue
        if not settings.enable_web_fetch:
            raw_pages.append(
                RawPage(
                    url=url,
                    title="web fetch disabled",
                    text=f"URL planejada para coleta futura: {url}",
                )
            )
            continue
        try:
            raw_pages.append(fetch_public_page(url))
        except Exception as exc:
            errors.append(f"Falha ao coletar {url}: {exc}")

    if not raw_pages:
        seed_text = state.get("query") or _json(state.get("inbound_profile", {}))
        raw_pages.append(RawPage(url="local://query", title="Consulta do usuario", text=seed_text))

    return {"raw_pages": [page.model_dump() for page in raw_pages], "errors": errors}


def extractor_agent(state: AgentState) -> AgentState:
    inbound = state.get("inbound_profile") or {}
    raw_pages = state.get("raw_pages", [])
    text = _all_text(raw_pages)
    lower = text.lower()

    name = inbound.get("nome") or inbound.get("name") or "Startup nao identificada"
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
        "MLOps",
        "Kubernetes",
    ]
    competitor_terms = ["Bedrock", "Vertex AI", "Azure OpenAI", "OpenAI", "Anthropic", "Gemini"]
    stack = [term for term in stack_terms if term.lower() in lower]
    competitors = [term for term in competitor_terms if term.lower() in lower]

    evidence = [
        Evidence(
            fonte_url=page.get("url", "local"),
            trecho_resumido=(page.get("text", "")[:320] or "Sem trecho textual disponivel."),
            data_coleta=page.get("collected_at", utc_now_iso()),
        )
        for page in raw_pages[:6]
    ]

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
        stack_concorrente_detectada=competitors,
        evidencias=evidence,
    )
    return {"profile": profile.model_dump()}


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
    native_rules = {
        "modelo/dataset proprietario": {
            "keywords": ["modelo proprietario", "dataset proprietario", "fine-tuning", "treinamos"],
            "points": 22,
            "why": "Indica diferenciacao tecnica defensavel, nao apenas interface sobre modelo de terceiro.",
        },
        "infraestrutura GPU ou self-hosted": {
            "keywords": ["gpu", "self-hosted", "triton", "tensorrt", "cuda", "nvidia"],
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
            ],
            "points": 14,
            "why": "Vagas tecnicas sugerem investimento continuo em capacidade interna de IA.",
        },
        "setor regulado": {
            "keywords": ["lgpd", "bacen", "anvisa", "hipaa", "healthtech", "fintech", "saude", "credito"],
            "points": 10,
            "why": "Setores regulados tendem a exigir governanca, dados proprietarios e integracao real.",
        },
        "integracao profunda": {
            "keywords": ["erp", "crm", "ehr", "core bancario", "prontuario"],
            "points": 14,
            "why": "Integracao com sistema critico e mais dificil de copiar que uma UI generica.",
        },
        "automacao de processo": {
            "keywords": ["automacao", "workflow", "processo", "operacoes"],
            "points": 10,
            "why": "Automacao de trabalho/processo aponta para IA no produto, nao so assistente lateral.",
        },
        "P&D ou parceria academica": {
            "keywords": ["p&d", "universidade", "universidades", "pesquisa"],
            "points": 14,
            "why": "P&D e parceria academica reforcam profundidade tecnica e barreira de entrada.",
        },
        "IA aplicada ao produto central": {
            "keywords": ["visao computacional", "computer vision", "deteccao", "predicao", "monitoramento"],
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
            "keywords": ["openai api", "anthropic api", "api externa", "unica api"],
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
    return {"profile": profile.model_dump()}


def evidence_validator_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    errors = state.get("errors", [])

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
    entries = retrieve_knowledge(query, limit=7)
    return {"retrieved_entries": [entry.model_dump() for entry in entries]}


def recommendation_agent(state: AgentState) -> AgentState:
    profile = _profile_from_state(state)
    retrieved = state.get("retrieved_entries", [])
    evidence = profile.evidencias[:2]
    recommendations: list[Recommendation] = []
    seen: set[str] = set()

    for entry in retrieved:
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
                evidencias=evidence,
            )
        )

    if not any(rec.tecnologia == "NVIDIA Inception" for rec in recommendations):
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
    return {"profile": profile.model_dump()}


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
    motives: list[str] = []
    confidence = 0.72
    if profile.sinais_wrapper_risco and profile.classificacao == "AI-native":
        motives.append("Classificacao AI-native coexistindo com sinais wrapper; revisar.")
        confidence -= 0.25
    if profile.classificacao == "indeterminado":
        motives.append("Evidencia insuficiente para recomendacao automatica.")
        confidence -= 0.3
    if any(case.tipo in {"fracasso", "alerta"} for case in cases) and profile.sinais_wrapper_risco:
        motives.append("Perfil parecido com casos historicos de alerta/fracasso.")
        confidence -= 0.2

    status = "aprovado" if confidence >= 0.6 and not motives else "revisao_humana"
    judge = JudgeResult(status=status, confianca=max(0.0, confidence), motivos=motives)
    return {
        "profile": profile.model_dump(),
        "judge": judge.model_dump(),
        "human_review_required": state.get("human_review_required", False) or status != "aprovado",
    }


def briefing_agent(state: AgentState) -> AgentState:
    settings = _settings(state)
    profile = _profile_from_state(state)
    judge = JudgeResult.model_validate(state.get("judge", {}))
    prompt = PROMPTS["briefing"]
    llm_text = invoke_text(
        build_chat_model(settings),
        prompt.system,
        prompt.user_template.format(profile=_json(profile.model_dump()), judge=_json(judge.model_dump())),
    )
    if llm_text:
        return {"briefing_pt": llm_text}

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
    briefing = f"""# Briefing NVIDIA Startup AI Radar: {profile.nome}

## Diagnostico
- Classificacao: {profile.classificacao}
- Score de maturidade de IA: {profile.score_maturidade_ia:.0f}/100
- Score de risco wrapper: {profile.score_wrapper_risco:.0f}/100
- Setor: {profile.setor or "Nao identificado"}
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
Priorizar uma conversa tecnica curta para validar stack, volume de inferencia, restricoes de compliance e abertura para Inception.
"""
    return {"briefing_pt": briefing}


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
