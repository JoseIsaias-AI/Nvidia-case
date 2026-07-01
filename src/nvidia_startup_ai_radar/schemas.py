"""Shared data contracts used by all LangGraph nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, HttpUrl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Evidence(BaseModel):
    fonte_url: str = "local"
    trecho_resumido: str
    data_coleta: str = Field(default_factory=utc_now_iso)


class SignalEvidence(BaseModel):
    sinal: str
    evidencia_trecho: str
    fonte_url: str = "local"
    data_coleta: str = Field(default_factory=utc_now_iso)


class ScoreComponent(BaseModel):
    componente: str
    tipo: Literal["positivo", "negativo", "neutro"]
    pontos: float
    justificativa: str
    evidencias: list[SignalEvidence] = Field(default_factory=list)


class Recommendation(BaseModel):
    tecnologia: str
    justificativa_tecnica: str
    justificativa_negocio: str
    prioridade: Literal["alta", "media", "baixa"]
    complexidade: Literal["baixa", "media", "alta"]
    proxima_acao: str
    evidencias: list[Evidence] = Field(default_factory=list)


class EconomicEstimate(BaseModel):
    cenario_assumido: str = "Nao informado"
    metodologia: str = "Sem benchmark medido; recomendacao exige validacao com GenAI-Perf."
    economia_estimada_percentual: float | None = None
    fonte_benchmark: str = "NVIDIA GenAI-Perf / Technical Blog"
    nivel_confianca: Literal[
        "medido",
        "estimado_por_metodologia_publica",
        "projecao_com_premissas",
        "indisponivel",
    ] = "indisponivel"


class StartupProfile(BaseModel):
    id: str | None = None
    nome: str = "Startup nao identificada"
    site: str | None = None
    ano_fundacao: int | None = None
    setor: str | None = None
    subsetor: str | None = None
    estagio_funding: str | None = None
    valor_captado_total: str | None = None
    investidores: list[str] = Field(default_factory=list)
    headcount_estimado: int | None = None
    origem: Literal["outbound", "inbound"] = "outbound"
    produto_descricao: str | None = None
    publico_alvo: str | None = None
    sinais_ai_native: list[SignalEvidence] = Field(default_factory=list)
    sinais_wrapper_risco: list[SignalEvidence] = Field(default_factory=list)
    stack_tecnica_detectada: list[str] = Field(default_factory=list)
    stack_concorrente_detectada: list[str] = Field(default_factory=list)
    stack_concorrente_evidencias: list[SignalEvidence] = Field(default_factory=list)
    score_maturidade_ia: float = 0.0
    score_componentes: list[ScoreComponent] = Field(default_factory=list)
    score_wrapper_risco: float = 0.0
    explicacao_classificacao: str | None = None
    classificacao: Literal["AI-native", "AI-enabled", "non-AI", "indeterminado"] = (
        "indeterminado"
    )
    evidencias: list[Evidence] = Field(default_factory=list)
    casos_similares: list[dict[str, Any]] = Field(default_factory=list)
    recomendacoes_nvidia: list[Recommendation] = Field(default_factory=list)
    estimativa_economica: EconomicEstimate = Field(default_factory=EconomicEstimate)
    ultima_atualizacao: str = Field(default_factory=utc_now_iso)


class RawPage(BaseModel):
    url: str = "local"
    title: str | None = None
    text: str
    collected_at: str = Field(default_factory=utc_now_iso)
    scrape_method: str = "local"
    scrape_success: bool = True
    served_from_cache: bool = False
    failure_reason: str | None = None
    robots_allowed: bool | None = None
    cache_key: str | None = None


class KnowledgeEntry(BaseModel):
    id: str
    tecnologia: str
    categoria: str
    problema_que_resolve: str
    descricao_tecnica: str
    descricao_negocio: str
    complexidade_implementacao: Literal["baixa", "media", "alta"]
    sinais_de_gatilho: list[str] = Field(default_factory=list)
    casos_de_uso_tipicos: list[str] = Field(default_factory=list)
    fonte_url: str | HttpUrl
    data_ultima_verificacao: str = "2026-06-23"


class HistoricalCase(BaseModel):
    empresa: str
    tipo: Literal["sucesso", "fracasso", "pivot", "alerta"]
    setor: str
    ano: int | None = None
    data_moat: bool = False
    infra_propria: bool = False
    dependencia_api_externa: bool = False
    setor_regulado: bool = False
    resumo_o_que_aconteceu: str
    licao_estruturada: str
    fonte_url: str = "docs/guia-completo-do-case.md"


class JudgeResult(BaseModel):
    status: Literal["aprovado", "revisao_humana", "bloqueado"] = "aprovado"
    confianca: float = 0.7
    motivos: list[str] = Field(default_factory=list)
    divergencias_golden_set: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    execution_id: str
    run_mode: Literal["outbound", "inbound"]
    query: str
    output_language: Literal["pt", "en", "both"]
    inbound_profile: dict[str, Any]
    discover_candidates: bool
    discovery_limit: int
    discovery_results_per_query: int
    discovery_fetch_pages: bool
    discovery_delay_seconds: float
    discovery_search_workers: int
    planned_searches: list[str]
    urls: list[str]
    candidate_leads: list[dict[str, Any]]
    raw_pages: list[dict[str, Any]]
    profile: dict[str, Any]
    retrieved_entries: list[dict[str, Any]]
    rag_db_path: str
    judge: dict[str, Any]
    briefing_pt: str
    briefing_en: str
    human_review_required: bool
    agent_execution_modes: dict[str, str]
    agent_execution_log: list[dict[str, Any]]
    agent_traces: list[dict[str, Any]]
    llm_provider_unavailable: bool
    llm_provider_unavailable_reason: str
    errors: list[str]
