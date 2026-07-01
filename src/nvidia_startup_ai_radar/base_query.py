"""Natural-language querying over persisted startup profiles."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from nvidia_startup_ai_radar.config import get_settings
from nvidia_startup_ai_radar.llm import build_chat_model, invoke_structured_with_retry
from nvidia_startup_ai_radar.rag import embed_text, normalize_text
from nvidia_startup_ai_radar.storage import DEFAULT_DB_PATH, list_profile_records


Classification = Literal["AI-native", "AI-enabled", "non-AI", "indeterminado"]


class StartupBaseFilter(BaseModel):
    setor: str | None = None
    classificacao: Classification | None = None
    required_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    required_technologies: list[str] = Field(default_factory=list)
    missing_technologies: list[str] = Field(default_factory=list)
    required_competitor_stack: list[str] = Field(default_factory=list)


def _fallback_filter(question: str) -> StartupBaseFilter:
    normalized = normalize_text(question)
    data = StartupBaseFilter()
    if any(term in normalized for term in ["saude", "health", "clinico", "clinical", "hospital"]):
        data.setor = "Healthtech"
    elif any(term in normalized for term in ["fintech", "credito", "pagamento", "banco", "pix"]):
        data.setor = "Fintech"
    elif any(term in normalized for term in ["seguranca", "cyber", "security"]):
        data.setor = "Seguranca"

    if "ai-native" in normalized or "ai native" in normalized:
        data.classificacao = "AI-native"
    elif "ai-enabled" in normalized or "ai enabled" in normalized:
        data.classificacao = "AI-enabled"
    elif "non-ai" in normalized or "non ai" in normalized:
        data.classificacao = "non-AI"

    term_map = {
        "llm": ["llm", "large language model"],
        "gpu": ["gpu", "nvidia"],
        "triton": ["triton"],
        "tensorrt": ["tensorrt", "tensorrt-llm"],
        "guardrails": ["guardrails", "nemo guardrails"],
        "bedrock": ["bedrock", "aws bedrock", "amazon bedrock"],
        "vertex": ["vertex ai", "google vertex ai"],
        "azure openai": ["azure openai"],
    }
    for label, variants in term_map.items():
        if any(variant in normalized for variant in variants):
            if label == "bedrock":
                data.required_competitor_stack.append("AWS Bedrock")
            elif label == "vertex":
                data.required_competitor_stack.append("Google Vertex AI")
            elif label == "azure openai":
                data.required_competitor_stack.append("Azure OpenAI")
            else:
                data.required_terms.append(label)

    negation_near_guardrails = any(
        phrase in normalized
        for phrase in [
            "sem guardrails",
            "nao tem guardrails",
            "nao têm guardrails",
            "sem nemo guardrails",
            "nao usa guardrails",
        ]
    )
    if negation_near_guardrails:
        data.required_terms = [term for term in data.required_terms if term != "guardrails"]
        data.missing_technologies.append("guardrails")

    return data


def _llm_filter(question: str) -> StartupBaseFilter | None:
    settings = get_settings()
    llm = build_chat_model(settings)
    if llm is None:
        return None
    result, _errors = invoke_structured_with_retry(
        llm,
        StartupBaseFilter,
        (
            "Voce traduz perguntas sobre startups para filtros estruturados. "
            "Use somente campos do schema. Tecnologias podem incluir LLM, GPU, Triton, "
            "TensorRT, NeMo Guardrails, AWS Bedrock, Google Vertex AI e Azure OpenAI. "
            "Retorne JSON valido."
        ),
        f"Pergunta: {question}",
        max_retries=1,
    )
    return result


def translate_question_to_filter(question: str) -> StartupBaseFilter:
    llm_filter = _llm_filter(question)
    if llm_filter is not None:
        fallback = _fallback_filter(question)
        if not any(
            [
                llm_filter.setor,
                llm_filter.classificacao,
                llm_filter.required_terms,
                llm_filter.excluded_terms,
                llm_filter.required_technologies,
                llm_filter.missing_technologies,
                llm_filter.required_competitor_stack,
            ]
        ):
            return fallback
        return llm_filter
    return _fallback_filter(question)


def _profile_text(record: dict) -> str:
    profile = record.get("profile", {})
    parts = [
        str(record.get("nome", "")),
        str(record.get("setor", "")),
        str(record.get("classificacao", "")),
        str(profile.get("produto_descricao") or ""),
        " ".join(profile.get("stack_tecnica_detectada") or []),
        " ".join(profile.get("stack_concorrente_detectada") or []),
        " ".join(signal.get("sinal", "") for signal in profile.get("sinais_ai_native", [])),
        " ".join(signal.get("sinal", "") for signal in profile.get("sinais_wrapper_risco", [])),
        " ".join(rec.get("tecnologia", "") for rec in profile.get("recomendacoes_nvidia", [])),
        " ".join(ev.get("trecho_resumido", "") for ev in profile.get("evidencias", [])),
    ]
    return " ".join(part for part in parts if part)


def _evidence_excerpt(profile: dict) -> str:
    evidences = profile.get("evidencias") or []
    if not evidences:
        return ""
    return str(evidences[0].get("trecho_resumido") or "")[:360]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _contains_any(text: str, terms: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(term) in normalized for term in terms)


def _matches_filter(record: dict, query_filter: StartupBaseFilter) -> bool:
    profile = record.get("profile", {})
    profile_text = _profile_text(record)
    normalized_text = normalize_text(profile_text)

    if query_filter.setor and normalize_text(query_filter.setor) not in normalize_text(str(record.get("setor", ""))):
        return False
    if query_filter.classificacao and record.get("classificacao") != query_filter.classificacao:
        return False
    for term in query_filter.required_terms:
        if normalize_text(term) not in normalized_text:
            return False
    for term in query_filter.excluded_terms:
        if normalize_text(term) in normalized_text:
            return False
    for technology in query_filter.required_technologies:
        if normalize_text(technology) not in normalized_text:
            return False
    for technology in query_filter.missing_technologies:
        if normalize_text(technology) in normalized_text:
            return False
    competitor_stack = " ".join(profile.get("stack_concorrente_detectada") or [])
    for competitor in query_filter.required_competitor_stack:
        if not _contains_any(competitor_stack, [competitor]):
            return False
    return True


def query_startup_base(
    question: str,
    *,
    db_path=DEFAULT_DB_PATH,
    limit: int = 20,
    scan_limit: int = 500,
) -> dict:
    """Return structured startup rows matching a natural-language question."""

    query_filter = translate_question_to_filter(question)
    records = list_profile_records(db_path, limit=scan_limit)
    query_embedding = embed_text(question)
    rows: list[dict] = []
    for record in records:
        if not _matches_filter(record, query_filter):
            continue
        profile = record.get("profile", {})
        text = _profile_text(record)
        semantic_score = _cosine(query_embedding, embed_text(text))
        rows.append(
            {
                "run_id": record["run_id"],
                "nome": record["nome"],
                "setor": record.get("setor"),
                "classificacao": record.get("classificacao"),
                "score_maturidade_ia": record.get("score_maturidade_ia"),
                "score_wrapper_risco": record.get("score_wrapper_risco"),
                "stack_concorrente_detectada": profile.get("stack_concorrente_detectada", []),
                "tecnologias_recomendadas": [
                    rec.get("tecnologia") for rec in profile.get("recomendacoes_nvidia", [])
                ],
                "semantic_score": round(semantic_score, 4),
                "evidencia": _evidence_excerpt(profile),
            }
        )
    rows.sort(key=lambda item: (item["semantic_score"], item["score_maturidade_ia"] or 0), reverse=True)
    return {
        "question": question,
        "filter": query_filter.model_dump(),
        "count": len(rows[:limit]),
        "results": rows[:limit],
    }
