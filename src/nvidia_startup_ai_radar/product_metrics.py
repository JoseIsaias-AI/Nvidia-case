"""Product-facing analytics and readiness helpers for the web UI.

These helpers only read persisted outputs. They do not change the analysis
workflow, so the CLI, Streamlit dashboard, and LangGraph nodes keep the same
business behavior.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from nvidia_startup_ai_radar.rag import DEFAULT_RAG_DB_PATH, rag_index_stats
from nvidia_startup_ai_radar.storage import DEFAULT_DB_PATH, get_run, list_profile_records


CLASSIFICATION_COLORS = {
    "AI-native": "#5FA777",
    "AI-enabled": "#2F4BE0",
    "non-AI": "#D99A3C",
    "indeterminado": "#7C879C",
}

PRIORITY_COLORS = {
    "alta": "#D99A3C",
    "media": "#2F4BE0",
    "baixa": "#7C879C",
}

MATURITY_BINS = [
    {"id": "0-20", "min": 0, "max": 20, "color": "#B76E2D"},
    {"id": "20-40", "min": 20, "max": 40, "color": "#D99A3C"},
    {"id": "40-60", "min": 40, "max": 60, "color": "#B6A24A"},
    {"id": "60-80", "min": 60, "max": 80, "color": "#7CA65F"},
    {"id": "80-100", "min": 80, "max": 101, "color": "#5FA777"},
]

TECHNOLOGY_NEED_RULES = [
    {
        "need": "Inferencia otimizada",
        "technologies": {"NVIDIA NIM", "TensorRT-LLM", "Triton Inference Server", "NVIDIA TensorRT"},
        "terms": [
            "llm",
            "inferencia",
            "latencia",
            "custo",
            "throughput",
            "gpu",
            "triton",
            "tensorrt",
            "nim",
            "api externa",
        ],
    },
    {
        "need": "Governanca de assistentes",
        "technologies": {"NeMo Guardrails", "NVIDIA NeMo"},
        "terms": ["guardrails", "compliance", "lgpd", "agente", "assistente", "risco", "governanca"],
    },
    {
        "need": "Dados tabulares acelerados",
        "technologies": {"NVIDIA RAPIDS", "RAPIDS", "cuDF", "cuML"},
        "terms": ["credito", "fraude", "tabular", "etl", "pipeline de dados", "dados", "fintech", "rapids"],
    },
    {
        "need": "Saude e life sciences",
        "technologies": {"NVIDIA Clara", "MONAI"},
        "terms": ["saude", "health", "clin", "hospital", "oncologia", "prontuario", "anvisa", "clara"],
    },
    {
        "need": "Voz em tempo real",
        "technologies": {"NVIDIA Riva"},
        "terms": ["voz", "audio", "call center", "transcricao", "asr", "tts", "riva"],
    },
    {
        "need": "Visao computacional e edge",
        "technologies": {"NVIDIA TensorRT", "TensorRT-LLM", "Triton Inference Server", "NVIDIA Jetson"},
        "terms": ["visao", "camera", "edge", "jetson", "metropolis", "imagem", "video"],
    },
    {
        "need": "Ciberseguranca com IA",
        "technologies": {"NVIDIA Morpheus"},
        "terms": ["cyber", "seguranca", "ameaca", "anomalia", "morpheus", "logs"],
    },
    {
        "need": "Ecossistema NVIDIA",
        "technologies": {"NVIDIA Inception"},
        "terms": ["startup", "inception", "parceria", "vc", "ecossistema"],
    },
]

OVERVIEW_MIN_RECORDS = 3

NON_PRODUCT_PROFILE_NAMES = {
    "startup nao identificada",
    "startup não identificada",
    "nao identificado",
    "não identificado",
}

NON_PRODUCT_PROFILE_PATTERNS = (
    r"\bwikipedia\b",
    r"\benciclop[eé]dia\b",
    r"\btutorial\b",
    r"\bguia\b",
    r"\bo que sao\b",
    r"\bo que s[aã]o\b",
    r"\bcomo funciona\b",
    r"\bcomo inovam\b",
    r"\bamigo tech\b",
    r"\balura\b",
    r"\bufc\b",
    r"\bo que [eé]\b",
    r"\bwhat is\b",
    r"\bdefinition\b",
    r"\bexplained\b",
    r"\bdocumentation\b",
    r"\bdocs?\b",
    r"\burl planejada para coleta futura\b",
)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_label(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.lower()).strip()


def is_displayable_profile_record(record: dict[str, Any]) -> bool:
    """Hide failed/generic records from the product Radar, not from review/audit."""

    profile = record.get("profile") or {}
    name = _normalize_label(str(record.get("nome") or profile.get("nome") or ""))
    sector = _normalize_label(str(record.get("setor") or profile.get("setor") or ""))
    description = _normalize_label(str(profile.get("produto_descricao") or ""))
    text = " ".join([name, sector, description])
    if name in NON_PRODUCT_PROFILE_NAMES:
        return False
    if any(re.search(pattern, text) for pattern in NON_PRODUCT_PROFILE_PATTERNS):
        return False
    if (
        (not name or name in {"startup", "empresa"})
        and not profile.get("site")
        and _safe_float(record.get("score_maturidade_ia")) <= 0
    ):
        return False
    return True


def _iter_recommendations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for record in records:
        profile = record.get("profile") or {}
        for item in profile.get("recomendacoes_nvidia") or []:
            if isinstance(item, dict) and item.get("tecnologia"):
                recommendations.append(item)
    return recommendations


def _group_key_for_date(value: datetime, mode: str) -> str:
    if mode == "month":
        return value.strftime("%Y-%m")
    iso_year, iso_week, _ = value.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _classification_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(record.get("classificacao") or "indeterminado") for record in records)
    return [
        {
            "name": classification,
            "count": counts.get(classification, 0),
            "color": CLASSIFICATION_COLORS[classification],
        }
        for classification in CLASSIFICATION_COLORS
        if counts.get(classification, 0) > 0
    ]


def _sector_segments(records: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    sector_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        sector = str(record.get("setor") or "Sem setor")
        classification = str(record.get("classificacao") or "indeterminado")
        sector_counts[sector][classification] += 1
    rows: list[dict[str, Any]] = []
    for sector, counts in sector_counts.items():
        total = sum(counts.values())
        row = {
            "sector": sector,
            "total": total,
            "segments": [
                {
                    "classification": classification,
                    "count": counts.get(classification, 0),
                    "share": (counts.get(classification, 0) / total) if total else 0,
                    "color": CLASSIFICATION_COLORS[classification],
                }
                for classification in CLASSIFICATION_COLORS
                if counts.get(classification, 0) > 0
            ],
        }
        rows.append(row)
    rows.sort(key=lambda item: item["total"], reverse=True)
    return rows[:limit]


def _technology_ranking(records: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    by_technology: dict[str, Counter[str]] = defaultdict(Counter)
    for recommendation in _iter_recommendations(records):
        tech = str(recommendation.get("tecnologia") or "").strip()
        priority = str(recommendation.get("prioridade") or "media")
        if tech:
            by_technology[tech][priority] += 1
    rows: list[dict[str, Any]] = []
    for tech, priority_counts in by_technology.items():
        total = sum(priority_counts.values())
        predominant = priority_counts.most_common(1)[0][0] if priority_counts else "media"
        rows.append(
            {
                "technology": tech,
                "count": total,
                "predominant_priority": predominant,
                "color": PRIORITY_COLORS.get(predominant, PRIORITY_COLORS["media"]),
                "priority_counts": dict(priority_counts),
            }
        )
    rows.sort(key=lambda item: (-item["count"], item["technology"]))
    return rows[:limit]


def _recommendation_priority_value(value: str | None) -> int:
    return {"alta": 3, "media": 2, "baixa": 1}.get(str(value or "").lower(), 0)


def _record_profile(record: dict[str, Any]) -> dict[str, Any]:
    profile = record.get("profile")
    return profile if isinstance(profile, dict) else {}


def _profile_search_text(record: dict[str, Any]) -> str:
    return json.dumps(_record_profile(record), ensure_ascii=False, default=str).lower()


def infer_needs_for_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer technical needs from profile signals and NVIDIA recommendations.

    This is intentionally deterministic. It makes the product UI explainable in
    offline mode and mirrors the planning principle: diagnose the startup first,
    then map the NVIDIA stack.
    """

    text = json.dumps(profile, ensure_ascii=False, default=str).lower()
    recommendations = profile.get("recomendacoes_nvidia") or []
    recommended_technologies = {
        str(item.get("tecnologia") or "").strip()
        for item in recommendations
        if isinstance(item, dict) and item.get("tecnologia")
    }
    rows: list[dict[str, Any]] = []
    for rule in TECHNOLOGY_NEED_RULES:
        matching_technologies = sorted(recommended_technologies.intersection(rule["technologies"]))
        matched_terms = [term for term in rule["terms"] if term in text]
        if not matching_technologies and not matched_terms:
            continue
        confidence = min(1.0, (len(matching_technologies) * 0.45) + (len(matched_terms) * 0.12))
        rows.append(
            {
                "need": rule["need"],
                "technologies": matching_technologies,
                "evidence_terms": matched_terms[:5],
                "confidence": round(confidence, 2),
            }
        )
    rows.sort(key=lambda item: (-item["confidence"], item["need"]))
    return rows


def _primary_recommendation(recommendations: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in recommendations if isinstance(item, dict) and item.get("tecnologia")]
    if not valid:
        return None
    return sorted(
        valid,
        key=lambda item: (-_recommendation_priority_value(item.get("prioridade")), str(item.get("tecnologia") or "")),
    )[0]


def _opportunity_score(record: dict[str, Any], needs: list[dict[str, Any]]) -> int:
    profile = _record_profile(record)
    classification = str(record.get("classificacao") or profile.get("classificacao") or "")
    recommendations = profile.get("recomendacoes_nvidia") or []
    evidence_count = len(profile.get("evidencias") or []) + len(profile.get("sinais_ai_native") or [])
    score = _safe_float(record.get("score_maturidade_ia") or profile.get("score_maturidade_ia"))
    value = score * 0.55
    if classification == "AI-native":
        value += 20
    elif classification == "AI-enabled":
        value += 10
    if any(str(item.get("prioridade") or "") == "alta" for item in recommendations if isinstance(item, dict)):
        value += 12
    value += min(10, len(needs) * 3)
    value += min(8, evidence_count * 1.5)
    if profile.get("stack_concorrente_detectada"):
        value += 6
    if record.get("human_review_required") or record.get("needs_human_review"):
        value -= 10
    if record.get("review_status") == "rejeitado":
        value -= 25
    return int(max(0, min(100, round(value))))


def _recommendation_sources(recommendations: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for recommendation in recommendations:
        for evidence in recommendation.get("evidencias") or []:
            if isinstance(evidence, dict) and evidence.get("fonte_url"):
                urls.append(str(evidence["fonte_url"]))
    return list(dict.fromkeys(urls))[:5]


def _risk_flags(record: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if profile.get("stack_concorrente_detectada"):
        flags.append("Stack concorrente")
    if profile.get("sinais_wrapper_risco"):
        flags.append("Risco wrapper")
    if record.get("human_review_required") or record.get("needs_human_review"):
        flags.append("Pede validacao humana")
    if len(profile.get("evidencias") or []) < 2:
        flags.append("Pouca evidencia")
    if record.get("review_status") == "rejeitado":
        flags.append("Rejeitado")
    return flags


def _evidence_summary(profile: dict[str, Any]) -> str:
    for collection in ["sinais_ai_native", "stack_concorrente_evidencias", "evidencias", "sinais_wrapper_risco"]:
        for item in profile.get(collection) or []:
            if not isinstance(item, dict):
                continue
            text = item.get("evidencia_trecho") or item.get("trecho_resumido") or item.get("sinal")
            if text:
                return str(text)[:240]
    return "Sem evidencia resumida suficiente."


def _case_matches(profile: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in profile.get("casos_similares") or []:
        if not isinstance(item, dict):
            continue
        matches.append(
            {
                "case_id": item.get("case_id") or item.get("empresa") or item.get("title"),
                "tipo": item.get("tipo") or item.get("document_type"),
                "licao": item.get("licao") or item.get("licao_estruturada") or item.get("summary"),
            }
        )
    return matches[:3]


def _decision_bucket(
    record: dict[str, Any],
    profile: dict[str, Any],
    opportunity_score: int,
    recommendations: list[dict[str, Any]],
) -> str:
    if record.get("review_status") == "rejeitado":
        return "Fora de envio"
    if profile.get("stack_concorrente_detectada"):
        return "Migrar/substituir stack"
    if recommendations and opportunity_score >= 55 and record.get("review_status", "aprovado") == "aprovado":
        return "Priorizar abordagem"
    if recommendations:
        return "Validar prova tecnica"
    return "Nutrir/qualificar"


def build_opportunity_matrix(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Build the decision layer: startup comparison and NVIDIA stack needs."""

    records = [
        record
        for record in list_profile_records(db_path=db_path, limit=2000)
        if is_displayable_profile_record(record)
    ]
    rows: list[dict[str, Any]] = []
    technology_counts: Counter[str] = Counter()
    need_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    competitor_count = 0
    ready_count = 0

    for record in records:
        profile = _record_profile(record)
        recommendations = [
            item
            for item in profile.get("recomendacoes_nvidia") or []
            if isinstance(item, dict) and item.get("tecnologia")
        ]
        needs = infer_needs_for_profile(profile)
        primary = _primary_recommendation(recommendations)
        for recommendation in recommendations:
            technology_counts[str(recommendation.get("tecnologia"))] += 1
        for need in needs:
            need_counts[str(need.get("need"))] += 1
        if profile.get("stack_concorrente_detectada"):
            competitor_count += 1
        if record.get("review_status", "aprovado") == "aprovado" and recommendations:
            ready_count += 1

        action_type = (
            "Migracao/substituicao"
            if profile.get("stack_concorrente_detectada")
            else "Adocao/expansao NVIDIA"
        )
        opportunity = _opportunity_score(record, needs)
        decision_bucket = _decision_bucket(record, profile, opportunity, recommendations)
        decision_counts[decision_bucket] += 1
        recommended_stack = [
            str(recommendation.get("tecnologia"))
            for recommendation in recommendations
            if recommendation.get("tecnologia")
        ]
        rows.append(
            {
                "run_id": record.get("run_id"),
                "name": record.get("nome") or profile.get("nome"),
                "sector": record.get("setor") or profile.get("setor") or "Nao informado",
                "classification": record.get("classificacao") or profile.get("classificacao") or "indeterminado",
                "maturity_score": round(_safe_float(record.get("score_maturidade_ia")), 2),
                "opportunity_score": opportunity,
                "action_type": action_type,
                "decision_bucket": decision_bucket,
                "review_status": record.get("review_status", "aprovado"),
                "needs": needs,
                "recommendations": recommendations,
                "recommended_stack": list(dict.fromkeys(recommended_stack)),
                "primary_recommendation": primary,
                "next_action": (primary or {}).get("proxima_acao")
                or "Validar evidencias tecnicas antes de abordagem comercial.",
                "competitor_stack": profile.get("stack_concorrente_detectada") or [],
                "risk_flags": _risk_flags(record, profile),
                "evidence_summary": _evidence_summary(profile),
                "source_urls": _recommendation_sources(recommendations),
                "case_matches": _case_matches(profile),
                "evidence_count": len(profile.get("evidencias") or []),
            }
        )

    rows.sort(key=lambda item: (-item["opportunity_score"], -item["maturity_score"], item["name"] or ""))
    return {
        "summary": {
            "startup_count": len(rows),
            "ready_count": ready_count,
            "competitor_count": competitor_count,
            "technology_count": len(technology_counts),
            "top_technology": technology_counts.most_common(1)[0][0] if technology_counts else None,
            "priority_count": decision_counts.get("Priorizar abordagem", 0),
            "migration_count": decision_counts.get("Migrar/substituir stack", 0),
            "validation_count": decision_counts.get("Validar prova tecnica", 0),
        },
        "decision_counts": [
            {"bucket": bucket, "count": count}
            for bucket, count in decision_counts.most_common()
        ],
        "technology_counts": [
            {"technology": technology, "count": count}
            for technology, count in technology_counts.most_common()
        ],
        "need_counts": [
            {"need": need, "count": count}
            for need, count in need_counts.most_common()
        ],
        "items": rows,
    }


def _mapping_evolution(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dates = [parsed for record in records if (parsed := _parse_date(record.get("created_at")))]
    if not dates:
        return []
    dates.sort()
    span_days = (dates[-1] - dates[0]).days
    mode = "month" if span_days > 120 else "week"
    counts = Counter(_group_key_for_date(date, mode) for date in dates)
    cumulative = 0
    points: list[dict[str, Any]] = []
    for key in sorted(counts):
        cumulative += counts[key]
        points.append({"period": key, "count": counts[key], "total": cumulative, "grouping": mode})
    return points


def _maturity_distribution(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = [_safe_float(record.get("score_maturidade_ia")) for record in records]
    for bucket in MATURITY_BINS:
        count = sum(1 for score in scores if bucket["min"] <= score < bucket["max"])
        rows.append({"range": bucket["id"], "count": count, "color": bucket["color"]})
    return rows


def build_overview(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Build home overview chart data from persisted profile records."""

    records = [
        record
        for record in list_profile_records(db_path=db_path, limit=2000)
        if is_displayable_profile_record(record)
    ]
    has_minimum = len(records) >= OVERVIEW_MIN_RECORDS
    recommendations = _iter_recommendations(records)
    return {
        "record_count": len(records),
        "minimum_required": OVERVIEW_MIN_RECORDS,
        "empty_message": "Mapeie mais startups para ver esse panorama",
        "classification_by_sector": {
            "ready": has_minimum,
            "classification_counts": _classification_counts(records) if has_minimum else [],
            "sector_segments": _sector_segments(records) if has_minimum else [],
        },
        "technology_ranking": {
            "ready": has_minimum and bool(recommendations),
            "items": _technology_ranking(records) if has_minimum else [],
        },
        "mapping_evolution": {
            "ready": has_minimum,
            "points": _mapping_evolution(records) if has_minimum else [],
        },
        "maturity_distribution": {
            "ready": has_minimum,
            "bins": _maturity_distribution(records) if has_minimum else [],
        },
    }


def _profile_text(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, default=str).lower()


def profile_signal_axes(profile: dict[str, Any]) -> dict[str, float]:
    """Compute normalized profile dimensions from collected signals."""

    text = _profile_text(profile)
    positive_signals = profile.get("sinais_ai_native") or []
    risk_signals = profile.get("sinais_wrapper_risco") or []
    stack = " ".join(str(item).lower() for item in profile.get("stack_tecnica_detectada") or [])
    score = _safe_float(profile.get("score_maturidade_ia"))
    wrapper_risk = _safe_float(profile.get("score_wrapper_risco"))

    own_infra_terms = ["gpu", "nvidia", "triton", "tensorrt", "cuda", "self-hosted", "infraestrutura"]
    data_terms = ["dado propriet", "dados propriet", "dataset", "base propria", "data moat", "moat"]
    regulated_terms = ["fintech", "banco", "credito", "fraude", "saude", "health", "clin", "lgpd", "anvisa", "seguranca"]
    integration_terms = ["workflow", "integracao", "integrado", "operacional", "automacao", "producao"]
    research_terms = ["p&d", "pesquisa", "universidade", "academ", "paper", "laboratorio"]

    def term_score(terms: list[str], weight: float = 34.0) -> float:
        matches = sum(1 for term in terms if term in text or term in stack)
        return min(100.0, matches * weight)

    return {
        "Infra propria": max(term_score(own_infra_terms, 28.0), min(100.0, len(positive_signals) * 18.0)),
        "Dados exclusivos": term_score(data_terms, 40.0),
        "Setor critico": term_score(regulated_terms, 25.0),
        "Integracao": max(term_score(integration_terms, 30.0), min(100.0, score)),
        "P&D": term_score(research_terms, 35.0),
        "Baixo risco": max(0.0, min(100.0, 100.0 - wrapper_risk - (len(risk_signals) * 18.0))),
    }


def build_profile_radar(run_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    run = get_run(run_id, db_path=db_path)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    profile = run.get("profile") or {}
    current = profile_signal_axes(profile)

    native_records = [
        record
        for record in list_profile_records(db_path=db_path, limit=2000)
        if record.get("classificacao") == "AI-native" and is_displayable_profile_record(record)
    ]
    reference_axes: dict[str, float] | None = None
    if native_records:
        totals = {axis: 0.0 for axis in current}
        for record in native_records:
            axes = profile_signal_axes(record.get("profile") or {})
            for axis, value in axes.items():
                totals[axis] += value
        reference_axes = {axis: round(value / len(native_records), 2) for axis, value in totals.items()}

    rows = [
        {
            "axis": axis,
            "startup": round(value, 2),
            "reference": None if reference_axes is None else reference_axes.get(axis, 0.0),
        }
        for axis, value in current.items()
    ]
    return {
        "run_id": run_id,
        "startup_name": run.get("nome"),
        "reference_count": len(native_records),
        "reference_available": reference_axes is not None,
        "empty_reference_message": "Mapeie startups AI-native para ver o perfil de referencia",
        "axes": rows,
    }


def claim_quality_for_run(run: dict[str, Any]) -> dict[str, Any]:
    """Compute a conservative claim-support summary from persisted evidence."""

    profile = run.get("profile") or {}
    evidence_count = len(profile.get("evidencias") or [])
    signal_count = len(profile.get("sinais_ai_native") or []) + len(profile.get("sinais_wrapper_risco") or [])
    recommendation_count = len(profile.get("recomendacoes_nvidia") or [])
    supported_count = evidence_count + signal_count
    briefing = run.get("briefing_en") or run.get("briefing_pt") or ""
    bullet_claims = [line for line in briefing.splitlines() if line.strip().startswith("- ")]
    claim_count = max(len(bullet_claims), recommendation_count)
    unsupported_count = max(0, claim_count - supported_count)
    unsupported_rate = unsupported_count / claim_count if claim_count else 0.0
    critical_unsupported = unsupported_count if evidence_count == 0 and claim_count else 0
    export_ready = run.get("review_status", "aprovado") == "aprovado" and critical_unsupported == 0
    return {
        "claim_count": claim_count,
        "supported_evidence_count": supported_count,
        "unsupported_claim_count": unsupported_count,
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "critical_unsupported_claim_count": critical_unsupported,
        "recommendation_count": recommendation_count,
        "evidence_coverage": round(min(1.0, supported_count / claim_count), 4) if claim_count else 1.0,
        "export_ready": export_ready,
        "status": "pronto" if export_ready else "precisa_revisao",
    }


def build_quality_summary(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    records = list_profile_records(db_path=db_path, limit=2000)
    per_run = []
    for record in records:
        quality = claim_quality_for_run(record)
        per_run.append(
            {
                "run_id": record.get("run_id"),
                "name": record.get("nome"),
                "classification": record.get("classificacao"),
                **quality,
            }
        )
    if not per_run:
        return {"run_count": 0, "averages": {}, "runs": []}
    return {
        "run_count": len(per_run),
        "averages": {
            "evidence_coverage": round(sum(item["evidence_coverage"] for item in per_run) / len(per_run), 4),
            "unsupported_claim_rate": round(
                sum(item["unsupported_claim_rate"] for item in per_run) / len(per_run),
                4,
            ),
            "export_ready_rate": round(sum(1 for item in per_run if item["export_ready"]) / len(per_run), 4),
        },
        "runs": per_run,
    }


def build_readiness(
    db_path: str | Path = DEFAULT_DB_PATH,
    rag_db_path: str | Path = DEFAULT_RAG_DB_PATH,
) -> dict[str, Any]:
    records = [
        record
        for record in list_profile_records(db_path=db_path, limit=2000)
        if is_displayable_profile_record(record)
    ]
    try:
        stats = rag_index_stats(rag_db_path)
    except Exception as exc:  # pragma: no cover - defensive UI helper
        stats = {"chunk_count": 0, "error": str(exc)}
    checks = [
        {
            "id": "database",
            "label": "Base de startups",
            "status": "ok" if Path(db_path).exists() else "attention",
            "detail": f"{len(records)} registros carregados" if records else "Sem startups mapeadas ainda",
        },
        {
            "id": "knowledge",
            "label": "Base de conhecimento",
            "status": "ok" if int(stats.get("chunk_count") or 0) > 0 else "attention",
            "detail": f"{stats.get('chunk_count', 0)} trechos indexados",
        },
        {
            "id": "groq",
            "label": "Analise com Groq",
            "status": "ok" if os.getenv("LLM_PROVIDER", "").lower() == "groq" and os.getenv("GROQ_API_KEY") else "optional",
            "detail": "Groq ativo" if os.getenv("LLM_PROVIDER", "").lower() == "groq" and os.getenv("GROQ_API_KEY") else "Fallback local disponivel",
        },
        {
            "id": "web",
            "label": "Coleta web",
            "status": "ok" if os.getenv("RADAR_ENABLE_WEB_FETCH", "").lower() in {"1", "true", "yes", "on"} else "optional",
            "detail": "Busca web ligada" if os.getenv("RADAR_ENABLE_WEB_FETCH", "").lower() in {"1", "true", "yes", "on"} else "Modo local protegido",
        },
    ]
    return {
        "ready": all(check["status"] != "attention" for check in checks if check["id"] != "database"),
        "checks": checks,
        "settings": {
            "profile_db": str(db_path),
            "knowledge_db": str(rag_db_path),
            "llm_provider": os.getenv("LLM_PROVIDER", "none") or "none",
        },
    }


def load_json_resource(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def visible_text_guard(frontend_dir: str | Path) -> dict[str, Any]:
    """Scan frontend source for forbidden visible-copy terms."""

    forbidden = ["agente", "agent", "llm", "prompt", "rag", "embedding", "vetor", "chunk", "token", "pipeline"]
    hits: list[dict[str, Any]] = []
    root = Path(frontend_dir)
    for path in root.rglob("*"):
        if any(part in {"node_modules", "dist", ".vite"} for part in path.parts):
            continue
        if path.suffix.lower() not in {".tsx", ".ts", ".jsx", ".js"}:
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in forbidden):
                hits.append({"file": str(path), "line": line_no, "text": line.strip()})
    return {"forbidden_terms": forbidden, "hit_count": len(hits), "hits": hits}
