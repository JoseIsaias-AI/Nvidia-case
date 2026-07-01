"""End-to-end golden set evaluation for the full radar pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.schemas import StartupProfile, utc_now_iso


DEFAULT_GOLDEN_SET_FIXTURE = Path("tests") / "fixtures" / "golden_set.json"
DEFAULT_GOLDEN_SET_REPORT = Path("reports") / "golden_set_eval.md"

Classification = Literal["AI-native", "AI-enabled", "non-AI", "indeterminado"]
Priority = Literal["alta", "media", "baixa"]


@dataclass(frozen=True)
class GoldenSetCase:
    id: str
    question_id: str
    startup_name: str
    sector: str
    input_text: str
    expected_classification: Classification
    expected_priority: Priority
    expected_technologies: list[str]
    expected_human_review: bool
    expected_rationale: str


def _json_load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_golden_set(path: str | Path = DEFAULT_GOLDEN_SET_FIXTURE) -> list[GoldenSetCase]:
    payload = _json_load(path)
    return [GoldenSetCase(**case) for case in payload.get("cases", [])]


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _technology_overlap(actual: list[str], expected: list[str]) -> list[str]:
    actual_norm = {_normalize(item): item for item in actual}
    hits: list[str] = []
    for expected_item in expected:
        expected_norm = _normalize(expected_item)
        if any(expected_norm == key or expected_norm in key or key in expected_norm for key in actual_norm):
            hits.append(expected_item)
    return hits


def _priority_rank(priority: str) -> int:
    return {"baixa": 1, "media": 2, "alta": 3}.get(priority, 0)


def _actual_top_priority(profile: StartupProfile) -> str | None:
    if not profile.recomendacoes_nvidia:
        return None
    return max((recommendation.prioridade for recommendation in profile.recomendacoes_nvidia), key=_priority_rank)


def _inbound_profile(case: GoldenSetCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "nome": case.startup_name,
        "setor": case.sector,
        "origem": "inbound",
        "produto_descricao": case.input_text,
        "evidencias": [
            {
                "fonte_url": f"local://golden-set/{case.id}",
                "trecho_resumido": case.input_text,
                "data_coleta": utc_now_iso(),
            }
        ],
    }


def _case_result(case: GoldenSetCase, state: dict[str, Any]) -> dict[str, Any]:
    profile = StartupProfile.model_validate(state.get("profile") or {})
    actual_technologies = [recommendation.tecnologia for recommendation in profile.recomendacoes_nvidia]
    technology_hits = _technology_overlap(actual_technologies, case.expected_technologies)
    technology_applicable = bool(case.expected_technologies)
    top_priority = _actual_top_priority(profile)
    classification_ok = profile.classificacao == case.expected_classification
    technology_ok = bool(technology_hits) if technology_applicable else not actual_technologies or top_priority == "baixa"
    priority_ok = top_priority == case.expected_priority
    human_review = bool(state.get("human_review_required"))
    review_ok = human_review == case.expected_human_review
    errors: list[str] = []
    if not classification_ok:
        errors.append(f"classificacao esperada {case.expected_classification}, veio {profile.classificacao}")
    if technology_applicable and not technology_ok:
        errors.append(
            "tecnologia esperada sem sobreposicao: "
            f"{', '.join(case.expected_technologies)} vs {', '.join(actual_technologies) or 'nenhuma'}"
        )
    if not priority_ok:
        errors.append(f"prioridade esperada {case.expected_priority}, veio {top_priority or 'nenhuma'}")
    if not review_ok:
        errors.append(f"revisao humana esperada {case.expected_human_review}, veio {human_review}")

    return {
        "id": case.id,
        "question_id": case.question_id,
        "startup_name": case.startup_name,
        "expected_classification": case.expected_classification,
        "actual_classification": profile.classificacao,
        "classification_ok": classification_ok,
        "expected_priority": case.expected_priority,
        "actual_top_priority": top_priority,
        "priority_ok": priority_ok,
        "expected_technologies": case.expected_technologies,
        "actual_technologies": actual_technologies,
        "technology_hits": technology_hits,
        "technology_applicable": technology_applicable,
        "technology_ok": technology_ok,
        "expected_human_review": case.expected_human_review,
        "human_review_required": human_review,
        "review_ok": review_ok,
        "score_maturidade_ia": profile.score_maturidade_ia,
        "score_wrapper_risco": profile.score_wrapper_risco,
        "agent_execution_modes": state.get("agent_execution_modes", {}),
        "errors": errors,
    }


def _metric_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    technology_applicable = [result for result in results if result["technology_applicable"]]
    return {
        "case_count": total,
        "classification_accuracy": _metric_ratio(
            sum(1 for result in results if result["classification_ok"]),
            total,
        ),
        "technology_hit_rate": _metric_ratio(
            sum(1 for result in technology_applicable if result["technology_ok"]),
            len(technology_applicable),
        ),
        "priority_accuracy": _metric_ratio(
            sum(1 for result in results if result["priority_ok"]),
            total,
        ),
        "human_review_count": sum(1 for result in results if result["human_review_required"]),
        "expected_human_review_count": sum(1 for result in results if result["expected_human_review"]),
        "failed_case_count": sum(1 for result in results if result["errors"]),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown_report(evaluation: dict[str, Any]) -> str:
    metrics = evaluation["metrics"]
    rows = evaluation["cases"]
    lines = [
        "# Golden Set Evaluation",
        "",
        f"- Gerado em: `{evaluation['evaluated_at']}`",
        f"- Fixture: `{evaluation['fixture_path']}`",
        f"- RAG DB: `{evaluation['rag_db_path']}`",
        f"- Casos: {metrics['case_count']}",
        f"- Acuracia de classificacao: {_pct(metrics['classification_accuracy'])}",
        f"- Taxa de acerto de tecnologia: {_pct(metrics['technology_hit_rate'])}",
        f"- Acuracia de prioridade: {_pct(metrics['priority_accuracy'])}",
        f"- Casos sinalizados para revisao humana: {metrics['human_review_count']} "
        f"(esperado: {metrics['expected_human_review_count']})",
        "",
        "## Casos",
        "",
        "| Caso | Esperado | Obtido | Tech hit | Prioridade | Revisao | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        status = "OK" if not row["errors"] else "ERRO"
        tech_hit = ", ".join(row["technology_hits"]) if row["technology_hits"] else "-"
        expected = row["expected_classification"]
        actual = row["actual_classification"]
        priority = f"{row['expected_priority']} -> {row['actual_top_priority'] or '-'}"
        review = f"{row['expected_human_review']} -> {row['human_review_required']}"
        lines.append(
            "| "
            f"{row['startup_name']} | {expected} | {actual} | {tech_hit} | "
            f"{priority} | {review} | {status} |"
        )

    failed = [row for row in rows if row["errors"]]
    lines.extend(["", "## Erros Para Priorizar", ""])
    if not failed:
        lines.append("Nenhum erro detectado nos criterios configurados.")
    for row in failed:
        lines.append(f"### {row['startup_name']}")
        for error in row["errors"]:
            lines.append(f"- {error}")
        lines.append(f"- Tecnologias obtidas: {', '.join(row['actual_technologies']) or 'nenhuma'}")
        lines.append(f"- Modos dos agentes: `{json.dumps(row['agent_execution_modes'], ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def evaluate_pipeline_golden_set(
    *,
    fixture_path: str | Path = DEFAULT_GOLDEN_SET_FIXTURE,
    report_path: str | Path = DEFAULT_GOLDEN_SET_REPORT,
    rag_db_path: str | Path | None = None,
    output_language: Literal["pt", "en", "both"] = "pt",
) -> dict[str, Any]:
    """Run all golden set cases through the full graph and write a Markdown report."""

    cases = load_golden_set(fixture_path)
    results: list[dict[str, Any]] = []
    for case in cases:
        state = run_radar(
            query="",
            inbound_profile=_inbound_profile(case),
            output_language=output_language,
            rag_db_path=str(rag_db_path) if rag_db_path else None,
        )
        results.append(_case_result(case, state))

    evaluation = {
        "evaluated_at": utc_now_iso(),
        "fixture_path": str(fixture_path),
        "report_path": str(report_path),
        "rag_db_path": str(rag_db_path) if rag_db_path else None,
        "metrics": summarize_results(results),
        "cases": results,
    }
    report = render_markdown_report(evaluation)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return evaluation
