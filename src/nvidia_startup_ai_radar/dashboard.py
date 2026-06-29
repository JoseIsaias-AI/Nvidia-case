"""Streamlit dashboard for the NVIDIA Startup AI Radar."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nvidia_startup_ai_radar.exporting import briefing_markdown, markdown_to_pdf_bytes, slugify
from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.storage import (
    DEFAULT_DB_PATH,
    get_run,
    list_distinct_values,
    list_recent_runs,
    save_run,
)


CLASSIFICATIONS = ["AI-native", "AI-enabled", "non-AI", "indeterminado"]


def _load_json_payload(uploaded_file: Any, raw_text: str) -> dict[str, Any]:
    if uploaded_file is not None:
        return json.loads(uploaded_file.getvalue().decode("utf-8"))
    if raw_text.strip():
        return json.loads(raw_text)
    return {}


def _avg_score(runs: list[dict[str, Any]], field: str) -> float:
    values = [float(run[field]) for run in runs if run.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def _format_run_option(run: dict[str, Any]) -> str:
    return (
        f"#{run['run_id']} - {run['nome']} | {run['classificacao']} | "
        f"{run['score_maturidade_ia']:.0f}/100"
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="NVIDIA Startup AI Radar", layout="wide")
    st.title("NVIDIA Startup AI Radar")

    db_path = st.sidebar.text_input("SQLite", value=str(DEFAULT_DB_PATH))
    limit = st.sidebar.slider("Linhas", min_value=10, max_value=200, value=50, step=10)

    tab_analyze, tab_profiles, tab_briefing = st.tabs(["Analise", "Perfis", "Briefing"])

    with tab_analyze:
        left, right = st.columns([1.2, 1])
        with left:
            query = st.text_area("Consulta outbound", height=160)
            output_language = st.segmented_control(
                "Idioma",
                options=["pt", "en", "both"],
                default="pt",
            )
            run_clicked = st.button("Rodar e salvar", type="primary", use_container_width=True)
        with right:
            inbound_text = st.text_area("Perfil inbound JSON", height=160)
            uploaded_file = st.file_uploader("Arquivo JSON inbound", type=["json"])

        if run_clicked:
            payload_error = False
            try:
                inbound_profile = _load_json_payload(uploaded_file, inbound_text)
            except json.JSONDecodeError as exc:
                st.error(f"JSON invalido: {exc}")
                inbound_profile = {}
                payload_error = True

            if not query.strip() and not inbound_profile:
                st.error("Informe uma consulta outbound ou um perfil inbound JSON.")
            elif not payload_error:
                with st.spinner("Executando grafo e persistindo perfil..."):
                    final_state = run_radar(
                        query=query.strip(),
                        inbound_profile=inbound_profile,
                        output_language=output_language,
                    )
                    run_id = save_run(final_state, db_path)
                    st.session_state["selected_run_id"] = run_id
                st.success(f"Perfil salvo como run_id={run_id}.")
                st.markdown(final_state.get("briefing_en") or final_state.get("briefing_pt", ""))

    with tab_profiles:
        classifications = ["Todos", *CLASSIFICATIONS]
        sectors = ["Todos", *list_distinct_values("setor", db_path)]
        filter_cols = st.columns([1, 1, 1, 1])
        with filter_cols[0]:
            selected_classification = st.selectbox("Classificacao", classifications)
        with filter_cols[1]:
            selected_sector = st.selectbox("Setor", sectors)
        with filter_cols[2]:
            search = st.text_input("Busca")
        with filter_cols[3]:
            review_only = st.toggle("Revisao humana")

        runs = list_recent_runs(
            db_path,
            limit=limit,
            classificacao=None if selected_classification == "Todos" else selected_classification,
            setor=None if selected_sector == "Todos" else selected_sector,
            human_review_required=True if review_only else None,
            search=search or None,
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Execucoes", len(runs))
        metric_cols[1].metric("Score medio", f"{_avg_score(runs, 'score_maturidade_ia'):.0f}/100")
        metric_cols[2].metric("Risco medio", f"{_avg_score(runs, 'score_wrapper_risco'):.0f}/100")
        metric_cols[3].metric(
            "Revisao humana",
            sum(1 for run in runs if run.get("human_review_required")),
        )

        if runs:
            st.dataframe(runs, hide_index=True, use_container_width=True)
            options = {_format_run_option(run): run["run_id"] for run in runs}
            selected = st.selectbox("Abrir execucao", list(options))
            if st.button("Abrir briefing", use_container_width=True):
                st.session_state["selected_run_id"] = options[selected]
        else:
            st.info("Nenhuma execucao encontrada para os filtros atuais.")

    with tab_briefing:
        recent_runs = list_recent_runs(db_path, limit=limit)
        default_run_id = st.session_state.get("selected_run_id")
        if default_run_id is None and recent_runs:
            default_run_id = recent_runs[0]["run_id"]

        run_id = st.number_input("run_id", min_value=1, value=int(default_run_id or 1), step=1)
        run = get_run(int(run_id), db_path)
        if run is None:
            st.info("Execucao nao encontrada.")
            return

        profile = run["profile"]
        header_cols = st.columns(4)
        header_cols[0].metric("Startup", run["nome"])
        header_cols[1].metric("Classificacao", run["classificacao"])
        header_cols[2].metric("Maturidade", f"{run['score_maturidade_ia']:.0f}/100")
        header_cols[3].metric("Risco wrapper", f"{run['score_wrapper_risco']:.0f}/100")

        briefing = run.get("briefing_en") or run.get("briefing_pt") or ""
        markdown_export = briefing_markdown(run)
        filename = f"run-{run['run_id']}-{slugify(run['nome'])}"
        download_cols = st.columns([1, 1, 2])
        download_cols[0].download_button(
            "Markdown",
            data=markdown_export,
            file_name=f"{filename}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        download_cols[1].download_button(
            "PDF",
            data=markdown_to_pdf_bytes(markdown_export, title=run["nome"]),
            file_name=f"{filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown(briefing)

        recommendations = profile.get("recomendacoes_nvidia", [])
        if recommendations:
            st.subheader("Recomendacoes")
            for recommendation in recommendations:
                with st.expander(recommendation.get("tecnologia", "Tecnologia")):
                    st.write(recommendation)

        with st.expander("StartupProfile JSON"):
            st.json(profile)
        if run["errors"]:
            with st.expander("Erros"):
                st.json(run["errors"])


def launch() -> None:
    dashboard_path = Path(__file__).resolve()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path), *sys.argv[1:]],
        check=False,
    )


if __name__ == "__main__":
    main()
