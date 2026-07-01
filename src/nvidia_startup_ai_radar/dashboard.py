"""Streamlit dashboard for the NVIDIA Startup AI Radar."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nvidia_startup_ai_radar.base_query import query_startup_base
from nvidia_startup_ai_radar.discovery import (
    DEFAULT_DISCOVERY_DB_PATH,
    DEFAULT_DISCOVERY_OUTPUT,
    DISCOVERY_CAMPAIGNS,
    candidates_as_dicts,
    discover_startups,
    list_discovery_candidates,
    save_candidates,
    save_candidates_sqlite,
)
from nvidia_startup_ai_radar.exporting import briefing_markdown, markdown_to_pdf_bytes, slugify
from nvidia_startup_ai_radar.pipeline import run_radar
from nvidia_startup_ai_radar.rag import (
    DEFAULT_RAG_DB_PATH,
    evaluate_rag_golden_set,
    rag_index_stats,
    rag_search,
    rebuild_rag_index,
)
from nvidia_startup_ai_radar.rag_ingestion import DEFAULT_RAW_SOURCE_DIR, ingest_rag_sources
from nvidia_startup_ai_radar.storage import (
    DEFAULT_DB_PATH,
    agent_trace_summary,
    get_agent_traces,
    get_run,
    list_distinct_values,
    list_recent_runs,
    list_review_queue,
    save_run,
    update_review_status,
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
    rag_db_path = st.sidebar.text_input("RAG SQLite", value=str(DEFAULT_RAG_DB_PATH))
    rag_source_dir = st.sidebar.text_input("Fontes RAG brutas", value=str(DEFAULT_RAW_SOURCE_DIR))
    discovery_output = st.sidebar.text_input("Descoberta JSONL", value=str(DEFAULT_DISCOVERY_OUTPUT))
    discovery_db_path = st.sidebar.text_input("Descoberta SQLite", value=str(DEFAULT_DISCOVERY_DB_PATH))
    limit = st.sidebar.slider("Linhas", min_value=10, max_value=200, value=50, step=10)

    (
        tab_home,
        tab_discovery,
        tab_analyze,
        tab_profiles,
        tab_review,
        tab_ask,
        tab_briefing,
        tab_observability,
        tab_rag,
    ) = st.tabs(
        [
            "Guia",
            "Descoberta",
            "Analisar",
            "Base",
            "Revisao",
            "Pergunte",
            "Briefing",
            "Observabilidade",
            "RAG",
        ]
    )

    with tab_home:
        st.subheader("Fluxo do sistema")
        st.write(
            "1. Use **Analisar** para colar uma startup, tese ou perfil inbound.\n\n"
            "2. O grafo coleta evidencias, estrutura o perfil, classifica maturidade de IA, "
            "consulta o RAG NVIDIA e gera recomendacoes.\n\n"
            "3. Use **Descoberta** para garimpar startups reais na web publica.\n\n"
            "4. Use **Base** para ver execucoes salvas.\n\n"
            "5. Use **Revisao** para aprovar ou rejeitar briefings sinalizados.\n\n"
            "6. Use **Pergunte** para consultar a base salva em linguagem natural.\n\n"
            "7. Use **Briefing** para revisar e baixar Markdown/PDF.\n\n"
            "8. Use **Observabilidade** para auditar latencia, erro e modo por agente.\n\n"
            "9. Use **RAG** para baixar docs oficiais NVIDIA, reindexar e testar chunks."
        )
        st.info(
            "Exemplo rapido para a aba Analisar: "
            "Noleak usa NVIDIA GPUs, TensorRT e Triton Inference Server para visao computacional "
            "em cameras de seguranca, com P&D em universidades."
        )
        st.write("Status atual do RAG:")
        st.json(rag_index_stats(rag_db_path))

    with tab_discovery:
        st.subheader("Descoberta outbound de startups")
        st.caption(
            "O produto ja traz as campanhas e queries do planejamento. Rode o radar completo "
            "ou escolha um foco; nao e necessario escrever query manualmente."
        )
        campaign_options = {campaign.label: key for key, campaign in DISCOVERY_CAMPAIGNS.items()}
        campaign_label = st.selectbox(
            "Campanha de descoberta",
            list(campaign_options),
            index=list(campaign_options).index("Radar completo"),
        )
        selected_campaign_key = campaign_options[campaign_label]
        selected_campaign = DISCOVERY_CAMPAIGNS[selected_campaign_key]
        st.info(selected_campaign.description)

        discovery_cols = st.columns([1, 1, 1, 1])
        with discovery_cols[0]:
            discovery_limit = st.slider("Candidatas", min_value=10, max_value=100, value=50, step=10)
        with discovery_cols[1]:
            results_per_query = st.slider("Profundidade", min_value=2, max_value=10, value=6)
        with discovery_cols[2]:
            fetch_pages = st.toggle("Buscar paginas", value=True)
        with discovery_cols[3]:
            run_discovery = st.button("Rodar descoberta", type="primary", use_container_width=True)

        advanced_queries: list[str] | None = None
        with st.expander("Ajustes avancados"):
            st.caption("Uso opcional para debug. Em modo produto, deixe vazio e use a campanha selecionada.")
            advanced_query_text = st.text_area(
                "Queries customizadas opcionais",
                value="",
                height=120,
            )
            if advanced_query_text.strip():
                advanced_queries = [line.strip() for line in advanced_query_text.splitlines() if line.strip()]
            st.write({"queries_internas_da_campanha": len(selected_campaign.queries)})

        if run_discovery:
            with st.spinner("Buscando fontes publicas e extraindo sinais..."):
                candidates = discover_startups(
                    queries=advanced_queries,
                    campaign=selected_campaign_key,
                    limit=discovery_limit,
                    results_per_query=results_per_query,
                    fetch_pages=fetch_pages,
                )
                save_candidates(candidates, discovery_output)
                save_candidates_sqlite(candidates, discovery_db_path, replace=True)
                st.session_state["discovery_candidates"] = candidates_as_dicts(candidates)
            st.success(
                f"{len(candidates)} candidatas salvas em {discovery_output} e {discovery_db_path} "
                f"pela campanha {selected_campaign.label}."
            )

        load_cols = st.columns([1, 1, 2])
        with load_cols[0]:
            selected_quality = st.selectbox("Tier salvo", ["Todos", "alta", "media", "baixa", "triagem"])
        with load_cols[1]:
            if st.button("Carregar salvas", use_container_width=True):
                st.session_state["discovery_candidates"] = list_discovery_candidates(
                    discovery_db_path,
                    limit=limit,
                    quality_tier=None if selected_quality == "Todos" else selected_quality,
                )

        candidates = st.session_state.get("discovery_candidates", [])
        if candidates:
            table_rows = [
                {
                    "score": candidate["score"],
                    "tier": candidate.get("quality_tier", ""),
                    "startup": candidate["name"],
                    "fonte": candidate["source_domain"],
                    "tipo": candidate.get("source_type", ""),
                    "time": candidate.get("team_size"),
                    "nvidia": ", ".join(candidate["nvidia_signals"][:4]),
                    "concorrente": ", ".join(candidate["competitor_stack_signals"][:4]),
                    "framework": ", ".join(candidate["ai_framework_signals"][:4]),
                    "url": candidate["url"],
                }
                for candidate in candidates
            ]
            st.dataframe(table_rows, hide_index=True, use_container_width=True)
            for candidate in candidates[:10]:
                label = f"{candidate['name']} | score {candidate['score']} | {candidate['source_domain']}"
                with st.expander(label):
                    st.write(candidate["url"])
                    if candidate.get("company_website"):
                        st.write(candidate["company_website"])
                    st.write(candidate["evidence_excerpt"])
                    st.json(
                        {
                            "query": candidate["source_query"],
                            "tier": candidate.get("quality_tier"),
                            "acao": candidate.get("recommended_action"),
                            "location": candidate.get("location"),
                            "team_size": candidate.get("team_size"),
                            "nvidia_signals": candidate["nvidia_signals"],
                            "ai_framework_signals": candidate["ai_framework_signals"],
                            "competitor_stack_signals": candidate["competitor_stack_signals"],
                            "maturity_signals": candidate["maturity_signals"],
                            "wrapper_risk_signals": candidate["wrapper_risk_signals"],
                            "collected_at": candidate["collected_at"],
                        }
                    )
            options = {f"{candidate['name']} | {candidate['score']} | {candidate.get('quality_tier', '')}": candidate for candidate in candidates}
            selected_candidate_label = st.selectbox("Candidata para analise profunda", list(options))
            if st.button("Analisar candidata selecionada", type="primary", use_container_width=True):
                selected_candidate = options[selected_candidate_label]
                query = selected_candidate.get("analysis_query") or (
                    f"{selected_candidate['name']}. Fonte: {selected_candidate['url']}. "
                    f"Evidencia: {selected_candidate['evidence_excerpt']}"
                )
                with st.spinner("Executando grafo para a candidata selecionada..."):
                    final_state = run_radar(
                        query=str(query),
                        output_language="pt",
                        rag_db_path=rag_db_path,
                    )
                    run_id = save_run(final_state, db_path)
                    st.session_state["selected_run_id"] = run_id
                st.success(f"Analise salva como run_id={run_id}. Abra em Briefing.")
                st.markdown(final_state.get("briefing_pt", ""))
        else:
            st.info("Rode a busca para montar uma lista priorizada de startups candidatas.")

    with tab_analyze:
        st.subheader("Rodar uma nova analise")
        st.caption(
            "Cole uma descricao outbound ou um JSON inbound. O resultado sera salvo no SQLite "
            "e ficara disponivel nas abas Base e Briefing."
        )
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
                        rag_db_path=rag_db_path,
                    )
                    run_id = save_run(final_state, db_path)
                    st.session_state["selected_run_id"] = run_id
                st.success(f"Perfil salvo como run_id={run_id}.")
                st.markdown(final_state.get("briefing_en") or final_state.get("briefing_pt", ""))

    with tab_profiles:
        st.subheader("Execucoes salvas")
        st.caption("Filtre startups ja analisadas por classificacao, setor ou revisao humana.")
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

    with tab_review:
        st.subheader("Fila de revisao humana")
        st.caption(
            "Execucoes sinalizadas pelo Evidence Validator ou Judge ficam pendentes ate aprovacao humana. "
            "Briefings rejeitados nao sao tratados como prontos para envio."
        )
        review_filter = st.segmented_control(
            "Status",
            options=["todos", "pendente", "aprovado", "rejeitado"],
            default="pendente",
        )
        queue = list_review_queue(
            db_path,
            limit=limit,
            review_status=None if review_filter == "todos" else review_filter,
        )
        status_counts = {
            status: sum(1 for item in queue if item.get("review_status") == status)
            for status in ["pendente", "aprovado", "rejeitado"]
        }
        review_cols = st.columns(4)
        review_cols[0].metric("Na fila", len(queue))
        review_cols[1].metric("Pendentes", status_counts["pendente"])
        review_cols[2].metric("Aprovados", status_counts["aprovado"])
        review_cols[3].metric("Rejeitados", status_counts["rejeitado"])

        if not queue:
            st.info("Nenhuma execucao sinalizada para esse status.")
        for item in queue:
            label = (
                f"#{item['run_id']} - {item['nome']} | {item['classificacao']} | "
                f"{item['review_status']}"
            )
            with st.expander(label, expanded=item.get("review_status") == "pendente"):
                if item.get("review_status") == "rejeitado":
                    st.error("Rejeitado: nao aparece como pronto para envio.")
                elif item.get("review_status") == "aprovado":
                    st.success("Aprovado para exportacao/envio.")
                else:
                    st.warning("Pendente de decisao humana.")

                st.write(
                    {
                        "setor": item.get("setor"),
                        "score_maturidade_ia": item.get("score_maturidade_ia"),
                        "score_wrapper_risco": item.get("score_wrapper_risco"),
                        "created_at": item.get("created_at"),
                        "reviewed_at": item.get("reviewed_at"),
                    }
                )
                st.markdown("**Motivos da sinalizacao**")
                for reason in item.get("review_motivos") or ["Sem motivo estruturado salvo."]:
                    st.write(f"- {reason}")
                briefing = item.get("briefing_pt") or item.get("briefing_en") or ""
                if briefing:
                    with st.expander("Preview do briefing"):
                        st.markdown(briefing)
                note_key = f"review-note-{item['run_id']}"
                note = st.text_area(
                    "Nota da revisao",
                    value=item.get("review_nota") or "",
                    key=note_key,
                    height=80,
                )
                action_cols = st.columns([1, 1, 3])
                if action_cols[0].button("Aprovar", key=f"approve-{item['run_id']}", use_container_width=True):
                    update_review_status(item["run_id"], "aprovado", note, db_path)
                    st.success("Briefing aprovado.")
                    st.rerun()
                if action_cols[1].button("Rejeitar", key=f"reject-{item['run_id']}", use_container_width=True):
                    update_review_status(item["run_id"], "rejeitado", note, db_path)
                    st.error("Briefing rejeitado.")
                    st.rerun()

    with tab_ask:
        st.subheader("Pergunte a base")
        st.caption(
            "Digite uma pergunta em linguagem natural. O sistema traduz para filtros estruturados "
            "e ranqueia startups salvas pelas evidencias do StartupProfile."
        )
        question = st.text_input(
            "Pergunta",
            value="quais startups de saude usam LLM mas nao tem guardrails",
        )
        ask_cols = st.columns([1, 1, 2])
        with ask_cols[0]:
            ask_limit = st.slider("Resultados", min_value=3, max_value=50, value=10)
        with ask_cols[1]:
            scan_limit = st.slider("Escanear", min_value=50, max_value=1000, value=500, step=50)
        with ask_cols[2]:
            ask_clicked = st.button("Consultar base", type="primary", use_container_width=True)

        if ask_clicked and question.strip():
            with st.spinner("Traduzindo pergunta e buscando evidencias salvas..."):
                answer = query_startup_base(
                    question.strip(),
                    db_path=db_path,
                    limit=ask_limit,
                    scan_limit=scan_limit,
                )
            st.write("Filtro interpretado")
            st.json(answer["filter"])
            results = answer["results"]
            if results:
                st.dataframe(
                    [
                        {
                            "run_id": row["run_id"],
                            "startup": row["nome"],
                            "setor": row["setor"],
                            "classificacao": row["classificacao"],
                            "score": row["score_maturidade_ia"],
                            "stack concorrente": ", ".join(row["stack_concorrente_detectada"]),
                            "tecnologias": ", ".join(str(item) for item in row["tecnologias_recomendadas"] if item),
                            "similaridade": row["semantic_score"],
                        }
                        for row in results
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                for row in results:
                    with st.expander(f"#{row['run_id']} - {row['nome']}"):
                        st.write(row["evidencia"])
                        st.json(row)
            else:
                st.info("Nenhuma startup salva bateu com o filtro interpretado.")

    with tab_briefing:
        st.subheader("Briefing executivo")
        st.caption("Abra uma execucao salva, revise recomendacoes e baixe Markdown ou PDF.")
        recent_runs = list_recent_runs(db_path, limit=limit)
        default_run_id = st.session_state.get("selected_run_id")
        if default_run_id is None and recent_runs:
            default_run_id = recent_runs[0]["run_id"]

        run_id = st.number_input("run_id", min_value=1, value=int(default_run_id or 1), step=1)
        run = get_run(int(run_id), db_path)
        if run is None:
            st.info("Execucao nao encontrada.")
        else:
            profile = run["profile"]
            header_cols = st.columns(4)
            header_cols[0].metric("Startup", run["nome"])
            header_cols[1].metric("Classificacao", run["classificacao"])
            header_cols[2].metric("Maturidade", f"{run['score_maturidade_ia']:.0f}/100")
            header_cols[3].metric("Risco wrapper", f"{run['score_wrapper_risco']:.0f}/100")
            review_status = run.get("review_status", "aprovado")
            if review_status == "rejeitado":
                st.error(f"Briefing rejeitado. Nota: {run.get('review_nota') or 'sem nota'}")
            elif review_status == "pendente":
                st.warning("Briefing pendente de revisao humana; ainda nao esta pronto para envio.")
            else:
                st.success("Briefing aprovado/pronto para exportacao.")

            briefing = run.get("briefing_en") or run.get("briefing_pt") or ""
            filename = f"run-{run['run_id']}-{slugify(run['nome'])}"
            if review_status == "aprovado":
                markdown_export = briefing_markdown(run)
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
            else:
                st.info("A exportacao fica disponivel depois da aprovacao humana.")
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

    with tab_observability:
        st.subheader("Observabilidade do pipeline")
        st.caption("Trace por execucao e metricas agregadas de latencia, erro e modo de execucao por agente.")
        recent_runs = list_recent_runs(db_path, limit=limit)
        if not recent_runs:
            st.info("Nenhuma execucao salva ainda.")
        else:
            default_run_id = st.session_state.get("selected_run_id")
            options = {_format_run_option(run): run["run_id"] for run in recent_runs}
            option_labels = list(options)
            default_index = 0
            if default_run_id in options.values():
                default_index = list(options.values()).index(default_run_id)
            selected_label = st.selectbox("Execucao", option_labels, index=default_index, key="observability-run")
            selected_run_id = int(options[selected_label])
            traces = get_agent_traces(selected_run_id, db_path)

            if not traces:
                st.warning("Esta execucao ainda nao possui trace estruturado salvo.")
            else:
                timeline = [
                    {
                        "ordem": index + 1,
                        "agente": trace["agent"],
                        "inicio": trace["started_at"],
                        "fim": trace["ended_at"],
                        "latencia_ms": trace["latency_ms"],
                        "sucesso": trace["success"],
                        "modo_execucao": trace["modo_execucao"],
                        "tokens_usados": trace["tokens_used"],
                        "custo_estimado_usd": trace["estimated_cost_usd"],
                        "erro": trace["error_message"],
                    }
                    for index, trace in enumerate(traces)
                ]
                total_latency = sum(float(trace["latency_ms"] or 0) for trace in traces)
                error_count = sum(1 for trace in traces if not trace["success"])
                llm_count = sum(1 for trace in traces if trace["modo_execucao"] == "llm")
                fallback_count = sum(1 for trace in traces if str(trace["modo_execucao"]).startswith("fallback"))
                cost_sum = sum(float(trace["estimated_cost_usd"] or 0) for trace in traces)

                metric_cols = st.columns(5)
                metric_cols[0].metric("Agentes", len(traces))
                metric_cols[1].metric("Latencia total", f"{total_latency:.0f} ms")
                metric_cols[2].metric("Erros", error_count)
                metric_cols[3].metric("LLM/Fallback", f"{llm_count}/{fallback_count}")
                metric_cols[4].metric("Custo estimado", f"US$ {cost_sum:.4f}")

                st.dataframe(timeline, hide_index=True, use_container_width=True)
                latency_rows = [
                    {"agente": row["agente"], "latencia_ms": row["latencia_ms"]}
                    for row in timeline
                ]
                st.bar_chart(latency_rows, x="agente", y="latencia_ms")

        st.subheader("Metricas agregadas")
        summary = agent_trace_summary(db_path)
        if summary:
            st.dataframe(summary, hide_index=True, use_container_width=True)
        else:
            st.info("Sem traces agregados ainda.")

    with tab_rag:
        st.subheader("Base RAG NVIDIA")
        st.caption(
            "Aqui voce ingere fontes oficiais, fontes conceituais, casos historicos e textos manuais; "
            "depois reconstroi chunks/embeddings e audita a busca hibrida do agente."
        )
        stats = rag_index_stats(rag_db_path)
        stat_cols = st.columns(5)
        stat_cols[0].metric("Chunks", stats["chunk_count"])
        stat_cols[1].metric("Dimensao", stats["embedding_dimension"])
        stat_cols[2].metric("Docs", stats.get("document_count", 0))
        stat_cols[3].metric("Dedupe", stats.get("deduplicated_count", 0))
        stat_cols[4].metric("Cases", stats["by_type"].get("historical_case", 0))
        st.caption(
            f"Embedding: {stats.get('embedding_backend') or 'nao indexado'} | "
            f"Modelo: {stats.get('embedding_model') or 'nao indexado'}"
        )

        action_cols = st.columns([1, 1, 2])
        if action_cols[0].button("Ingerir corpus RAG", type="primary", use_container_width=True):
            with st.spinner("Ingerindo fontes RAG e reindexando..."):
                manifest = ingest_rag_sources(rag_source_dir)
                index = rebuild_rag_index(rag_db_path, source_dir=rag_source_dir)
                st.json({"ingest": manifest, "index": index})
        if action_cols[1].button("Reindexar", use_container_width=True):
            with st.spinner("Reconstruindo chunks, embeddings e indice hibrido..."):
                st.json(rebuild_rag_index(rag_db_path, source_dir=rag_source_dir))
        if action_cols[2].button("Avaliar golden set", use_container_width=True):
            with st.spinner("Rodando avaliacao do golden set..."):
                st.json(evaluate_rag_golden_set(rag_db_path))

        rag_query = st.text_area(
            "Consulta RAG",
            value="healthtech IA prontuario LGPD compliance clinico",
            height=90,
        )
        rag_limit = st.slider("Top-k", min_value=3, max_value=15, value=7)
        if st.button("Buscar chunks", use_container_width=True):
            results = rag_search(rag_query, db_path=rag_db_path, limit=rag_limit)
            for result in results:
                label = (
                    f"{result['tecnologia']} | {result['document_type']} | "
                    f"score {result['rerank_score']:.3f}"
                )
                with st.expander(label):
                    st.write(
                        {
                            "chunk_id": result["chunk_id"],
                            "categoria": result["category"],
                            "fonte": result["fonte_url"],
                            "semantic_score": round(result["semantic_score"], 4),
                            "bm25_score": round(result["bm25_score"], 4),
                            "metadata_score": round(result["metadata_score"], 4),
                            "hybrid_score": round(result["hybrid_score"], 4),
                            "rerank_score": round(result["rerank_score"], 4),
                        }
                    )
                    st.text(result["chunk_text"])


def launch() -> None:
    dashboard_path = Path(__file__).resolve()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path), *sys.argv[1:]],
        check=False,
    )


if __name__ == "__main__":
    main()
