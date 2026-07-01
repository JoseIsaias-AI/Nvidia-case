"""Official NVIDIA source fetching for the RAG knowledge base."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from nvidia_startup_ai_radar.schemas import utc_now_iso


DEFAULT_OFFICIAL_SOURCE_DIR = Path("data") / "rag_sources" / "official_nvidia"


@dataclass(frozen=True)
class OfficialSource:
    id: str
    tecnologia: str
    category: str
    url: str
    sinais_de_gatilho: list[str]
    casos_de_uso_tipicos: list[str]
    problema_que_resolve: str
    descricao_negocio: str
    complexidade_implementacao: str = "media"


OFFICIAL_NVIDIA_SOURCES: list[OfficialSource] = [
    OfficialSource(
        id="official-nvidia-nim",
        tecnologia="NVIDIA NIM",
        category="LLM serving",
        url="https://docs.nvidia.com/nim/index.html",
        sinais_de_gatilho=["llm", "api", "self-hosted", "latencia", "custo", "deploy"],
        casos_de_uso_tipicos=["agentes", "copilots", "inferencia LLM"],
        problema_que_resolve="Deploy otimizado de modelos como microservicos de inferencia.",
        descricao_negocio="Ajuda startups a reduzir dependencia de APIs externas e levar IA para producao.",
    ),
    OfficialSource(
        id="official-nvidia-nim-llm",
        tecnologia="NVIDIA NIM",
        category="LLM serving",
        url="https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/overview.html",
        sinais_de_gatilho=["llm", "openai", "api compativel", "inferencia", "producao"],
        casos_de_uso_tipicos=["LLM em producao", "chat corporativo", "agentes"],
        problema_que_resolve="Servir LLMs em producao com microservicos NVIDIA NIM.",
        descricao_negocio="Acelera a implantacao de modelos generativos em cloud, datacenter e workstations.",
    ),
    OfficialSource(
        id="official-nemo-guardrails",
        tecnologia="NeMo Guardrails",
        category="AI governance",
        url="https://docs.nvidia.com/nemo/guardrails/latest/about/rail-types.html",
        sinais_de_gatilho=["guardrails", "compliance", "governanca", "seguranca", "rag"],
        casos_de_uso_tipicos=["agentes seguros", "RAG confiavel", "setores regulados"],
        problema_que_resolve="Aplicar rails de seguranca, topico, dialogo e retrieval em sistemas LLM.",
        descricao_negocio="Ajuda a vender IA com mais controle, confianca e aderencia regulatoria.",
    ),
    OfficialSource(
        id="official-triton",
        tecnologia="Triton Inference Server",
        category="Inference serving",
        url="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html",
        sinais_de_gatilho=["gpu", "inferencia", "throughput", "batching", "mlops"],
        casos_de_uso_tipicos=["visao computacional", "LLM serving", "MLOps"],
        problema_que_resolve="Servir modelos de IA com multiplos backends e alto throughput.",
        descricao_negocio="Melhora confiabilidade operacional e uso de infraestrutura.",
    ),
    OfficialSource(
        id="official-triton-architecture",
        tecnologia="Triton Inference Server",
        category="Inference serving",
        url="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html",
        sinais_de_gatilho=["batching", "scheduler", "model repository", "grpc", "http"],
        casos_de_uso_tipicos=["deploy de modelos", "serving multi-backend"],
        problema_que_resolve="Arquitetura de serving com repositorio de modelos, HTTP/gRPC e schedulers.",
        descricao_negocio="Base tecnica para discutir producao, throughput e operacao de modelos.",
    ),
    OfficialSource(
        id="official-tensorrt-llm",
        tecnologia="TensorRT-LLM",
        category="LLM optimization",
        url="https://docs.nvidia.com/tensorrt-llm/index.html",
        sinais_de_gatilho=["llm", "latencia", "throughput", "gpu", "otimizacao"],
        casos_de_uso_tipicos=["LLM em escala", "inferencia eficiente", "batching"],
        problema_que_resolve="Otimizar inferencia de LLMs em GPUs NVIDIA.",
        descricao_negocio="Pode melhorar economia unitaria em startups com volume de inferencia.",
        complexidade_implementacao="alta",
    ),
    OfficialSource(
        id="official-rapids",
        tecnologia="RAPIDS",
        category="Data processing",
        url="https://docs.nvidia.com/rapids/index.html",
        sinais_de_gatilho=["dados tabulares", "cudf", "cuml", "etl", "analytics"],
        casos_de_uso_tipicos=["fintech", "fraude", "risco de credito", "data science"],
        problema_que_resolve="Acelerar pipelines de data science e ML em GPU.",
        descricao_negocio="Reduz tempo de processamento e experimentacao em times de dados.",
    ),
    OfficialSource(
        id="official-riva",
        tecnologia="NVIDIA Riva",
        category="Speech AI",
        url="https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html",
        sinais_de_gatilho=["voz", "asr", "tts", "transcricao", "call center"],
        casos_de_uso_tipicos=["contact center", "assistentes de voz", "transcricao"],
        problema_que_resolve="Construir aplicacoes Speech AI aceleradas por GPU.",
        descricao_negocio="Habilita automacao de atendimento e produtos de voz em tempo real.",
    ),
    OfficialSource(
        id="official-riva-asr",
        tecnologia="NVIDIA Riva",
        category="Speech AI",
        url="https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html",
        sinais_de_gatilho=["asr", "speech recognition", "streaming", "audio"],
        casos_de_uso_tipicos=["transcricao", "call center", "analytics de voz"],
        problema_que_resolve="Reconhecimento automatico de fala em batch ou streaming.",
        descricao_negocio="Transforma audio em texto para assistencia, compliance e analise.",
    ),
    OfficialSource(
        id="official-clara",
        tecnologia="NVIDIA Clara",
        category="Healthcare AI",
        url="https://docs.nvidia.com/clara/index.html",
        sinais_de_gatilho=["saude", "healthcare", "imagem medica", "genomica", "hospitais"],
        casos_de_uso_tipicos=["healthtech", "imagem medica", "drug discovery"],
        problema_que_resolve="Ferramentas e frameworks acelerados para IA em saude.",
        descricao_negocio="Aumenta fit tecnico em startups de saude e life sciences.",
        complexidade_implementacao="alta",
    ),
    OfficialSource(
        id="official-monai",
        tecnologia="NVIDIA Clara",
        category="Healthcare AI",
        url="https://docs.nvidia.com/clara/monai/index.html",
        sinais_de_gatilho=["monai", "medical imaging", "healthcare", "treinamento"],
        casos_de_uso_tipicos=["imagem medica", "modelos clinicos", "pesquisa medica"],
        problema_que_resolve="Ambiente e ferramentas para desenvolvimento de IA em imagem medica.",
        descricao_negocio="Ajuda healthtechs a desenvolver e padronizar modelos de dominio medico.",
        complexidade_implementacao="alta",
    ),
    OfficialSource(
        id="official-morpheus",
        tecnologia="NVIDIA Morpheus",
        category="Cybersecurity AI",
        url="https://docs.nvidia.com/morpheus/index.html",
        sinais_de_gatilho=["cybersecurity", "telemetria", "logs", "anomalia", "ameacas"],
        casos_de_uso_tipicos=["SIEM", "deteccao de ameacas", "fraude"],
        problema_que_resolve="Pipelines acelerados para deteccao e mitigacao de ameacas.",
        descricao_negocio="Ajuda startups de seguranca a analisar telemetria e logs em escala.",
        complexidade_implementacao="alta",
    ),
    OfficialSource(
        id="official-ai-enterprise",
        tecnologia="NVIDIA AI Enterprise",
        category="Enterprise platform",
        url="https://docs.nvidia.com/ai-enterprise/index.html",
        sinais_de_gatilho=["enterprise", "suporte", "sla", "producao", "governanca"],
        casos_de_uso_tipicos=["enterprise AI", "private cloud", "setores regulados"],
        problema_que_resolve="Plataforma empresarial para desenvolver, implantar e gerenciar IA.",
        descricao_negocio="Reduz risco de implantacao enterprise com suporte e ciclo de vida.",
    ),
    OfficialSource(
        id="official-inception",
        tecnologia="NVIDIA Inception",
        category="Startup program",
        url="https://www.nvidia.com/en-us/startups/",
        sinais_de_gatilho=["startup", "funding", "vc", "go-to-market", "creditos"],
        casos_de_uso_tipicos=["introducao comercial", "parceria", "VC Alliance"],
        problema_que_resolve="Acesso ao ecossistema NVIDIA para startups.",
        descricao_negocio="Oferece ferramentas, treinamento, precos preferenciais, parceiros e investidores.",
        complexidade_implementacao="baixa",
    ),
    OfficialSource(
        id="official-genai-perf",
        tecnologia="NVIDIA GenAI-Perf",
        category="Benchmarking",
        url="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html",
        sinais_de_gatilho=["benchmark", "ttft", "throughput", "latencia", "custo"],
        casos_de_uso_tipicos=["benchmark LLM", "estimativa economica", "comparacao de serving"],
        problema_que_resolve="Medir throughput e latencia de modelos generativos.",
        descricao_negocio="Permite estimar economia com metodologia em vez de numeros inventados.",
    ),
    OfficialSource(
        id="official-llm-cost-blog",
        tecnologia="NVIDIA GenAI-Perf",
        category="Benchmarking",
        url="https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/",
        sinais_de_gatilho=["custo", "llm", "tco", "token", "benchmark"],
        casos_de_uso_tipicos=["estimativa economica", "TCO", "ROI de inferencia"],
        problema_que_resolve="Calcular custo de inferencia LLM com metricas e formulas de TCO.",
        descricao_negocio="Dá base citavel para discutir custo por token e ROI.",
    ),
]


def _safe_filename(source_id: str) -> str:
    return f"{source_id}.json"


def _extract_text(url: str, timeout: int = 25) -> str:
    import requests
    import trafilatura

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "NVIDIA-Startup-AI-Radar/0.1 (+academic research prototype)"},
    )
    response.raise_for_status()
    extracted = trafilatura.extract(
        response.text,
        include_links=False,
        include_tables=True,
        include_formatting=True,
    )
    text = (extracted or "").strip()
    if len(text) >= 200:
        return text

    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", response.text)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = unescape(html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


def fetch_official_sources(
    output_dir: str | Path = DEFAULT_OFFICIAL_SOURCE_DIR,
    max_chars: int = 45000,
) -> dict[str, Any]:
    """Fetch official NVIDIA pages and store local JSON snapshots for RAG."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fetched: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for source in OFFICIAL_NVIDIA_SOURCES:
        try:
            text = _extract_text(source.url)
            if len(text) > max_chars:
                text = text[:max_chars]
            payload = {
                "id": source.id,
                "tecnologia": source.tecnologia,
                "category": source.category,
                "url": source.url,
                "domain": urlparse(source.url).netloc,
                "text": text,
                "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "fetched_at": utc_now_iso(),
                "metadata": {
                    "tecnologia": source.tecnologia,
                    "categoria": source.category,
                    "sinais_de_gatilho": source.sinais_de_gatilho,
                    "casos_de_uso_tipicos": source.casos_de_uso_tipicos,
                    "problema_que_resolve": source.problema_que_resolve,
                    "descricao_negocio": source.descricao_negocio,
                    "descricao_tecnica": text[:700],
                    "complexidade_implementacao": source.complexidade_implementacao,
                    "fonte_url": source.url,
                    "document_priority": 1.2,
                },
            }
            file_path = path / _safe_filename(source.id)
            file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            fetched.append(
                {
                    "id": source.id,
                    "url": source.url,
                    "chars": len(text),
                    "path": str(file_path),
                    "low_content": len(text) < 500,
                }
            )
        except Exception as exc:
            failed.append({"id": source.id, "url": source.url, "error": str(exc)})

    manifest = {
        "fetched_at": utc_now_iso(),
        "source_count": len(OFFICIAL_NVIDIA_SOURCES),
        "fetched_count": len(fetched),
        "failed_count": len(failed),
        "fetched": fetched,
        "failed": failed,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def list_official_source_files(source_dir: str | Path = DEFAULT_OFFICIAL_SOURCE_DIR) -> list[Path]:
    path = Path(source_dir)
    if not path.exists():
        return []
    return sorted(file for file in path.glob("*.json") if file.name != "manifest.json")


def load_official_source_payloads(source_dir: str | Path = DEFAULT_OFFICIAL_SOURCE_DIR) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for file_path in list_official_source_files(source_dir):
        try:
            payloads.append(json.loads(file_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return payloads
