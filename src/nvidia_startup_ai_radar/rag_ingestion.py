"""Raw corpus ingestion for the local RAG index.

Ingestion is intentionally separate from indexing: this module fetches or
materializes source documents into local JSON snapshots with provenance and
status; ``rag.py`` is responsible for chunking, embeddings and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
import json
import logging
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import urlparse
import urllib.robotparser
import urllib.request

from nvidia_startup_ai_radar.knowledge_base import HISTORICAL_CASES
from nvidia_startup_ai_radar.official_sources import OFFICIAL_NVIDIA_SOURCES, _extract_text
from nvidia_startup_ai_radar.schemas import HistoricalCase, utc_now_iso


logger = logging.getLogger(__name__)

DEFAULT_RAW_SOURCE_DIR = Path("data") / "raw_sources"
USER_AGENT = "NVIDIA-Startup-AI-Radar/0.1 (+research prototype; contact: local)"
MAX_RAW_SOURCE_CHARS = 90000

SourceStatus = Literal["ingested", "failed", "blocked_by_robots", "manual"]


@dataclass(frozen=True)
class RagSourceSpec:
    id: str
    title: str
    collection: str
    document_type: str
    source_url: str
    tecnologia: str
    categoria: str
    sinais_de_gatilho: list[str]
    casos_de_uso_tipicos: list[str]
    problema_que_resolve: str
    descricao_negocio: str
    complexidade_implementacao: Literal["baixa", "media", "alta"] = "media"
    inline_text: str | None = None
    document_priority: float = 1.0


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "source"


def _safe_filename(source_id: str) -> str:
    return f"{_slug(source_id)}.json"


def _metadata_from_spec(spec: RagSourceSpec, collected_at: str) -> dict[str, Any]:
    return {
        "tecnologia": spec.tecnologia,
        "categoria": spec.categoria,
        "problema_que_resolve": spec.problema_que_resolve,
        "descricao_negocio": spec.descricao_negocio,
        "descricao_tecnica": spec.inline_text[:900] if spec.inline_text else "",
        "complexidade_implementacao": spec.complexidade_implementacao,
        "sinais_de_gatilho": spec.sinais_de_gatilho,
        "casos_de_uso_tipicos": spec.casos_de_uso_tipicos,
        "fonte_url": spec.source_url,
        "source_url": spec.source_url,
        "data_ultima_verificacao": collected_at,
        "document_priority": spec.document_priority,
        "collection": spec.collection,
    }


def _payload(
    *,
    spec: RagSourceSpec,
    status: SourceStatus,
    text: str = "",
    collected_at: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    collected_at = collected_at or utc_now_iso()
    text = text.strip()
    metadata = _metadata_from_spec(spec, collected_at)
    if text:
        metadata["descricao_tecnica"] = text[:900]
    result = {
        "schema_version": "rag_raw_source_v1",
        "id": spec.id,
        "title": spec.title,
        "collection": spec.collection,
        "document_type": spec.document_type,
        "source_url": spec.source_url,
        "domain": urlparse(spec.source_url).netloc,
        "status": status,
        "collected_at": collected_at,
        "text": text,
        "text_sha256": sha256(text.encode("utf-8")).hexdigest() if text else None,
        "metadata": metadata,
    }
    if error:
        result["error"] = error
    return result


def _write_payload(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    collection = str(payload.get("collection") or "misc")
    path = Path(output_dir) / collection
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / _safe_filename(str(payload["id"]))
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def _robots_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return True
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        request = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", errors="ignore")
        parser.parse(body.splitlines())
    except Exception as exc:
        logger.warning("Nao foi possivel ler robots.txt de %s: %s", parsed.netloc, exc)
        return True
    return parser.can_fetch(USER_AGENT, url)


def _clean_manual_text(text: str) -> str:
    text = unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _official_source_specs() -> list[RagSourceSpec]:
    specs = [
        RagSourceSpec(
            id=source.id,
            title=source.tecnologia,
            collection="official_nvidia",
            document_type="official_nvidia_doc",
            source_url=source.url,
            tecnologia=source.tecnologia,
            categoria=source.category,
            sinais_de_gatilho=source.sinais_de_gatilho,
            casos_de_uso_tipicos=source.casos_de_uso_tipicos,
            problema_que_resolve=source.problema_que_resolve,
            descricao_negocio=source.descricao_negocio,
            complexidade_implementacao=source.complexidade_implementacao,
            document_priority=1.25,
        )
        for source in OFFICIAL_NVIDIA_SOURCES
    ]
    specs.append(
        RagSourceSpec(
            id="official-nemo-framework",
            title="NVIDIA NeMo Framework",
            collection="official_nvidia",
            document_type="official_nvidia_doc",
            source_url="https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html",
            tecnologia="NVIDIA NeMo",
            categoria="Generative AI framework",
            sinais_de_gatilho=["llm", "fine-tuning", "rag", "agentes", "treinamento", "guardrails"],
            casos_de_uso_tipicos=["LLM customizado", "RAG", "agentes empresariais"],
            problema_que_resolve="Construir, customizar e operar modelos generativos com componentes NVIDIA.",
            descricao_negocio="Ajuda startups com IA generativa profunda a sair de wrapper fino para stack propria.",
            complexidade_implementacao="alta",
            document_priority=1.25,
        )
    )
    return specs


CONCEPTUAL_SOURCE_SPECS: list[RagSourceSpec] = [
    RagSourceSpec(
        id="concept-nvidia-ai-five-layer-cake",
        title="AI Is a 5-Layer Cake",
        collection="conceptual_ai_native",
        document_type="conceptual_source",
        source_url="https://blogs.nvidia.com/blog/ai-5-layer-cake/",
        tecnologia="AI-native infrastructure thesis",
        categoria="AI-native vs wrapper",
        sinais_de_gatilho=["infraestrutura", "modelos", "aplicacoes", "energia", "gpu", "ai factory"],
        casos_de_uso_tipicos=["tese AI-native", "avaliacao de profundidade tecnica"],
        problema_que_resolve="Distinguir aplicacoes que puxam a stack inteira de simples interfaces sobre modelo.",
        descricao_negocio="Da base estrategica para priorizar startups que geram demanda real por compute NVIDIA.",
        document_priority=1.05,
    ),
    RagSourceSpec(
        id="concept-sequoia-generative-ai-act-o1",
        title="Sequoia - Generative AI's Act o1",
        collection="conceptual_ai_native",
        document_type="conceptual_source",
        source_url="https://sequoiacap.com/article/generative-ais-act-o1/",
        tecnologia="Agentic reasoning applications",
        categoria="AI-native vs wrapper",
        sinais_de_gatilho=["reasoning", "agentes", "workflow", "llm", "foundation model"],
        casos_de_uso_tipicos=["agentes", "software vertical", "aplicacoes cognitivas"],
        problema_que_resolve="Avaliar quando a startup entrega arquitetura cognitiva nova em vez de UI fina.",
        descricao_negocio="Ajuda a qualificar defensibilidade na camada de aplicacao e raciocinio.",
        document_priority=1.0,
    ),
    RagSourceSpec(
        id="concept-emergence-ai-native-services",
        title="Emergence - AI-Native Services",
        collection="conceptual_ai_native",
        document_type="conceptual_source",
        source_url="https://www.emcap.com/ai-native-services",
        tecnologia="AI-native services",
        categoria="AI-native vs wrapper",
        sinais_de_gatilho=["servicos ai-native", "workflow", "outcome-based", "data flywheel"],
        casos_de_uso_tipicos=["B2B software", "servicos automatizados", "operacoes"],
        problema_que_resolve="Separar empresas que vendem resultado operacional de ferramentas rasas.",
        descricao_negocio="Apoia o score de maturidade para startups B2B que automatizam trabalho real.",
        document_priority=1.0,
    ),
    RagSourceSpec(
        id="concept-crv-ai-wrapper",
        title="CRV - What is an AI Wrapper",
        collection="conceptual_ai_native",
        document_type="conceptual_source",
        source_url="https://www.crv.com/content/what-is-an-ai-wrapper",
        tecnologia="AI wrapper defensibility",
        categoria="Wrapper risk",
        sinais_de_gatilho=["api externa", "wrapper", "dados proprietarios", "workflow", "moat"],
        casos_de_uso_tipicos=["risco wrapper", "avaliacao defensibilidade", "vertical saas"],
        problema_que_resolve="Classificar risco de produto que depende de API externa sem dados/processo proprio.",
        descricao_negocio="Evita priorizar startups que podem virar feature de um provedor de modelo.",
        document_priority=0.95,
    ),
]


def _historical_case_text(case: HistoricalCase) -> str:
    traits = [
        "data_moat" if case.data_moat else "",
        "infra_propria" if case.infra_propria else "",
        "dependencia_api_externa" if case.dependencia_api_externa else "",
        "setor_regulado" if case.setor_regulado else "",
    ]
    return f"""# {case.empresa}

## Tipo
{case.tipo}

## Setor
{case.setor}

## Caracteristicas
{", ".join(trait for trait in traits if trait) or "Sem caracteristica booleana marcada."}

## O que aconteceu
{case.resumo_o_que_aconteceu}

## Licao estruturada
{case.licao_estruturada}

## Uso no radar
Este caso entra no RAG como comparavel para julgar maturidade AI-native, risco de wrapper,
moat de dados, profundidade de infraestrutura e ajuste ao ecossistema NVIDIA.
"""


def _historical_case_specs() -> list[RagSourceSpec]:
    specs: list[RagSourceSpec] = []
    for case in HISTORICAL_CASES:
        specs.append(
            RagSourceSpec(
                id=f"case-{_slug(case.empresa)}",
                title=case.empresa,
                collection="historical_cases",
                document_type="historical_case",
                source_url=case.fonte_url,
                tecnologia=case.empresa,
                categoria=case.setor,
                sinais_de_gatilho=[
                    case.setor,
                    case.tipo,
                    "data moat" if case.data_moat else "",
                    "infra propria" if case.infra_propria else "",
                    "api externa" if case.dependencia_api_externa else "",
                    "setor regulado" if case.setor_regulado else "",
                ],
                casos_de_uso_tipicos=["case bank", "golden set", "comparacao historica"],
                problema_que_resolve=case.resumo_o_que_aconteceu,
                descricao_negocio=case.licao_estruturada,
                inline_text=_historical_case_text(case),
                document_priority=0.85,
            )
        )
    return specs


GOLDEN_SET_SOURCE_SPECS: list[RagSourceSpec] = [
    RagSourceSpec(
        id="golden-fintech-ai-native-cloudwalk-qi-tech",
        title="Golden set - CloudWalk e QI Tech",
        collection="golden_set",
        document_type="golden_case",
        source_url="docs/guia-completo-do-case.md",
        tecnologia="RAPIDS / NeMo Guardrails / NVIDIA NIM",
        categoria="Fintech regulada",
        sinais_de_gatilho=["fintech", "credito", "fraude", "dados tabulares", "setor regulado"],
        casos_de_uso_tipicos=["credito", "fraude", "pagamentos", "compliance"],
        problema_que_resolve="Fintechs AI-native com dados proprietarios e decisao de negocio dependente de IA.",
        descricao_negocio="Devem recuperar RAPIDS, governanca e caminho Inception/NIM quando houver LLM.",
        inline_text=(
            "# Golden set - CloudWalk e QI Tech\n\n"
            "Pergunta: fintech com dados proprietarios, credito, fraude, regulacao e automacao de decisao "
            "de negocio deve ser classificada como AI-native ou apenas AI-enabled?\n\n"
            "Fontes esperadas no top-5: RAPIDS para dados tabulares e fraude, NeMo Guardrails para governanca "
            "em setor regulado, NVIDIA NIM se houver LLM/API externa, NVIDIA Inception como proxima acao."
        ),
    ),
    RagSourceSpec(
        id="golden-healthtech-laura-oncoai",
        title="Golden set - Laura e OncoAI",
        collection="golden_set",
        document_type="golden_case",
        source_url="docs/guia-completo-do-case.md",
        tecnologia="NVIDIA Clara / NeMo Guardrails",
        categoria="Healthtech regulada",
        sinais_de_gatilho=["healthtech", "clinico", "prontuario", "anvisa", "lgpd", "oncologia"],
        casos_de_uso_tipicos=["imagem medica", "monitoramento clinico", "governanca"],
        problema_que_resolve="Healthtechs com dados clinicos precisam de governanca e stack especializada.",
        descricao_negocio="Deve recuperar Clara, NeMo Guardrails e casos de sucesso/fracasso em saude.",
        inline_text=(
            "# Golden set - Laura e OncoAI\n\n"
            "Pergunta: healthtech com IA clinica, dados sensiveis, prontuario, oncologia ou risco regulatorio "
            "precisa de quais tecnologias NVIDIA?\n\n"
            "Fontes esperadas no top-5: NVIDIA Clara para saude/imagem/life sciences, NeMo Guardrails para "
            "governanca e LGPD, casos Laura e Cydoc como comparaveis."
        ),
    ),
    RagSourceSpec(
        id="golden-computer-vision-noleak-mr-turing",
        title="Golden set - Noleak e Mr. Turing",
        collection="golden_set",
        document_type="golden_case",
        source_url="docs/guia-completo-do-case.md",
        tecnologia="Triton Inference Server / TensorRT-LLM",
        categoria="Computer vision e edge AI",
        sinais_de_gatilho=["computer vision", "visao computacional", "camera", "gpu", "triton", "tensorrt"],
        casos_de_uso_tipicos=["video analytics", "edge AI", "inferencia GPU"],
        problema_que_resolve="Startups com inferencia de visao em producao precisam de serving e otimizacao.",
        descricao_negocio="Deve recuperar Triton, TensorRT/TensorRT-LLM e Inception.",
        inline_text=(
            "# Golden set - Noleak e Mr. Turing\n\n"
            "Pergunta: startup de visao computacional em cameras, edge AI ou seguranca com GPU deve apontar "
            "para qual stack NVIDIA?\n\n"
            "Fontes esperadas no top-5: Triton Inference Server para serving, TensorRT/TensorRT-LLM para "
            "otimizacao, Noleak como case historico e NVIDIA Inception."
        ),
    ),
    RagSourceSpec(
        id="golden-banking-infra-stark-bank",
        title="Golden set - Stark Bank",
        collection="golden_set",
        document_type="golden_case",
        source_url="docs/guia-completo-do-case.md",
        tecnologia="RAPIDS / NVIDIA AI Enterprise",
        categoria="Fintech infra B2B",
        sinais_de_gatilho=["infra bancaria", "enterprise", "compliance", "dados tabulares", "automacao"],
        casos_de_uso_tipicos=["core bancario", "automacao financeira", "enterprise AI"],
        problema_que_resolve="Infra B2B regulada precisa de dados acelerados e plataforma enterprise.",
        descricao_negocio="Deve recuperar RAPIDS, AI Enterprise, Guardrails e Inception.",
        inline_text=(
            "# Golden set - Stark Bank\n\n"
            "Pergunta: fintech de infraestrutura bancaria B2B, com integracao profunda e automacao, tem "
            "moat mais forte que uma UI generica?\n\n"
            "Fontes esperadas no top-5: RAPIDS, NVIDIA AI Enterprise, NeMo Guardrails e caso Stark Bank."
        ),
    ),
    RagSourceSpec(
        id="golden-wrapper-risk-wuri",
        title="Golden set - Wuri",
        collection="golden_set",
        document_type="golden_case",
        source_url="docs/guia-completo-do-case.md",
        tecnologia="NVIDIA NIM / NeMo Guardrails",
        categoria="Wrapper risk",
        sinais_de_gatilho=["wrapper", "api externa", "openai", "pivot", "sem moat"],
        casos_de_uso_tipicos=["risco wrapper", "migracao self-hosted", "governanca"],
        problema_que_resolve="Wrapper com pivots e dependencia de API deve ser tratado como risco alto.",
        descricao_negocio="Deve recuperar fontes conceituais wrapper, Wuri e tecnologias de reducao de dependencia.",
        inline_text=(
            "# Golden set - Wuri\n\n"
            "Pergunta: startup de IA generativa com pivots sucessivos, dependencia de API externa e pouco dado "
            "proprietario deve ser priorizada?\n\n"
            "Fontes esperadas no top-5: Wuri como alerta, CRV/Sequoia/Emergence sobre defensibilidade, "
            "NVIDIA NIM para reduzir dependencia e NeMo Guardrails se houver compliance."
        ),
    ),
    RagSourceSpec(
        id="golden-devtools-codewhisper",
        title="Golden set - CodeWhisper",
        collection="golden_set",
        document_type="golden_case",
        source_url="docs/guia-completo-do-case.md",
        tecnologia="AI wrapper defensibility / NVIDIA NIM",
        categoria="Dev tools",
        sinais_de_gatilho=["dev tools", "feature copiada", "provider", "api externa", "copilot"],
        casos_de_uso_tipicos=["dev tools", "avaliacao de moat", "modelo proprietario"],
        problema_que_resolve="Dev tool que pode virar feature de provedor precisa provar workflow/dado proprio.",
        descricao_negocio="Deve recuperar CodeWhisper, wrapper risk e caminho NIM/Triton se houver inferencia propria.",
        inline_text=(
            "# Golden set - CodeWhisper\n\n"
            "Pergunta: dev tool cuja feature principal pode ser copiada por um modelo/lab tem defensibilidade?\n\n"
            "Fontes esperadas no top-5: CodeWhisper como fracasso, fontes wrapper/AI-native e NVIDIA NIM "
            "quando a tese envolver modelo self-hosted."
        ),
    ),
    RagSourceSpec(
        id="golden-healthtech-go-to-market-cydoc",
        title="Golden set - Cydoc",
        collection="golden_set",
        document_type="golden_case",
        source_url="docs/guia-completo-do-case.md",
        tecnologia="NVIDIA Clara / NVIDIA Inception",
        categoria="Healthtech go-to-market",
        sinais_de_gatilho=["healthtech", "vendas", "integracao", "modelo de negocio", "clinico"],
        casos_de_uso_tipicos=["go-to-market", "saude", "validacao comercial"],
        problema_que_resolve="Tecnologia clinica validada ainda pode falhar por vendas, integracao e negocio.",
        descricao_negocio="Deve recuperar Cydoc, Clara e Inception para proxima acao comercial/tecnica.",
        inline_text=(
            "# Golden set - Cydoc\n\n"
            "Pergunta: healthtech com tecnologia validada, mas risco em vendas, integracao e modelo de negocio "
            "deve receber qual recomendacao?\n\n"
            "Fontes esperadas no top-5: Cydoc como fracasso, NVIDIA Clara para stack tecnica e NVIDIA Inception "
            "para diagnostico, parceiros e GTM."
        ),
    ),
]


def rag_source_specs() -> list[RagSourceSpec]:
    return [
        *_official_source_specs(),
        *CONCEPTUAL_SOURCE_SPECS,
        *_historical_case_specs(),
        *GOLDEN_SET_SOURCE_SPECS,
    ]


def _fetch_spec(spec: RagSourceSpec, max_chars: int, network_enabled: bool) -> dict[str, Any]:
    if spec.inline_text is not None:
        return _payload(spec=spec, status="ingested", text=spec.inline_text)
    if not network_enabled:
        return _payload(spec=spec, status="failed", error="network_disabled")
    if not _robots_allowed(spec.source_url):
        return _payload(spec=spec, status="blocked_by_robots", error="robots.txt disallows fetch")
    try:
        text = _extract_text(spec.source_url)
        if len(text) > max_chars:
            text = text[:max_chars]
        return _payload(spec=spec, status="ingested", text=text)
    except Exception as exc:
        logger.warning("Falha ao ingerir %s: %s", spec.source_url, exc)
        return _payload(spec=spec, status="failed", error=str(exc))


def _manual_payload_from_file(file_path: Path, root: Path) -> dict[str, Any] | None:
    try:
        text = _clean_manual_text(file_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        text = _clean_manual_text(file_path.read_text(encoding="latin-1"))
    if not text:
        return None
    relative = file_path.relative_to(root)
    tecnologia = relative.parts[0] if len(relative.parts) > 1 else file_path.stem
    first_line = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), file_path.stem)
    source_id = f"manual-{_slug(str(relative.with_suffix('')))}"
    collected_at = utc_now_iso()
    spec = RagSourceSpec(
        id=source_id,
        title=first_line[:120],
        collection="manual",
        document_type="manual_source",
        source_url=f"local://raw_sources/{relative.as_posix()}",
        tecnologia=tecnologia,
        categoria="Manual RAG source",
        sinais_de_gatilho=[tecnologia, "manual"],
        casos_de_uso_tipicos=["manual ingestion"],
        problema_que_resolve="Fonte manual adicionada ao corpus RAG quando scraping automatico nao basta.",
        descricao_negocio="Permite manter qualidade do corpus mesmo com paginas bloqueadas, JS pesado ou login.",
        inline_text=text,
        document_priority=1.0,
    )
    return _payload(spec=spec, status="manual", text=text, collected_at=collected_at)


def _load_manual_payloads(source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR) -> list[dict[str, Any]]:
    root = Path(source_dir)
    if not root.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for file_path in sorted(root.rglob("*")):
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        if any(part.startswith(".") for part in file_path.parts):
            continue
        payload = _manual_payload_from_file(file_path, root)
        if payload:
            payloads.append(payload)
    return payloads


def ingest_rag_sources(
    output_dir: str | Path = DEFAULT_RAW_SOURCE_DIR,
    *,
    max_chars: int = MAX_RAW_SOURCE_CHARS,
    network_enabled: bool = True,
) -> dict[str, Any]:
    """Fetch/cache the curated RAG corpus and return an ingestion manifest."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    fetched: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for spec in rag_source_specs():
        payload = _fetch_spec(spec, max_chars=max_chars, network_enabled=network_enabled)
        payloads.append(payload)
        file_path = _write_payload(path, payload)
        row = {
            "id": payload["id"],
            "title": payload["title"],
            "status": payload["status"],
            "url": payload["source_url"],
            "chars": len(payload.get("text") or ""),
            "path": str(file_path),
        }
        if payload["status"] == "ingested":
            fetched.append(row)
        elif payload["status"] == "blocked_by_robots":
            blocked.append({**row, "error": payload.get("error")})
        else:
            failed.append({**row, "error": payload.get("error")})

    manual_payloads = _load_manual_payloads(path)
    for payload in manual_payloads:
        file_path = _write_payload(path, payload)
        fetched.append(
            {
                "id": payload["id"],
                "title": payload["title"],
                "status": payload["status"],
                "url": payload["source_url"],
                "chars": len(payload.get("text") or ""),
                "path": str(file_path),
            }
        )
        payloads.append(payload)

    manifest = {
        "schema_version": "rag_ingestion_manifest_v1",
        "collected_at": utc_now_iso(),
        "source_count": len(rag_source_specs()),
        "manual_count": len(manual_payloads),
        "fetched_count": len(fetched),
        "failed_count": len(failed),
        "blocked_count": len(blocked),
        "output_dir": str(path),
        "fetched": fetched,
        "failed": failed,
        "blocked": blocked,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def list_ingested_source_files(source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR) -> list[Path]:
    path = Path(source_dir)
    if not path.exists():
        return []
    return sorted(file for file in path.rglob("*.json") if file.name != "manifest.json")


def load_ingested_source_payloads(source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for file_path in list_ingested_source_files(source_dir):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("text") and payload.get("status", "ingested") in {"ingested", "manual"}:
            payloads.append(payload)
    payloads.extend(_load_manual_payloads(source_dir))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for payload in payloads:
        key = str(payload.get("id") or payload.get("source_url"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(payload)
    return unique
