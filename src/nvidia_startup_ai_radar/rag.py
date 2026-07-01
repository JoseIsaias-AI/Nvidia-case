"""Local hybrid RAG index for NVIDIA Startup AI Radar.

The module keeps the public MVP contract stable while upgrading the internals:
raw-source ingestion is loaded from ``rag_ingestion.py``, embeddings prefer a
local sentence-transformers model with SQLite cache, retrieval stays hybrid
(vector + BM25), and reranking is pluggable with deterministic fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b, sha256
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Iterable

from nvidia_startup_ai_radar.knowledge_base import HISTORICAL_CASES, KNOWLEDGE_ENTRIES
from nvidia_startup_ai_radar.official_sources import (
    DEFAULT_OFFICIAL_SOURCE_DIR,
    load_official_source_payloads,
)
from nvidia_startup_ai_radar.rag_ingestion import (
    DEFAULT_RAW_SOURCE_DIR,
    load_ingested_source_payloads,
)
from nvidia_startup_ai_radar.schemas import HistoricalCase, KnowledgeEntry, utc_now_iso


logger = logging.getLogger(__name__)

DEFAULT_RAG_DB_PATH = Path("data") / "radar_rag.sqlite"
DEFAULT_EMBEDDING_CACHE_DB = Path("data") / "rag_embedding_cache.sqlite"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
EMBEDDING_DIMENSION = 384
DEDUPE_SIMILARITY_THRESHOLD = 0.992

_SENTENCE_TRANSFORMER_MODELS: dict[str, Any] = {}
_RERANKER_CACHE: dict[str, "RerankerBackend"] = {}
_WARNED_KEYS: set[str] = set()
_LAST_EMBEDDING_INFO: dict[str, Any] = {
    "requested_backend": os.getenv("RAG_EMBEDDING_BACKEND", "auto"),
    "backend": "deterministic",
    "model": "deterministic-hashing-v1",
    "cache_db": str(DEFAULT_EMBEDDING_CACHE_DB),
    "fallback_reason": "not_used_yet",
}


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    document_type: str
    title: str
    category: str
    source_url: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    document_type: str
    title: str
    category: str
    source_url: str
    text: str
    token_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TextUnit:
    text: str
    section_path: str


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: RagChunk
    tokens: list[str]
    embedding: list[float]
    content_hash: str


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    logger.warning(message, *args)


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return ascii_text.lower()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-]+", normalize_text(text))
        if len(token) > 2
    ]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _embedding_cache_path() -> Path:
    return Path(os.getenv("RAG_EMBEDDING_CACHE_DB", str(DEFAULT_EMBEDDING_CACHE_DB)))


def _connect_embedding_cache(cache_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(cache_path or _embedding_cache_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            backend_id TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (backend_id, text_hash)
        )
        """
    )
    return connection


def _embedding_cache_get(backend_id: str, text_hash: str) -> list[float] | None:
    try:
        with _connect_embedding_cache() as connection:
            row = connection.execute(
                """
                SELECT embedding_json
                FROM embedding_cache
                WHERE backend_id = ? AND text_hash = ?
                """,
                (backend_id, text_hash),
            ).fetchone()
    except sqlite3.Error as exc:
        _warn_once("embedding-cache-read", "Falha ao ler cache de embeddings: %s", exc)
        return None
    if row is None:
        return None
    embedding = _json_loads(row["embedding_json"], None)
    if not isinstance(embedding, list):
        return None
    return [float(value) for value in embedding]


def _embedding_cache_set(backend_id: str, text_hash: str, embedding: list[float]) -> None:
    try:
        with _connect_embedding_cache() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (
                    backend_id,
                    text_hash,
                    dimension,
                    embedding_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (backend_id, text_hash, len(embedding), _json_dumps(embedding), utc_now_iso()),
            )
    except sqlite3.Error as exc:
        _warn_once("embedding-cache-write", "Falha ao gravar cache de embeddings: %s", exc)


def _hash_index(token: str, dimension: int = EMBEDDING_DIMENSION) -> int:
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _deterministic_embed_text(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    tokens = tokenize(text)
    vector = [0.0] * dimension
    if not tokens:
        return vector

    counts = Counter(tokens)
    for token, count in counts.items():
        vector[_hash_index(token, dimension)] += 1.0 + math.log(count)

    for first, second in zip(tokens, tokens[1:]):
        vector[_hash_index(f"{first}_{second}", dimension)] += 0.55

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _neural_embed_text(text: str, model_name: str) -> list[float]:
    model = _SENTENCE_TRANSFORMER_MODELS.get(model_name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _SENTENCE_TRANSFORMER_MODELS[model_name] = model

    encoded = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    if hasattr(encoded, "tolist"):
        vector = encoded.tolist()
    else:
        vector = list(encoded)
    vector = [float(value) for value in vector]
    if not vector:
        raise ValueError("sentence-transformers returned an empty embedding")
    return vector


def _embedding_backend_request() -> tuple[str, str]:
    requested = os.getenv("RAG_EMBEDDING_BACKEND", "auto").strip().lower() or "auto"
    if requested not in {"auto", "neural", "deterministic"}:
        _warn_once(
            "embedding-backend-invalid",
            "RAG_EMBEDDING_BACKEND=%s invalido; usando auto.",
            requested,
        )
        requested = "auto"
    model = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    return requested, model


def embed_text(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    """Embed text with a local neural model, cached, with deterministic fallback."""

    global _LAST_EMBEDDING_INFO
    requested, model_name = _embedding_backend_request()
    text_hash = sha256(text.encode("utf-8")).hexdigest()

    if requested in {"auto", "neural"}:
        backend_id = f"sentence-transformers:{model_name}:normalize:v1"
        cached = _embedding_cache_get(backend_id, text_hash)
        if cached is not None:
            _LAST_EMBEDDING_INFO = {
                "requested_backend": requested,
                "backend": "sentence-transformers",
                "model": model_name,
                "cache_db": str(_embedding_cache_path()),
                "fallback_reason": None,
            }
            return cached
        try:
            embedding = _neural_embed_text(text, model_name)
            _embedding_cache_set(backend_id, text_hash, embedding)
            _LAST_EMBEDDING_INFO = {
                "requested_backend": requested,
                "backend": "sentence-transformers",
                "model": model_name,
                "cache_db": str(_embedding_cache_path()),
                "fallback_reason": None,
            }
            return embedding
        except Exception as exc:
            _warn_once(
                "embedding-neural-fallback",
                "Embedding neural indisponivel (%s). Usando fallback deterministico local.",
                exc,
            )
            fallback_reason = str(exc)
    else:
        fallback_reason = "RAG_EMBEDDING_BACKEND=deterministic"

    backend_id = f"deterministic-hashing:v1:dim{dimension}"
    cached = _embedding_cache_get(backend_id, text_hash)
    if cached is not None:
        _LAST_EMBEDDING_INFO = {
            "requested_backend": requested,
            "backend": "deterministic",
            "model": "deterministic-hashing-v1",
            "cache_db": str(_embedding_cache_path()),
            "fallback_reason": fallback_reason,
        }
        return cached

    embedding = _deterministic_embed_text(text, dimension=dimension)
    _embedding_cache_set(backend_id, text_hash, embedding)
    _LAST_EMBEDDING_INFO = {
        "requested_backend": requested,
        "backend": "deterministic",
        "model": "deterministic-hashing-v1",
        "cache_db": str(_embedding_cache_path()),
        "fallback_reason": fallback_reason,
    }
    return embedding


def embedding_backend_info() -> dict[str, Any]:
    requested, model_name = _embedding_backend_request()
    return {
        **_LAST_EMBEDDING_INFO,
        "requested_backend": requested,
        "requested_model": model_name,
        "cache_db": str(_embedding_cache_path()),
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(limit))


def _knowledge_document(entry: KnowledgeEntry) -> SourceDocument:
    text = f"""# {entry.tecnologia}

## Categoria
{entry.categoria}

## Problema que resolve
{entry.problema_que_resolve}

## Descricao tecnica
{entry.descricao_tecnica}

## Descricao de negocio
{entry.descricao_negocio}

## Complexidade
{entry.complexidade_implementacao}

## Sinais de gatilho
{", ".join(entry.sinais_de_gatilho)}

## Casos de uso tipicos
{", ".join(entry.casos_de_uso_tipicos)}
"""
    metadata = entry.model_dump()
    metadata["document_priority"] = 1.05
    return SourceDocument(
        doc_id=entry.id,
        document_type="knowledge_entry",
        title=entry.tecnologia,
        category=entry.categoria,
        source_url=str(entry.fonte_url),
        text=text,
        metadata=metadata,
    )


def _case_document(case: HistoricalCase) -> SourceDocument:
    traits = [
        "data_moat" if case.data_moat else "",
        "infra_propria" if case.infra_propria else "",
        "dependencia_api_externa" if case.dependencia_api_externa else "",
        "setor_regulado" if case.setor_regulado else "",
    ]
    text = f"""# {case.empresa}

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
"""
    metadata = case.model_dump()
    metadata["document_priority"] = 0.8
    return SourceDocument(
        doc_id=f"case-{re.sub(r'[^a-zA-Z0-9]+', '-', case.empresa.lower()).strip('-')}",
        document_type="historical_case",
        title=case.empresa,
        category=case.setor,
        source_url=case.fonte_url,
        text=text,
        metadata=metadata,
    )


def _methodology_document() -> SourceDocument:
    return SourceDocument(
        doc_id="methodology-economic-estimator",
        document_type="methodology",
        title="Metodologia economica NVIDIA GenAI-Perf",
        category="Benchmarking",
        source_url="docs/guia-completo-do-case.md",
        text=(
            "# Metodologia economica NVIDIA GenAI-Perf\n\n"
            "Nenhum numero de economia ou eficiencia deve entrar no briefing sem "
            "benchmark publico citavel ou premissas explicitas. Para workloads de "
            "inferencia, medir baseline atual e comparar TTFT, throughput, custo "
            "por token, taxa de erro e utilizacao de GPU com GenAI-Perf."
        ),
        metadata={
            "tecnologia": "NVIDIA GenAI-Perf",
            "categoria": "Benchmarking",
            "sinais_de_gatilho": ["custo", "latencia", "throughput", "ttft", "inferencia"],
            "complexidade_implementacao": "media",
            "problema_que_resolve": "Metodologia para estimar custo e latencia sem inventar numeros.",
            "descricao_negocio": "Aumenta credibilidade do briefing economico.",
            "data_ultima_verificacao": "2026-06-30",
            "document_priority": 0.9,
        },
    )


def _official_documents(source_dir: str | Path = DEFAULT_OFFICIAL_SOURCE_DIR) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for payload in load_official_source_payloads(source_dir):
        metadata = payload.get("metadata") or {}
        documents.append(
            SourceDocument(
                doc_id=payload["id"],
                document_type="official_nvidia_doc",
                title=payload.get("tecnologia") or payload["id"],
                category=payload.get("category") or metadata.get("categoria") or "NVIDIA official docs",
                source_url=payload.get("url") or metadata.get("fonte_url") or "local",
                text=(
                    f"# {payload.get('tecnologia') or payload['id']}\n\n"
                    f"## Fonte oficial\n{payload.get('url')}\n\n"
                    f"{payload.get('text', '')}"
                ),
                metadata={
                    **metadata,
                    "fetched_at": payload.get("fetched_at"),
                    "text_sha256": payload.get("text_sha256"),
                    "data_ultima_verificacao": metadata.get("data_ultima_verificacao")
                    or payload.get("fetched_at")
                    or "2026-06-30",
                    "document_priority": metadata.get("document_priority", 1.2),
                },
            )
        )
    return documents


def _ingested_documents(source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for payload in load_ingested_source_payloads(source_dir):
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        metadata = payload.get("metadata") or {}
        source_url = payload.get("source_url") or metadata.get("fonte_url") or "local"
        title = payload.get("title") or metadata.get("tecnologia") or payload.get("id") or "Fonte RAG"
        category = metadata.get("categoria") or payload.get("collection") or "RAG source"
        collected_at = payload.get("collected_at") or utc_now_iso()
        documents.append(
            SourceDocument(
                doc_id=str(payload.get("id") or sha256(text.encode("utf-8")).hexdigest()[:16]),
                document_type=str(payload.get("document_type") or "ingested_source"),
                title=str(title),
                category=str(category),
                source_url=str(source_url),
                text=f"# {title}\n\n## Fonte\n{source_url}\n\n{text}",
                metadata={
                    **metadata,
                    "fonte_url": source_url,
                    "source_url": source_url,
                    "collection": payload.get("collection"),
                    "ingestion_status": payload.get("status", "ingested"),
                    "data_ultima_verificacao": metadata.get("data_ultima_verificacao") or collected_at,
                    "text_sha256": payload.get("text_sha256"),
                    "document_priority": metadata.get("document_priority", 1.0),
                },
            )
        )
    return documents


def seed_documents(source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR) -> list[SourceDocument]:
    """Build source documents in the ingestion priority from the planning docs."""

    ingested_docs = _ingested_documents(source_dir)
    official_docs = [] if ingested_docs else _official_documents(source_dir)
    knowledge_docs = [_knowledge_document(entry) for entry in KNOWLEDGE_ENTRIES]

    seen_doc_ids = {document.doc_id for document in [*ingested_docs, *official_docs, *knowledge_docs]}
    case_docs = []
    for case in HISTORICAL_CASES:
        document = _case_document(case)
        if document.doc_id not in seen_doc_ids:
            case_docs.append(document)
            seen_doc_ids.add(document.doc_id)

    methodology = _methodology_document()
    return [*ingested_docs, *official_docs, *knowledge_docs, *case_docs, methodology]


def _section_path(heading_stack: list[tuple[int, str]]) -> str:
    return " > ".join(text for _, text in heading_stack if text)


def _split_sentences(block: str) -> list[str]:
    block = block.strip()
    if not block:
        return []
    if "\n- " in block or block.startswith("- "):
        return [line.strip() for line in block.splitlines() if line.strip()]
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", block) if sentence.strip()]


def _split_long_unit(unit: TextUnit, max_tokens: int) -> list[TextUnit]:
    if len(tokenize(unit.text)) <= max_tokens:
        return [unit]

    clauses = [part.strip() for part in re.split(r"(?<=[.;:])\s+", unit.text) if part.strip()]
    if len(clauses) <= 1:
        words = unit.text.split()
        pieces = [" ".join(words[index : index + max_tokens]) for index in range(0, len(words), max_tokens)]
        return [TextUnit(piece, unit.section_path) for piece in pieces if piece]

    result: list[TextUnit] = []
    current: list[str] = []
    current_tokens = 0
    for clause in clauses:
        clause_tokens = len(tokenize(clause))
        if current and current_tokens + clause_tokens > max_tokens:
            result.append(TextUnit(" ".join(current), unit.section_path))
            current = []
            current_tokens = 0
        current.append(clause)
        current_tokens += clause_tokens
    if current:
        result.append(TextUnit(" ".join(current), unit.section_path))
    return result


def _paragraph_units(text: str, max_unit_tokens: int = 260) -> list[TextUnit]:
    units: list[TextUnit] = []
    heading_stack: list[tuple[int, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", block)
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = [(existing_level, value) for existing_level, value in heading_stack if existing_level < level]
            heading_stack.append((level, heading))
            units.append(TextUnit(block, _section_path(heading_stack)))
            continue

        path = _section_path(heading_stack)
        for sentence in _split_sentences(block):
            contextual_sentence = f"{path}: {sentence}" if path and not sentence.startswith(path) else sentence
            unit = TextUnit(contextual_sentence, path)
            units.extend(_split_long_unit(unit, max_tokens=max_unit_tokens))
    return units


def _token_tail(text: str, max_tokens: int) -> str:
    words = text.split()
    return " ".join(words[-max_tokens:]) if len(words) > max_tokens else text


def _chunk_metadata(document: SourceDocument, section_paths: list[str], target_tokens: int, max_tokens: int, overlap_tokens: int) -> dict[str, Any]:
    metadata = dict(document.metadata)
    metadata.setdefault("tecnologia", document.title)
    metadata.setdefault("categoria", document.category)
    metadata.setdefault("problema_que_resolve", metadata.get("resumo_o_que_aconteceu") or document.text[:240])
    metadata.setdefault("descricao_tecnica", document.text[:700])
    metadata.setdefault("descricao_negocio", metadata.get("licao_estruturada") or document.text[:360])
    metadata.setdefault("complexidade_implementacao", "media")
    metadata.setdefault("sinais_de_gatilho", [])
    metadata.setdefault("casos_de_uso_tipicos", [])
    metadata.setdefault("fonte_url", document.source_url)
    metadata.setdefault("source_url", document.source_url)
    metadata.setdefault("data_ultima_verificacao", "2026-06-30")
    metadata["chunk_strategy"] = "semantic_section_paragraph_overlap"
    metadata["target_tokens"] = target_tokens
    metadata["max_tokens"] = max_tokens
    metadata["overlap_tokens"] = overlap_tokens
    metadata["overlap_ratio"] = round(overlap_tokens / max(target_tokens, 1), 3)
    metadata["caminho_secao"] = " | ".join(section_paths) if section_paths else document.title
    metadata["section_paths"] = section_paths
    return metadata


def chunk_document(
    document: SourceDocument,
    target_tokens: int = 240,
    max_tokens: int = 300,
    overlap_tokens: int = 45,
) -> list[RagChunk]:
    """Chunk by semantic units with metadata-rich context and bounded overlap."""

    units = _paragraph_units(document.text, max_unit_tokens=max_tokens - 25)
    chunks: list[RagChunk] = []
    current: list[TextUnit] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = "\n".join(unit.text for unit in current).strip()
        section_paths = sorted({unit.section_path for unit in current if unit.section_path})
        contextual_text = (
            f"Documento: {document.title}\n"
            f"Tipo: {document.document_type}\n"
            f"Categoria: {document.category}\n\n"
            f"{body}"
        )
        chunk_index = len(chunks)
        chunks.append(
            RagChunk(
                chunk_id=f"{document.doc_id}::chunk-{chunk_index}",
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                document_type=document.document_type,
                title=document.title,
                category=document.category,
                source_url=document.source_url,
                text=contextual_text,
                token_count=len(tokenize(contextual_text)),
                metadata=_chunk_metadata(document, section_paths, target_tokens, max_tokens, overlap_tokens),
            )
        )
        overlap = _token_tail(body, overlap_tokens)
        last_section = section_paths[-1] if section_paths else document.title
        current = [TextUnit(f"Contexto anterior: {overlap}", last_section)] if overlap else []
        current_tokens = len(tokenize(" ".join(unit.text for unit in current)))

    for unit in units:
        unit_tokens = len(tokenize(unit.text))
        if current and current_tokens + unit_tokens > max_tokens:
            flush()
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= target_tokens:
            flush()
    if current:
        flush()
    return chunks


def build_chunks(
    documents: Iterable[SourceDocument] | None = None,
    source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for document in documents or seed_documents(source_dir):
        chunks.extend(chunk_document(document))
    return chunks


def initialize_rag_store(db_path: str | Path = DEFAULT_RAG_DB_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                document_type TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                source_url TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                tokens_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_lookup
            ON rag_chunks (document_type, title, category)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_source
            ON rag_chunks (source_url, doc_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )


def _canonical_content_hash(text: str) -> str:
    canonical = re.sub(r"\s+", " ", normalize_text(text)).strip()
    return sha256(canonical.encode("utf-8")).hexdigest()


def _embed_and_deduplicate_chunks(chunks: Iterable[RagChunk]) -> tuple[list[EmbeddedChunk], int]:
    indexed: list[EmbeddedChunk] = []
    hashes_by_source: dict[str, set[str]] = {}
    embeddings_by_source: dict[str, list[list[float]]] = {}
    deduplicated_count = 0

    for chunk in chunks:
        tokens = tokenize(chunk.text)
        embedding = embed_text(chunk.text)
        content_hash = _canonical_content_hash(chunk.text)
        source_key = chunk.source_url or chunk.doc_id
        hashes = hashes_by_source.setdefault(source_key, set())
        source_embeddings = embeddings_by_source.setdefault(source_key, [])

        if content_hash in hashes:
            deduplicated_count += 1
            continue
        if any(cosine_similarity(embedding, existing) >= DEDUPE_SIMILARITY_THRESHOLD for existing in source_embeddings):
            deduplicated_count += 1
            continue

        hashes.add(content_hash)
        source_embeddings.append(embedding)
        indexed.append(EmbeddedChunk(chunk=chunk, tokens=tokens, embedding=embedding, content_hash=content_hash))

    return indexed, deduplicated_count


def _set_meta(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    for key, value in values.items():
        connection.execute(
            "INSERT OR REPLACE INTO rag_index_meta (key, value) VALUES (?, ?)",
            (key, str(value) if not isinstance(value, (dict, list)) else _json_dumps(value)),
        )


def rebuild_rag_index(
    db_path: str | Path = DEFAULT_RAG_DB_PATH,
    source_dir: str | Path = DEFAULT_RAW_SOURCE_DIR,
) -> dict[str, Any]:
    """Rebuild the local RAG index from ingested and seeded documents."""

    documents = seed_documents(source_dir)
    chunks = build_chunks(documents)
    embedded_chunks, deduplicated_count = _embed_and_deduplicate_chunks(chunks)
    embedding_info = embedding_backend_info()
    embedding_dimension = len(embedded_chunks[0].embedding) if embedded_chunks else EMBEDDING_DIMENSION
    initialize_rag_store(db_path)
    indexed_at = utc_now_iso()
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM rag_chunks")
        for item in embedded_chunks:
            chunk = item.chunk
            connection.execute(
                """
                INSERT INTO rag_chunks (
                    chunk_id,
                    doc_id,
                    chunk_index,
                    document_type,
                    title,
                    category,
                    source_url,
                    text,
                    token_count,
                    metadata_json,
                    tokens_json,
                    embedding_json,
                    content_hash,
                    indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.chunk_index,
                    chunk.document_type,
                    chunk.title,
                    chunk.category,
                    chunk.source_url,
                    chunk.text,
                    chunk.token_count,
                    _json_dumps(chunk.metadata),
                    _json_dumps(item.tokens),
                    _json_dumps(item.embedding),
                    item.content_hash,
                    indexed_at,
                ),
            )
        _set_meta(
            connection,
            {
                "indexed_at": indexed_at,
                "chunk_count": len(embedded_chunks),
                "document_count": len(documents),
                "candidate_chunk_count": len(chunks),
                "deduplicated_count": deduplicated_count,
                "embedding_dimension": embedding_dimension,
                "embedding_backend": embedding_info.get("backend"),
                "embedding_model": embedding_info.get("model"),
                "embedding_requested_backend": embedding_info.get("requested_backend"),
                "embedding_cache_db": embedding_info.get("cache_db"),
                "embedding_fallback_reason": embedding_info.get("fallback_reason") or "",
                "source_dir": str(source_dir),
            },
        )
    return {
        "db_path": str(db_path),
        "source_dir": str(source_dir),
        "chunk_count": len(embedded_chunks),
        "candidate_chunk_count": len(chunks),
        "deduplicated_count": deduplicated_count,
        "document_count": len(documents),
        "embedding_dimension": embedding_dimension,
        "embedding_backend": embedding_info.get("backend"),
        "embedding_model": embedding_info.get("model"),
        "embedding_cache_db": embedding_info.get("cache_db"),
        "indexed_at": indexed_at,
    }


def rag_index_stats(db_path: str | Path = DEFAULT_RAG_DB_PATH) -> dict[str, Any]:
    initialize_rag_store(db_path)
    with _connect(db_path) as connection:
        chunk_count = connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        rows = connection.execute("SELECT key, value FROM rag_index_meta").fetchall()
        by_type = connection.execute(
            "SELECT document_type, COUNT(*) AS count FROM rag_chunks GROUP BY document_type"
        ).fetchall()
        source_count = connection.execute("SELECT COUNT(DISTINCT source_url) FROM rag_chunks").fetchone()[0]
    meta = {row["key"]: row["value"] for row in rows}
    return {
        "db_path": str(db_path),
        "chunk_count": chunk_count,
        "embedding_dimension": int(meta.get("embedding_dimension", EMBEDDING_DIMENSION)),
        "embedding_backend": meta.get("embedding_backend"),
        "embedding_model": meta.get("embedding_model"),
        "embedding_requested_backend": meta.get("embedding_requested_backend"),
        "embedding_fallback_reason": meta.get("embedding_fallback_reason") or None,
        "embedding_cache_db": meta.get("embedding_cache_db"),
        "indexed_at": meta.get("indexed_at"),
        "document_count": int(meta.get("document_count", 0) or 0),
        "source_count": int(source_count or 0),
        "candidate_chunk_count": int(meta.get("candidate_chunk_count", 0) or 0),
        "deduplicated_count": int(meta.get("deduplicated_count", 0) or 0),
        "source_dir": meta.get("source_dir"),
        "by_type": {row["document_type"]: row["count"] for row in by_type},
    }


def ensure_rag_index(db_path: str | Path = DEFAULT_RAG_DB_PATH) -> None:
    stats = rag_index_stats(db_path)
    if stats["chunk_count"] == 0:
        rebuild_rag_index(db_path)


def _load_chunks(db_path: str | Path) -> list[dict[str, Any]]:
    initialize_rag_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                chunk_id,
                doc_id,
                chunk_index,
                document_type,
                title,
                category,
                source_url,
                text,
                token_count,
                metadata_json,
                tokens_json,
                embedding_json
            FROM rag_chunks
            ORDER BY doc_id, chunk_index
            """
        ).fetchall()
    chunks: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["metadata"] = _json_loads(item.pop("metadata_json"), {})
        item["tokens"] = _json_loads(item.pop("tokens_json"), [])
        item["embedding"] = _json_loads(item.pop("embedding_json"), [])
        chunks.append(item)
    return chunks


def _bm25_scores(query_tokens: list[str], chunks: list[dict[str, Any]]) -> dict[str, float]:
    if not query_tokens or not chunks:
        return {}
    n_docs = len(chunks)
    doc_lengths = [len(chunk["tokens"]) for chunk in chunks]
    avgdl = sum(doc_lengths) / n_docs if n_docs else 0.0
    document_frequency: Counter[str] = Counter()
    token_counts_by_chunk: dict[str, Counter[str]] = {}
    for chunk in chunks:
        counts = Counter(chunk["tokens"])
        token_counts_by_chunk[chunk["chunk_id"]] = counts
        for token in counts:
            document_frequency[token] += 1

    k1 = 1.5
    b = 0.75
    scores: dict[str, float] = {}
    for chunk, doc_length in zip(chunks, doc_lengths):
        counts = token_counts_by_chunk[chunk["chunk_id"]]
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            df = document_frequency[token]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * (doc_length / avgdl if avgdl else 1.0))
            score += idf * (tf * (k1 + 1)) / denominator
        scores[chunk["chunk_id"]] = score
    return scores


def _metadata_boost(query: str, chunk: dict[str, Any]) -> float:
    normalized_query = normalize_text(query)
    metadata = chunk["metadata"]
    boost = 0.0
    title = normalize_text(chunk["title"])
    category = normalize_text(chunk["category"])
    tecnologia = normalize_text(str(metadata.get("tecnologia") or ""))
    if title and title in normalized_query:
        boost += 0.2
    if tecnologia and tecnologia in normalized_query:
        boost += 0.18
    if category and category in normalized_query:
        boost += 0.08
    for trigger in metadata.get("sinais_de_gatilho", []):
        trigger_text = normalize_text(str(trigger))
        if trigger_text and trigger_text in normalized_query:
            boost += 0.08
    for use_case in metadata.get("casos_de_uso_tipicos", []):
        use_case_text = normalize_text(str(use_case))
        if use_case_text and use_case_text in normalized_query:
            boost += 0.05
    if chunk["document_type"] in {"knowledge_entry", "official_nvidia_doc"}:
        boost += 0.05
    try:
        priority = float(metadata.get("document_priority", 1.0))
        boost += min(max(priority - 1.0, 0.0), 0.1)
    except (TypeError, ValueError):
        pass
    return min(boost, 0.45)


class RerankerBackend:
    name = "base"

    def rerank(self, query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class HeuristicRerankerBackend(RerankerBackend):
    name = "heuristic"

    def rerank(self, query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        query_tokens = set(tokenize(query))
        selected: list[dict[str, Any]] = []
        seen_docs: set[str] = set()
        seen_technologies: set[str] = set()
        remaining = [dict(candidate) for candidate in candidates]
        for candidate in remaining:
            chunk_tokens = set(candidate["tokens"])
            exact_overlap = len(query_tokens & chunk_tokens)
            candidate["base_rerank_score"] = candidate["hybrid_score"] + min(exact_overlap * 0.015, 0.18)
            candidate["reranker_backend"] = self.name

        while remaining and len(selected) < limit:
            for candidate in remaining:
                diversity_penalty = 0.0
                if candidate["doc_id"] in seen_docs:
                    diversity_penalty += 0.08
                if str(candidate.get("tecnologia") or candidate["title"]) in seen_technologies:
                    diversity_penalty += 0.35
                candidate["rerank_score"] = candidate["base_rerank_score"] - diversity_penalty
            remaining.sort(key=lambda item: item["rerank_score"], reverse=True)
            chosen = remaining.pop(0)
            selected.append(chosen)
            seen_docs.add(chosen["doc_id"])
            seen_technologies.add(str(chosen.get("tecnologia") or chosen["title"]))
        return selected


class FlashRankRerankerBackend(RerankerBackend):
    name = "flashrank"

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        from flashrank import Ranker, RerankRequest

        self.model_name = model_name
        self._request_cls = RerankRequest
        self._ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        passages = [
            {
                "id": str(index),
                "text": (candidate.get("chunk_text") or candidate.get("text") or "")[:4000],
                "meta": {"chunk_id": candidate["chunk_id"]},
            }
            for index, candidate in enumerate(candidates)
        ]
        results = self._ranker.rerank(self._request_cls(query=query, passages=passages))
        score_by_index: dict[int, float] = {}
        for result in results:
            try:
                score_by_index[int(result["id"])] = float(result.get("score", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

        ranked: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            item = dict(candidate)
            cross_score = score_by_index.get(index, 0.0)
            item["cross_encoder_score"] = cross_score
            item["rerank_score"] = cross_score + (0.05 * item["hybrid_score"])
            item["reranker_backend"] = self.name
            item["reranker_model"] = self.model_name
            ranked.append(item)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:limit]


class CohereRerankerBackend(RerankerBackend):
    name = "cohere"

    def __init__(self, model_name: str | None = None) -> None:
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("RAG_COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY ausente")
        import cohere

        self.model_name = model_name or os.getenv("RAG_COHERE_MODEL", "rerank-multilingual-v3.0")
        self._client = cohere.Client(api_key)

    def rerank(self, query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        documents = [(candidate.get("chunk_text") or candidate.get("text") or "")[:4000] for candidate in candidates]
        response = self._client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=min(limit, len(documents)),
        )
        ranked: list[dict[str, Any]] = []
        for result in response.results:
            item = dict(candidates[int(result.index)])
            item["cross_encoder_score"] = float(result.relevance_score)
            item["rerank_score"] = float(result.relevance_score) + (0.05 * item["hybrid_score"])
            item["reranker_backend"] = self.name
            item["reranker_model"] = self.model_name
            ranked.append(item)
        return ranked[:limit]


def _build_local_reranker() -> RerankerBackend:
    model_name = os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    try:
        return FlashRankRerankerBackend(model_name=model_name)
    except Exception as exc:
        _warn_once(
            "reranker-local-fallback",
            "Reranker local FlashRank indisponivel (%s). Usando reranker heuristico deterministico.",
            exc,
        )
        return HeuristicRerankerBackend()


def get_reranker_backend() -> RerankerBackend:
    requested = os.getenv("RAG_RERANKER", "local").strip().lower() or "local"
    if requested not in {"local", "cohere", "heuristic"}:
        _warn_once("reranker-invalid", "RAG_RERANKER=%s invalido; usando local.", requested)
        requested = "local"
    model_key = os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    cache_key = f"{requested}:{model_key}:{bool(os.getenv('COHERE_API_KEY') or os.getenv('RAG_COHERE_API_KEY'))}"
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    if requested == "heuristic":
        backend: RerankerBackend = HeuristicRerankerBackend()
    elif requested == "cohere":
        try:
            backend = CohereRerankerBackend()
        except Exception as exc:
            _warn_once(
                "reranker-cohere-fallback",
                "RAG_RERANKER=cohere indisponivel (%s). Tentando reranker local.",
                exc,
            )
            backend = _build_local_reranker()
    else:
        backend = _build_local_reranker()

    _RERANKER_CACHE[cache_key] = backend
    return backend


def reranker_backend_info() -> dict[str, Any]:
    requested = os.getenv("RAG_RERANKER", "local").strip().lower() or "local"
    backend = _RERANKER_CACHE.get(
        f"{requested}:{os.getenv('RAG_RERANKER_MODEL', DEFAULT_RERANKER_MODEL)}:{bool(os.getenv('COHERE_API_KEY') or os.getenv('RAG_COHERE_API_KEY'))}"
    )
    return {
        "requested_backend": requested,
        "backend": backend.name if backend else None,
        "model": os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
    }


def _rerank_candidates(query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    backend = get_reranker_backend()
    try:
        return backend.rerank(query, candidates, limit)
    except Exception as exc:
        _warn_once(
            "reranker-runtime-fallback",
            "Falha no reranker %s (%s). Usando reranker heuristico deterministico.",
            backend.name,
            exc,
        )
        return HeuristicRerankerBackend().rerank(query, candidates, limit)


def _rag_search_impl(
    query: str,
    db_path: str | Path = DEFAULT_RAG_DB_PATH,
    limit: int = 7,
    candidate_pool: int = 30,
) -> list[dict[str, Any]]:
    """Search the local RAG index with hybrid vector + BM25 retrieval."""

    ensure_rag_index(db_path)
    chunks = _load_chunks(db_path)
    if not chunks:
        return []
    query_tokens = tokenize(query)
    query_embedding = embed_text(query)
    bm25 = _bm25_scores(query_tokens, chunks)
    max_bm25 = max(bm25.values()) if bm25 else 0.0

    candidates: list[dict[str, Any]] = []
    for chunk in chunks:
        semantic_score = cosine_similarity(query_embedding, chunk["embedding"])
        bm25_score = bm25.get(chunk["chunk_id"], 0.0)
        bm25_normalized = bm25_score / max_bm25 if max_bm25 else 0.0
        metadata_score = _metadata_boost(query, chunk)
        hybrid_score = (0.48 * semantic_score) + (0.37 * bm25_normalized) + (0.15 * metadata_score)
        if hybrid_score <= 0:
            continue
        metadata = chunk["metadata"]
        candidates.append(
            {
                **{key: value for key, value in chunk.items() if key not in {"embedding"}},
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "bm25_normalized": bm25_normalized,
                "metadata_score": metadata_score,
                "hybrid_score": hybrid_score,
                "tecnologia": metadata.get("tecnologia") or metadata.get("empresa") or chunk["title"],
                "problema_que_resolve": metadata.get("problema_que_resolve")
                or metadata.get("resumo_o_que_aconteceu")
                or chunk["text"][:180],
                "descricao_tecnica": metadata.get("descricao_tecnica") or chunk["text"][:240],
                "descricao_negocio": metadata.get("descricao_negocio")
                or metadata.get("licao_estruturada")
                or chunk["text"][:240],
                "complexidade_implementacao": metadata.get("complexidade_implementacao", "media"),
                "fonte_url": metadata.get("fonte_url") or chunk["source_url"],
                "caminho_secao": metadata.get("caminho_secao"),
                "data_ultima_verificacao": metadata.get("data_ultima_verificacao"),
                "chunk_text": chunk["text"],
            }
        )

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return _rerank_candidates(query, candidates[:candidate_pool], limit)


def rag_search(
    query: str,
    db_path: str | Path = DEFAULT_RAG_DB_PATH,
    limit: int = 7,
    candidate_pool: int = 30,
) -> list[dict[str, Any]]:
    """Search the local RAG index, returning an empty set on recoverable failures."""

    try:
        return _rag_search_impl(query, db_path=db_path, limit=limit, candidate_pool=candidate_pool)
    except Exception as exc:
        logger.warning("RAG search failed for query=%r db=%s: %s", query, db_path, exc)
        return []


GOLDEN_RAG_CASES: list[dict[str, Any]] = [
    {
        "startup": "CloudWalk / QI Tech",
        "query": "fintech credito fraude dados tabulares setor regulado AI-native",
        "expected": {"RAPIDS", "NeMo Guardrails", "NVIDIA NIM", "NVIDIA Inception"},
        "expected_source_keywords": {"cloudwalk", "qi tech", "fintech", "rapids"},
    },
    {
        "startup": "Laura / OncoAI",
        "query": "healthtech IA clinica prontuario oncologia LGPD Anvisa governanca",
        "expected": {"NVIDIA Clara", "NeMo Guardrails"},
        "expected_source_keywords": {"laura", "oncoai", "clara", "healthtech", "cydoc"},
    },
    {
        "startup": "Noleak / Mr. Turing",
        "query": "visao computacional cameras GPU Triton TensorRT NVIDIA edge AI",
        "expected": {"Triton Inference Server", "TensorRT-LLM", "NVIDIA Inception"},
        "expected_source_keywords": {"noleak", "mr turing", "triton", "tensorrt"},
    },
    {
        "startup": "Stark Bank",
        "query": "infra bancaria B2B fintech enterprise compliance dados automacao",
        "expected": {"RAPIDS", "NVIDIA AI Enterprise", "NeMo Guardrails", "NVIDIA Inception"},
        "expected_source_keywords": {"stark bank", "ai enterprise", "rapids"},
    },
    {
        "startup": "Wuri",
        "query": "startup wrapper OpenAI API externa pivots risco sem moat",
        "expected": {"NVIDIA NIM", "NeMo Guardrails", "NVIDIA Inception"},
        "expected_source_keywords": {"wuri", "wrapper", "crv", "sequoia", "emergence"},
    },
    {
        "startup": "CodeWhisper",
        "query": "dev tool copilot feature copiada provedor modelo API externa defensibilidade",
        "expected": {"NVIDIA NIM", "Triton Inference Server"},
        "expected_source_keywords": {"codewhisper", "wrapper", "dev tools", "defensibility"},
    },
    {
        "startup": "Cydoc",
        "query": "healthtech tecnologia validada falha vendas integracao modelo de negocio",
        "expected": {"NVIDIA Clara", "NVIDIA Inception", "NeMo Guardrails"},
        "expected_source_keywords": {"cydoc", "clara", "inception", "healthtech"},
    },
]


def _is_relevant_result(result: dict[str, Any], expected: set[str], expected_source_keywords: set[str]) -> bool:
    tecnologia = str(result.get("tecnologia") or "")
    tecnologia_norm = normalize_text(tecnologia)
    haystack = normalize_text(
        " ".join(
            [
                tecnologia,
                str(result.get("title") or ""),
                str(result.get("category") or ""),
                str(result.get("fonte_url") or ""),
                str(result.get("chunk_text") or "")[:1200],
            ]
        )
    )
    for expected_technology in expected:
        expected_norm = normalize_text(expected_technology)
        if expected_norm == tecnologia_norm or expected_norm in haystack:
            return True
    return any(normalize_text(keyword) in haystack for keyword in expected_source_keywords)


def evaluate_rag_golden_set(db_path: str | Path = DEFAULT_RAG_DB_PATH) -> dict[str, Any]:
    """Evaluate retrieval quality against the expanded planning golden set."""

    evaluations: list[dict[str, Any]] = []
    hits_at_5 = 0
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in GOLDEN_RAG_CASES:
        results = rag_search(case["query"], db_path=db_path, limit=5)
        expected = set(case["expected"])
        expected_keywords = set(case["expected_source_keywords"])
        result_rows: list[dict[str, Any]] = []
        relevant_count = 0
        rr = 0.0
        for index, result in enumerate(results, start=1):
            relevant = _is_relevant_result(result, expected, expected_keywords)
            relevant_count += int(relevant)
            if relevant and rr == 0.0:
                rr = 1 / index
            result_rows.append(
                {
                    "rank": index,
                    "tecnologia": result.get("tecnologia"),
                    "document_type": result.get("document_type"),
                    "chunk_id": result.get("chunk_id"),
                    "fonte_url": result.get("fonte_url"),
                    "relevant": relevant,
                }
            )
        hit = relevant_count > 0
        hits_at_5 += int(hit)
        precisions.append(relevant_count / 5)
        reciprocal_ranks.append(rr)
        evaluations.append(
            {
                "startup": case["startup"],
                "query": case["query"],
                "expected": sorted(expected),
                "expected_source_keywords": sorted(expected_keywords),
                "retrieved": result_rows,
                "hit_at_5": hit,
                "precision_at_5": relevant_count / 5,
                "reciprocal_rank": rr,
            }
        )

    precision_at_5 = sum(precisions) / len(precisions) if precisions else 0.0
    hit_rate_at_5 = hits_at_5 / len(GOLDEN_RAG_CASES) if GOLDEN_RAG_CASES else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    stats = rag_index_stats(db_path)
    warning = None
    if hit_rate_at_5 >= 0.99 or mrr >= 0.99:
        warning = (
            "Metricas quase perfeitas: revisar se o golden set esta facil demais "
            "ou se o corpus ainda esta pequeno/obvio."
        )
    elif stats["chunk_count"] < 50:
        warning = "Corpus RAG pequeno para avaliacao robusta; ingira fontes reais antes de confiar nas metricas."

    return {
        "cases": evaluations,
        "precision_at_5": precision_at_5,
        "precision_at_5_proxy": hit_rate_at_5,
        "hit_rate_at_5": hit_rate_at_5,
        "mrr": mrr,
        "case_count": len(GOLDEN_RAG_CASES),
        "warning": warning,
        "index_stats": {
            "chunk_count": stats["chunk_count"],
            "document_count": stats["document_count"],
            "embedding_backend": stats.get("embedding_backend"),
            "embedding_model": stats.get("embedding_model"),
            "deduplicated_count": stats.get("deduplicated_count"),
        },
    }
