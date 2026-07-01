"""Outbound startup discovery from public web sources.

The goal is not to replace paid datasets such as Crunchbase or Dealroom. This
module gives the project a practical public-web discovery layer that follows the
planning docs: credible sources, evidence snippets, collection timestamps,
deduplication, and explicit signals for NVIDIA fit or competitor stack.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

from nvidia_startup_ai_radar.schemas import utc_now_iso
from nvidia_startup_ai_radar.scraping import fetch_public_page


logger = logging.getLogger(__name__)

DEFAULT_DISCOVERY_OUTPUT = Path("data") / "discovery" / "startup_candidates.jsonl"
DEFAULT_DISCOVERY_DB_PATH = Path("data") / "radar_discovery.sqlite"
USER_AGENT = "NVIDIA-Startup-AI-Radar/0.1 (+public startup discovery research)"
YC_AI_INDUSTRY_URL = "https://www.ycombinator.com/companies/industry/artificial-intelligence"
AWS_STARTUPS_BLOG_URL = "https://aws.amazon.com/blogs/startups/"
NVIDIA_INCEPTION_SHOWCASE_URL = "https://www.nvidia.com/pt-br/startups/showcase/"
CASE_SOURCE_DOMAINS = {
    "StartSe": "startse.com",
    "Distrito": "distrito.me",
    "Endeavor Brasil": "endeavor.org.br",
    "ABStartups": "abstartups.com.br",
    "Cubo Itau": "cubo.network",
    "Brazil Journal": "braziljournal.com",
    "NeoFeed": "neofeed.com.br",
    "Startups.com.br": "startups.com.br",
    "Exame": "exame.com",
    "Valor": "valor.globo.com",
    "PEGN": "revistapegn.globo.com",
}
YC_DIRECT_INDUSTRY_URLS = [
    YC_AI_INDUSTRY_URL,
    "https://www.ycombinator.com/companies/industry/generative-ai",
    "https://www.ycombinator.com/companies/industry/machine-learning",
    "https://www.ycombinator.com/companies/industry/computer-vision",
    "https://www.ycombinator.com/companies/industry/robotics",
    "https://www.ycombinator.com/companies/industry/developer-tools",
]


@dataclass(frozen=True, slots=True)
class DiscoveryCampaign:
    key: str
    label: str
    description: str
    queries: tuple[str, ...]


COMPETITOR_STACK_QUERIES = (
    'site:aws.amazon.com/blogs/startups "Amazon Bedrock" startup AI',
    'site:aws.amazon.com/blogs/startups "SageMaker" startup "machine learning"',
    'site:cloud.google.com/customers "Vertex AI" startup',
    'site:cloud.google.com/customers "Gemini" startup AI',
    'site:customers.microsoft.com "Azure OpenAI" startup AI',
    'site:microsoft.com "Azure OpenAI" startup "case study"',
    '"OpenAI" "Anthropic" "startup" "cost" "latency"',
    '"Amazon Bedrock" "startup" "generative AI"',
    '"Vertex AI" "startup" "generative AI"',
    '"Azure OpenAI" "startup" "generative AI"',
)


AI_FRAMEWORK_QUERIES = (
    'site:ycombinator.com/companies "PyTorch" AI startup',
    'site:ycombinator.com/companies "TensorFlow" AI startup',
    'site:ycombinator.com/companies "JAX" "machine learning"',
    '"PyTorch" "GPU" "startup" "AI"',
    '"TensorFlow" "computer vision" "startup"',
    '"Hugging Face" "startup" "LLM"',
    '"RAG" "startup" "LLM" "enterprise"',
    '"agentic AI" "startup" "workflow"',
    '"MLOps" "startup" "model serving"',
    '"vector database" "startup" "AI"',
)


NVIDIA_FIT_QUERIES = (
    'site:nvidia.com/en-us/startups AI startup',
    'site:nvidia.com/pt-br/startups AI startup',
    'site:developer.nvidia.com/blog startup "TensorRT"',
    'site:developer.nvidia.com/blog startup "Triton Inference Server"',
    'site:developer.nvidia.com/blog startup "CUDA"',
    '"NVIDIA Inception" "startup" "AI"',
    '"NVIDIA GPU" "startup" "inference"',
    '"NVIDIA Jetson" "startup" "computer vision"',
    '"TensorRT" "startup" "LLM"',
    '"Triton Inference Server" "startup"',
)


CAREERS_INTENT_QUERIES = (
    '"startup" "careers" "ML engineer" "GPU"',
    '"startup" "jobs" "machine learning engineer" "inference"',
    '"startup" "careers" "MLOps" "model serving"',
    '"startup" "hiring" "research scientist" "deep learning"',
    '"startup" "jobs" "CUDA" "PyTorch"',
    '"startup" "careers" "latency" "throughput" "LLM"',
    '"startup" "jobs" "computer vision engineer" "edge"',
)


SECTOR_QUERIES = (
    '"healthtech" "AI" "startup" "clinical" "HIPAA"',
    '"healthtech" "LLM" "startup" "medical imaging"',
    '"fintech" "AI" "startup" "fraud" "credit"',
    '"cybersecurity" "AI" "startup" "LLM"',
    '"robotics" "AI" "startup" "simulation"',
    '"computer vision" "manufacturing" "startup" "inspection"',
    '"voice AI" "call center" "startup" "transcription"',
    '"autonomous vehicles" "startup" "computer vision" "GPU"',
    '"drug discovery" "foundation model" "startup"',
    '"energy" "AI" "startup" "optimization"',
)


AI_NATIVE_QUALITY_QUERIES = (
    '"proprietary dataset" "AI startup"',
    '"foundation model" "startup" "training"',
    '"AI infrastructure" "startup" "inference"',
    '"model deployment" "startup" "enterprise"',
    '"fine-tuning" "startup" "LLM"',
    '"real-time inference" "startup" "GPU"',
    '"edge AI" "startup" "computer vision"',
)


WRAPPER_RISK_QUERIES = (
    '"chat with PDF" "startup"',
    '"GPT wrapper" "startup"',
    '"AI content generator" "startup" "OpenAI"',
)


YC_DIRECTORY_QUERIES = (
    'site:ycombinator.com/companies "Artificial Intelligence" startup',
    'site:ycombinator.com/companies "Generative AI" startup',
    'site:ycombinator.com/companies "Machine Learning" startup',
    'site:ycombinator.com/companies "Computer Vision" startup',
    'site:ycombinator.com/companies "Robotics" startup',
)


DEFAULT_DISCOVERY_QUERIES = list(
    dict.fromkeys(
        [
            *YC_DIRECTORY_QUERIES,
            *COMPETITOR_STACK_QUERIES,
            *AI_FRAMEWORK_QUERIES,
            *NVIDIA_FIT_QUERIES,
            *CAREERS_INTENT_QUERIES,
            *SECTOR_QUERIES,
            *AI_NATIVE_QUALITY_QUERIES,
            *WRAPPER_RISK_QUERIES,
        ]
    )
)


DISCOVERY_CAMPAIGNS = {
    "full": DiscoveryCampaign(
        key="full",
        label="Radar completo",
        description="Roda todas as fontes e queries do planejamento: concorrentes, NVIDIA fit, frameworks, careers, setores e risco wrapper.",
        queries=tuple(DEFAULT_DISCOVERY_QUERIES),
    ),
    "competitors": DiscoveryCampaign(
        key="competitors",
        label="Stack concorrente",
        description="Detecta startups usando Bedrock, Vertex AI, Azure OpenAI, OpenAI, Anthropic, Gemini e similares.",
        queries=COMPETITOR_STACK_QUERIES,
    ),
    "nvidia_fit": DiscoveryCampaign(
        key="nvidia_fit",
        label="Fit NVIDIA",
        description="Busca sinais de GPU, CUDA, TensorRT, Triton, Jetson, Inception e ecossistema NVIDIA.",
        queries=NVIDIA_FIT_QUERIES,
    ),
    "frameworks": DiscoveryCampaign(
        key="frameworks",
        label="Frameworks de IA",
        description="Busca startups com PyTorch, TensorFlow, JAX, Hugging Face, RAG, MLOps e model serving.",
        queries=AI_FRAMEWORK_QUERIES,
    ),
    "careers": DiscoveryCampaign(
        key="careers",
        label="Vagas tecnicas",
        description="Busca sinais de intencao de contratacao: ML engineer, GPU, inference, MLOps e research scientist.",
        queries=CAREERS_INTENT_QUERIES,
    ),
    "sectors": DiscoveryCampaign(
        key="sectors",
        label="Setores prioritarios",
        description="Busca verticais relevantes: saude, fintech, cybersecurity, robotica, manufatura, voz e energia.",
        queries=SECTOR_QUERIES,
    ),
    "ai_native": DiscoveryCampaign(
        key="ai_native",
        label="AI-native forte",
        description="Busca sinais de dataset proprietario, foundation models, treinamento, inferencia em tempo real e edge AI.",
        queries=AI_NATIVE_QUALITY_QUERIES,
    ),
    "wrapper_risk": DiscoveryCampaign(
        key="wrapper_risk",
        label="Risco wrapper",
        description="Busca sinais de produto raso em cima de API externa para triagem negativa.",
        queries=WRAPPER_RISK_QUERIES,
    ),
}


def campaign_queries(campaign: str = "full") -> list[str]:
    selected = DISCOVERY_CAMPAIGNS.get(campaign, DISCOVERY_CAMPAIGNS["full"])
    return list(selected.queries)


def build_theme_discovery_queries(theme: str) -> list[str]:
    """Build source-scoped outbound queries for a user theme or sector."""

    topic = " ".join(theme.split()).strip()
    if not topic:
        return campaign_queries("full")
    base_queries = [
        f'"{topic}" startup IA Brasil',
        f'"{topic}" "inteligencia artificial" startup Brasil',
        f'"{topic}" "machine learning" startup Brasil',
        f'"{topic}" "AI-native" startup Brasil',
        f'"{topic}" "PyTorch" OR "TensorFlow" startup Brasil',
        f'"{topic}" GPU inferencia startup Brasil',
        f'"{topic}" "Amazon Bedrock" OR "Vertex AI" OR "Azure OpenAI" startup',
    ]
    source_queries: list[str] = []
    for label, domain in CASE_SOURCE_DOMAINS.items():
        source_queries.extend(
            [
                f'site:{domain} "{topic}" startup IA',
                f'site:{domain} "{topic}" "inteligencia artificial"',
                f'site:{domain} "{topic}" "machine learning"',
            ]
        )
        if label in {"Brazil Journal", "NeoFeed", "Exame", "Valor"}:
            source_queries.append(f'site:{domain} "{topic}" startup captacao IA')
    return list(dict.fromkeys([*base_queries, *source_queries]))


SOURCE_CREDIBILITY_BOOSTS = {
    "nvidia.com": 18,
    "developer.nvidia.com": 18,
    "ycombinator.com": 15,
    "aws.amazon.com": 14,
    "cloud.google.com": 14,
    "customers.microsoft.com": 14,
    "microsoft.com": 12,
    "startse.com": 12,
    "distrito.me": 12,
    "endeavor.org.br": 11,
    "abstartups.com.br": 11,
    "cubo.network": 11,
    "braziljournal.com": 10,
    "neofeed.com.br": 10,
    "startups.com.br": 10,
    "exame.com": 8,
    "valor.globo.com": 8,
    "revistapegn.globo.com": 7,
    "openstartups.net": 10,
    "wellfound.com": 8,
    "producthunt.com": 6,
    "github.com": 5,
}

TRUSTED_STARTUP_SOURCE_TYPES = {
    "yc_directory",
    "aws_startups_blog",
    "nvidia_inception_showcase",
}

NON_STARTUP_DOMAINS = {
    "wikipedia.org",
    "pt.wikipedia.org",
    "en.wikipedia.org",
    "pytorch.org",
    "tensorflow.org",
    "docs.python.org",
    "scikit-learn.org",
    "kubernetes.io",
    "github.com",
    "gitlab.com",
    "alura.com.br",
    "ufc.br",
}

GENERIC_AI_ENTITY_NAMES = {
    "ai",
    "agentic ai",
    "hugging face",
    "mlops",
    "openai",
    "pytorch",
    "tensorflow",
    "tensorflow documentation",
    "pytorch documentation",
    "tutorial pytorch",
}

NON_STARTUP_TITLE_PATTERNS = (
    r"\bwikipedia\b",
    r"\benciclop[eé]dia\b",
    r"\btutorial\b",
    r"\bguia\b",
    r"\bcomo instalar\b",
    r"\bcomo implementar\b",
    r"\bo que [eé]\b",
    r"\bwhat is\b",
    r"\bdefinition\b",
    r"\bexplained\b",
    r"\bexplica[cç][aã]o\b",
    r"\bbenef[ií]cios\b",
    r"\bdocumentation\b",
    r"\bdocs?\b",
    r"\bgithub\s*-\s*(pytorch|tensorflow)\b",
)

STARTUP_CONTEXT_TERMS = (
    "startup",
    "startups",
    "empresa",
    "companhia",
    "company",
    "companies",
    "founder",
    "founders",
    "founded",
    "fundada",
    "fundacao",
    "fundacao",
    "funding",
    "captacao",
    "captacao",
    "rodada",
    "seed",
    "series a",
    "series b",
    "venture",
    "vc",
    "portfolio",
    "cliente",
    "clientes",
    "customer",
    "case study",
    "customer story",
    "y combinator",
    "inception",
    "pitch",
)


SIGNAL_GROUPS = {
    "nvidia_fit": {
        "nvidia": 16,
        "cuda": 14,
        "gpu": 10,
        "tensor core": 10,
        "tensorrt": 18,
        "tensorrt-llm": 20,
        "triton inference server": 18,
        "nvidia nim": 20,
        "nemo guardrails": 16,
        "rapids": 12,
        "riva": 12,
        "clara": 14,
        "morpheus": 14,
        "inception": 10,
    },
    "ai_framework": {
        "ia": 4,
        "inteligencia artificial": 7,
        "inteligência artificial": 7,
        "aprendizado de maquina": 6,
        "aprendizado de máquina": 6,
        "artificial intelligence": 8,
        "generative ai": 8,
        "ai infrastructure": 9,
        "pytorch": 10,
        "tensorflow": 10,
        "jax": 8,
        "hugging face": 8,
        "transformer": 7,
        "llm": 7,
        "computer vision": 8,
        "deep learning": 8,
        "machine learning": 6,
        "fine-tuning": 8,
        "inference": 8,
        "mlops": 8,
        "vector database": 5,
        "rag": 6,
        "agentic": 5,
    },
    "competitor_stack": {
        "amazon bedrock": 13,
        "aws bedrock": 13,
        "sagemaker": 10,
        "vertex ai": 13,
        "google cloud ai": 8,
        "azure openai": 13,
        "azure ai": 8,
        "openai": 8,
        "anthropic": 8,
        "claude": 6,
        "gemini": 7,
        "mistral": 5,
        "cohere": 5,
    },
    "maturity": {
        "ml engineer": 8,
        "machine learning engineer": 9,
        "research scientist": 8,
        "model serving": 9,
        "model deployment": 8,
        "training": 6,
        "proprietary model": 9,
        "dataset": 6,
        "latency": 7,
        "throughput": 7,
        "production": 6,
        "enterprise": 5,
        "hipaa": 6,
        "lgpd": 6,
        "soc 2": 6,
    },
    "wrapper_risk": {
        "chat with pdf": -8,
        "gpt wrapper": -12,
        "prompt wrapper": -10,
        "no-code chatbot": -6,
        "content generator": -4,
    },
}


@dataclass(slots=True)
class SearchResult:
    query: str
    title: str
    url: str
    snippet: str
    source_type: str = "web_search"
    company_website: str = ""
    location: str = ""
    team_size: int | None = None


@dataclass(slots=True)
class DiscoveryCandidate:
    name: str
    url: str
    source_domain: str
    source_query: str
    title: str
    snippet: str
    evidence_excerpt: str
    score: float
    nvidia_signals: list[str]
    ai_framework_signals: list[str]
    competitor_stack_signals: list[str]
    maturity_signals: list[str]
    wrapper_risk_signals: list[str]
    collected_at: str
    source_type: str = "web_search"
    company_website: str = ""
    location: str = ""
    team_size: int | None = None
    quality_tier: str = "lead"
    recommended_action: str = ""
    analysis_query: str = ""


def normalize_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    if domain.endswith(".googleusercontent.com"):
        return "googleusercontent.com"
    return domain


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def decode_duckduckgo_url(url: str) -> str:
    parsed = urlparse(html.unescape(url))
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    if "bing.com" in parsed.netloc and parsed.path.startswith("/ck/"):
        target = parse_qs(parsed.query).get("u", [""])[0]
        if target.startswith("a1"):
            encoded = target[2:]
            encoded += "=" * (-len(encoded) % 4)
            try:
                return base64.urlsafe_b64decode(encoded).decode("utf-8")
            except Exception:
                pass
    if url.startswith("//"):
        return f"https:{url}"
    return html.unescape(url)


def _parse_bing_results(query: str, html_body: str, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    blocks = re.split(r'<li[^>]+class="[^"]*\bb_algo\b[^"]*"', html_body)
    for block in blocks[1:]:
        link_match = re.search(r"<h2[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, re.I | re.S)
        if not link_match:
            continue
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.I | re.S)
        title = clean_text(link_match.group(2))
        result_url = decode_duckduckgo_url(link_match.group(1))
        snippet = clean_text(snippet_match.group(1)) if snippet_match else ""
        if title and result_url.startswith(("http://", "https://")):
            results.append(SearchResult(query=query, title=title, url=result_url, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def _search_bing(query: str, max_results: int, timeout: int) -> list[SearchResult]:
    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return _parse_bing_results(query, response.text, max_results)


def search_web(query: str, max_results: int = 8, timeout: int = 4) -> list[SearchResult]:
    """Search the public web and parse organic results."""

    results: list[SearchResult] = []
    try:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        html_body = response.text
        blocks = re.split(r'<div[^>]+class="[^"]*result[^"]*"', html_body)
        for block in blocks[1:]:
            link_match = re.search(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not link_match:
                continue
            snippet_match = re.search(
                r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
                r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            title = clean_text(link_match.group(2))
            result_url = decode_duckduckgo_url(link_match.group(1))
            snippet_html = next((group for group in (snippet_match.groups() if snippet_match else []) if group), "")
            snippet = clean_text(snippet_html)
            if title and result_url.startswith(("http://", "https://")):
                results.append(SearchResult(query=query, title=title, url=result_url, snippet=snippet))
            if len(results) >= max_results:
                break
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for query=%r: %s", query, exc)
        results = []
    if results:
        return results
    try:
        return _search_bing(query, max_results=max_results, timeout=timeout)
    except Exception as exc:
        logger.warning("Bing search failed for query=%r: %s", query, exc)
        return []


def fetch_page_text(url: str, timeout: int = 12) -> str:
    """Fetch a candidate page through the shared layered scraper."""

    page = fetch_public_page(url, timeout=timeout)
    if not page.scrape_success:
        logger.warning("Discovery page fetch failed for %s: %s", url, page.failure_reason)
    return clean_text(page.text)


def _slug_to_name(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-") if part)


def _clean_candidate_title(title: str) -> str:
    title = re.sub(r"\s+\|\s+.*$", "", title)
    title = re.sub(r"\s+-\s+(Y Combinator|AWS|Google Cloud|Microsoft|NVIDIA).*$", "", title, flags=re.I)
    title = re.sub(r"^(Case Study|Customer Story|Startup Spotlight):\s*", "", title, flags=re.I)
    return title.strip(" -:|")


def extract_candidate_name(title: str, url: str) -> str:
    parsed = urlparse(url)
    if "ycombinator.com" in parsed.netloc and "/companies/" in parsed.path:
        parts = [part for part in parsed.path.split("/") if part]
        slug = parts[1] if len(parts) >= 2 and parts[0] == "companies" else ""
        is_job_page = len(parts) >= 3 and parts[2] == "jobs"
        if slug and slug not in {"companies", "industry"}:
            cleaned = _clean_candidate_title(title)
            if cleaned and not is_job_page and cleaned.lower() != slug:
                return cleaned[:90]
            return _slug_to_name(slug)
    title = _clean_candidate_title(title)
    return title[:90] or normalize_domain(url)


def extract_aws_startup_name(title: str) -> str | None:
    patterns = [
        r"^([A-Z][A-Za-z0-9 ._-]{2,60})[’']s\b",
        r"^How\s+([A-Z][A-Za-z0-9 ._-]{2,60}?)\s+(?:helps|uses|built|is|boosts|surfaces)\b",
        r"^How\s+([a-z][A-Za-z0-9 ._-]{2,60}?)\s+(?:helps|uses|built|is|boosts|surfaces)\b",
        r"\bHow\s+([A-Z][A-Za-z0-9 ._-]{2,60}?)\s+(?:helps|uses|built|is|boosts|surfaces)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            candidate = match.group(1).strip(" -:.,")
            if candidate.lower() not in {"aws", "amazon", "startup", "startups"}:
                return candidate[:90]
    return None


def _domain_matches(domain: str, blocked_domain: str) -> bool:
    return domain == blocked_domain or domain.endswith(f".{blocked_domain}")


def _normalized_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def has_startup_context(text: str, source_type: str = "web_search", url: str = "") -> bool:
    """Return True when the result has company/startup context, not just a topic."""

    if source_type in TRUSTED_STARTUP_SOURCE_TYPES:
        return True
    parsed = urlparse(url)
    if "ycombinator.com" in parsed.netloc and "/companies/" in parsed.path:
        return True
    lowered = text.lower()
    return any(term in lowered for term in STARTUP_CONTEXT_TERMS)


def has_company_identity(name: str, url: str, source_type: str = "web_search") -> bool:
    """Allow terse company cards such as 'Patronus AI' or 'Coactive'."""

    if source_type in TRUSTED_STARTUP_SOURCE_TYPES:
        return True
    domain = normalize_domain(url)
    if any(_domain_matches(domain, blocked_domain) for blocked_domain in NON_STARTUP_DOMAINS):
        return False
    normalized = _normalized_name(name)
    if not normalized or normalized in GENERIC_AI_ENTITY_NAMES:
        return False
    if any(re.search(pattern, name.lower()) for pattern in NON_STARTUP_TITLE_PATTERNS):
        return False
    words = normalized.split()
    return len(words) <= 4 and domain.endswith((".ai", ".io", ".com", ".co", ".app", ".dev"))


def is_non_startup_search_result(result: SearchResult, combined_text: str = "") -> bool:
    """Reject tutorials, docs, framework pages and encyclopedic results."""

    if result.source_type in TRUSTED_STARTUP_SOURCE_TYPES:
        return False
    domain = normalize_domain(result.url)
    title = clean_text(result.title)
    lowered_title = title.lower()
    path = urlparse(result.url).path.lower()
    name = _normalized_name(extract_candidate_name(title, result.url))

    if any(_domain_matches(domain, blocked_domain) for blocked_domain in NON_STARTUP_DOMAINS):
        if not has_startup_context(f"{title} {result.snippet} {combined_text}", result.source_type, result.url):
            return True
        if any(re.search(pattern, lowered_title) for pattern in NON_STARTUP_TITLE_PATTERNS):
            return True
        if name in GENERIC_AI_ENTITY_NAMES:
            return True

    if name in GENERIC_AI_ENTITY_NAMES:
        return True
    if any(re.search(pattern, lowered_title) for pattern in NON_STARTUP_TITLE_PATTERNS):
        return True
    if re.search(r"/(docs?|documentation|tutorial|learn|wiki)(/|$)", path):
        return True
    if "github.com" in domain and re.search(r"/(pytorch|tensorflow|huggingface|openai)(/|$)", path):
        return True
    return False


def find_signals(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    signals: dict[str, list[str]] = {}
    for group, terms in SIGNAL_GROUPS.items():
        matches = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)]
        signals[group] = sorted(matches)
    return signals


def score_discovery_candidate(text: str, url: str) -> float:
    signals = find_signals(text)
    score = 0.0
    for group, terms in signals.items():
        for term in terms:
            score += SIGNAL_GROUPS[group][term]
    domain = normalize_domain(url)
    for credible_domain, boost in SOURCE_CREDIBILITY_BOOSTS.items():
        if domain == credible_domain or domain.endswith(f".{credible_domain}"):
            score += boost
            break
    if signals["competitor_stack"] and (signals["ai_framework"] or signals["maturity"]):
        score += 8
    if signals["nvidia_fit"] and (signals["ai_framework"] or signals["maturity"]):
        score += 10
    return round(max(score, 0.0), 2)


def classify_quality_tier(
    score: float,
    signals: dict[str, list[str]],
    source_type: str = "web_search",
) -> str:
    if signals["nvidia_fit"] and (signals["ai_framework"] or signals["maturity"]):
        return "alta"
    if signals["competitor_stack"] and (signals["ai_framework"] or signals["maturity"]):
        return "alta"
    if source_type == "nvidia_inception_showcase":
        return "alta"
    if score >= 35:
        return "media"
    if score >= 18:
        return "baixa"
    return "triagem"


def recommended_action_for_candidate(signals: dict[str, list[str]], quality_tier: str) -> str:
    if signals["nvidia_fit"]:
        return "Priorizar contato: ja tem sinal tecnico aderente ao ecossistema NVIDIA."
    if signals["competitor_stack"]:
        return "Investigar migracao/otimizacao: ha stack concorrente detectada."
    if signals["maturity"]:
        return "Enriquecer com careers/site tecnico e rodar analise profunda."
    if quality_tier == "alta":
        return "Rodar analise profunda e mapear recomendacao NVIDIA."
    return "Manter em triagem ate obter mais evidencia tecnica."


def build_analysis_query(
    name: str,
    url: str,
    evidence: str,
    signals: dict[str, list[str]],
    source_type: str,
) -> str:
    parts = [
        f"Startup candidata: {name}.",
        f"Fonte principal: {url}.",
        f"Tipo de fonte: {source_type}.",
        f"Evidencia coletada: {evidence}",
    ]
    if signals["nvidia_fit"]:
        parts.append(f"Sinais NVIDIA: {', '.join(signals['nvidia_fit'])}.")
    if signals["ai_framework"]:
        parts.append(f"Sinais de IA/framework: {', '.join(signals['ai_framework'])}.")
    if signals["competitor_stack"]:
        parts.append(f"Stack concorrente detectada: {', '.join(signals['competitor_stack'])}.")
    if signals["maturity"]:
        parts.append(f"Sinais de maturidade tecnica: {', '.join(signals['maturity'])}.")
    return " ".join(parts)


def evidence_excerpt(text: str, signals: dict[str, list[str]], fallback: str, max_len: int = 520) -> str:
    terms = [term for terms in signals.values() for term in terms]
    if not text:
        return fallback[:max_len]
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    start = max(min(positions) - 160, 0) if positions else 0
    excerpt = text[start : start + max_len]
    return excerpt.strip() or fallback[:max_len]


def _query_signal_context(query: str) -> str:
    lowered = query.lower()
    terms: list[str] = []
    for term in ["ia", "inteligencia artificial", "inteligência artificial", "machine learning"]:
        if term in lowered:
            terms.append(term)
    return " ".join(dict.fromkeys(terms))


def build_candidate(result: SearchResult, fetch_pages: bool = True) -> DiscoveryCandidate:
    page_text = ""
    if fetch_pages:
        try:
            page_text = fetch_page_text(result.url)
        except Exception as exc:
            logger.warning("Discovery page fetch wrapper failed for %s: %s", result.url, exc)
            page_text = ""
    combined = " ".join(
        part for part in [_query_signal_context(result.query), result.title, result.snippet, page_text[:8000]] if part
    )
    signals = find_signals(combined)
    score = score_discovery_candidate(combined, result.url)
    excerpt = evidence_excerpt(combined, signals, result.snippet or result.title)
    quality_tier = classify_quality_tier(score, signals, result.source_type)
    name = extract_candidate_name(result.title, result.url)
    return DiscoveryCandidate(
        name=name,
        url=result.url,
        source_domain=normalize_domain(result.url),
        source_query=result.query,
        title=result.title,
        snippet=result.snippet,
        evidence_excerpt=excerpt,
        score=score,
        nvidia_signals=signals["nvidia_fit"],
        ai_framework_signals=signals["ai_framework"],
        competitor_stack_signals=signals["competitor_stack"],
        maturity_signals=signals["maturity"],
        wrapper_risk_signals=signals["wrapper_risk"],
        collected_at=utc_now_iso(),
        source_type=result.source_type,
        company_website=result.company_website,
        location=result.location,
        team_size=result.team_size,
        quality_tier=quality_tier,
        recommended_action=recommended_action_for_candidate(signals, quality_tier),
        analysis_query=build_analysis_query(name, result.url, excerpt, signals, result.source_type),
    )


def has_meaningful_signals(candidate: DiscoveryCandidate) -> bool:
    return any(
        [
            candidate.nvidia_signals,
            candidate.ai_framework_signals,
            candidate.competitor_stack_signals,
            candidate.maturity_signals,
            candidate.wrapper_risk_signals,
        ]
    )


def is_accepted_startup_candidate(candidate: DiscoveryCandidate) -> bool:
    """Keep only plausible company leads for the outbound product surface."""

    context = " ".join(
        [
            candidate.name,
            candidate.title,
            candidate.snippet,
            candidate.evidence_excerpt,
            candidate.company_website,
        ]
    )
    synthetic_result = SearchResult(
        query=candidate.source_query,
        title=candidate.title or candidate.name,
        url=candidate.url,
        snippet=candidate.snippet,
        source_type=candidate.source_type,
        company_website=candidate.company_website,
        location=candidate.location,
        team_size=candidate.team_size,
    )
    if is_non_startup_search_result(synthetic_result, context):
        return False
    if not has_meaningful_signals(candidate):
        return False
    if not has_startup_context(context, candidate.source_type, candidate.url) and not has_company_identity(
        candidate.name,
        candidate.url,
        candidate.source_type,
    ):
        return False
    if candidate.quality_tier == "triagem" and not (
        candidate.competitor_stack_signals or candidate.nvidia_signals or candidate.source_type in TRUSTED_STARTUP_SOURCE_TYPES
    ):
        return False
    return True


def fetch_yc_industry_results(
    industry_url: str = YC_AI_INDUSTRY_URL,
    max_results: int = 25,
    timeout: int = 20,
) -> list[SearchResult]:
    """Fetch YC's public industry page and parse embedded company data."""

    try:
        response = requests.get(industry_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except Exception as exc:
        logger.warning("YC industry fetch failed for %s: %s", industry_url, exc)
        return []
    page_match = re.search(r'data-page="([^"]+)"', response.text)
    if not page_match:
        return []
    payload = json.loads(html.unescape(page_match.group(1)))
    companies = payload.get("props", {}).get("companies", [])
    results: list[SearchResult] = []
    for company in companies[:max_results]:
        name = company.get("name") or company.get("slug") or "YC company"
        path = company.get("ycdc_company_url") or f"/companies/{company.get('slug', '')}"
        url = f"https://www.ycombinator.com{path}"
        parts = [
            company.get("one_liner", ""),
            company.get("long_description", ""),
            " ".join(company.get("industries", [])),
            company.get("location", ""),
            f"team_size={company.get('team_size')}" if company.get("team_size") else "",
        ]
        results.append(
            SearchResult(
                query=f"yc_industry:{industry_url}",
                title=str(name),
                url=url,
                snippet=clean_text(" ".join(str(part) for part in parts if part)),
                source_type="yc_directory",
                company_website=str(company.get("website") or ""),
                location=str(company.get("location") or ""),
                team_size=int(company["team_size"]) if company.get("team_size") else None,
            )
        )
    return results


def fetch_aws_startups_blog_results(
    url: str = AWS_STARTUPS_BLOG_URL,
    max_results: int = 18,
    timeout: int = 20,
) -> list[SearchResult]:
    """Parse AWS Startups blog cards from the public landing page."""

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except Exception as exc:
        logger.warning("AWS Startups blog fetch failed for %s: %s", url, exc)
        return []
    anchors: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.S | re.I):
        href = html.unescape(match.group(1))
        title = clean_text(match.group(2))
        if not title or "/blogs/startups/" not in href:
            continue
        if any(skip in href for skip in ["/category/", "/author/", "#"]):
            continue
        if title.lower() in {"permalink", "startup", "featured", "customer solutions"}:
            continue
        if href in seen:
            continue
        seen.add(href)
        anchors.append((title, href))
        if len(anchors) >= max_results:
            break

    results: list[SearchResult] = []
    for title, href in anchors:
        lowered = title.lower()
        if not any(term in lowered for term in ["ai", "bedrock", "startup", "generative", "llm", "aws"]):
            continue
        startup_name = extract_aws_startup_name(title)
        if not startup_name:
            continue
        results.append(
            SearchResult(
                query=f"aws_startups_blog:{url}",
                title=startup_name,
                url=href,
                snippet=f"AWS Startups public blog article: {title}",
                source_type="aws_startups_blog",
            )
        )
    return results


def fetch_nvidia_inception_showcase_results(
    url: str = NVIDIA_INCEPTION_SHOWCASE_URL,
    max_results: int = 12,
    timeout: int = 20,
) -> list[SearchResult]:
    """Parse public NVIDIA Inception showcase startup links."""

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except Exception as exc:
        logger.warning("NVIDIA Inception showcase fetch failed for %s: %s", url, exc)
        return []
    results: list[SearchResult] = []
    seen: set[str] = set()
    pattern = r'<a[^>]+href="([^"]+)"[^>]*>\s*Visite o Site da\s+([^<\n]+)'
    for match in re.finditer(pattern, response.text, re.S | re.I):
        company_url = html.unescape(match.group(1))
        name = clean_text(match.group(2))
        if not name or company_url in seen:
            continue
        seen.add(company_url)
        snippet = (
            f"{name} aparece no NVIDIA Inception Showcase, uma fonte oficial de startups "
            "com aplicacoes aceleradas pela NVIDIA e casos reais de IA."
        )
        results.append(
            SearchResult(
                query=f"nvidia_inception_showcase:{url}",
                title=name,
                url=company_url,
                snippet=snippet,
                source_type="nvidia_inception_showcase",
                company_website=company_url,
            )
        )
        if len(results) >= max_results:
            break
    return results


def dedupe_candidates(candidates: list[DiscoveryCandidate]) -> list[DiscoveryCandidate]:
    best_by_identity: dict[str, DiscoveryCandidate] = {}
    identity_by_alias: dict[str, str] = {}
    for candidate in candidates:
        url_key = candidate.url.lower().rstrip("/")
        name_key = re.sub(r"\W+", "", candidate.name.lower())
        aliases = [alias for alias in [url_key, name_key] if alias]
        identity = next((identity_by_alias[alias] for alias in aliases if alias in identity_by_alias), aliases[0])
        existing = best_by_identity.get(identity)
        if existing is None or candidate.score > existing.score:
            best_by_identity[identity] = candidate
        for alias in aliases:
            identity_by_alias[alias] = identity
    return sorted(best_by_identity.values(), key=lambda item: item.score, reverse=True)


def _safe_search_web(query: str, results_per_query: int) -> list[SearchResult]:
    try:
        return search_web(query, max_results=results_per_query)
    except Exception as exc:
        logger.warning("Search worker failed for query=%r: %s", query, exc)
        return []


def discover_startups(
    queries: list[str] | None = None,
    campaign: str = "full",
    limit: int = 20,
    results_per_query: int = 6,
    fetch_pages: bool = True,
    delay_seconds: float = 0.0,
    include_direct_sources: bool = True,
    search_workers: int = 12,
) -> list[DiscoveryCandidate]:
    """Run public outbound discovery and return ranked startup candidates."""

    base_queries = queries if queries is not None else campaign_queries(campaign)
    selected_queries = [query.strip() for query in base_queries if query.strip()]
    all_candidates: list[DiscoveryCandidate] = []
    if include_direct_sources:
        yc_budget = min(max(limit, results_per_query), 30)
        for industry_url in YC_DIRECT_INDUSTRY_URLS:
            try:
                yc_results = fetch_yc_industry_results(industry_url, max_results=yc_budget)
            except Exception as exc:
                logger.warning("YC direct source failed for %s: %s", industry_url, exc)
                yc_results = []
            for result in yc_results:
                candidate = build_candidate(result, fetch_pages=False)
                if is_accepted_startup_candidate(candidate):
                    all_candidates.append(candidate)
        for fetcher in [fetch_nvidia_inception_showcase_results, fetch_aws_startups_blog_results]:
            try:
                direct_results = fetcher(max_results=max(results_per_query * 3, 10))
            except Exception as exc:
                logger.warning("Direct discovery source %s failed: %s", fetcher.__name__, exc)
                direct_results = []
            for result in direct_results:
                candidate = build_candidate(result, fetch_pages=fetch_pages)
                if is_accepted_startup_candidate(candidate):
                    all_candidates.append(candidate)

    search_results: list[SearchResult] = []
    if selected_queries:
        worker_count = max(1, min(search_workers, len(selected_queries)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_safe_search_web, query, results_per_query): query
                for query in selected_queries
            }
            for future in as_completed(futures):
                try:
                    search_results.extend(future.result())
                except Exception as exc:
                    logger.warning("Search future failed for query=%r: %s", futures[future], exc)

    search_result_budget = max(limit * 3, 30)
    for result in search_results[:search_result_budget]:
        candidate = build_candidate(result, fetch_pages=fetch_pages)
        if candidate.score > 0 and is_accepted_startup_candidate(candidate):
            all_candidates.append(candidate)
        if delay_seconds:
            time.sleep(delay_seconds)
    return dedupe_candidates(all_candidates)[:limit]


def discover_startups_for_theme(
    theme: str,
    *,
    limit: int = 20,
    results_per_query: int = 6,
    fetch_pages: bool = True,
    delay_seconds: float = 0.0,
    include_direct_sources: bool = False,
    search_workers: int = 8,
) -> list[DiscoveryCandidate]:
    """Run outbound discovery for a theme using the case source list."""

    return discover_startups(
        queries=build_theme_discovery_queries(theme),
        campaign="full",
        limit=limit,
        results_per_query=results_per_query,
        fetch_pages=fetch_pages,
        delay_seconds=delay_seconds,
        include_direct_sources=include_direct_sources,
        search_workers=search_workers,
    )


def save_candidates(candidates: list[DiscoveryCandidate], output_path: str | Path = DEFAULT_DISCOVERY_OUTPUT) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def initialize_discovery_store(db_path: str | Path = DEFAULT_DISCOVERY_DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS discovery_candidates (
                candidate_key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                company_website TEXT,
                source_domain TEXT,
                source_type TEXT,
                source_query TEXT,
                title TEXT,
                snippet TEXT,
                evidence_excerpt TEXT,
                score REAL NOT NULL,
                quality_tier TEXT,
                recommended_action TEXT,
                analysis_query TEXT,
                nvidia_signals_json TEXT NOT NULL,
                ai_framework_signals_json TEXT NOT NULL,
                competitor_stack_signals_json TEXT NOT NULL,
                maturity_signals_json TEXT NOT NULL,
                wrapper_risk_signals_json TEXT NOT NULL,
                location TEXT,
                team_size INTEGER,
                collected_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_discovery_candidates_rank
            ON discovery_candidates (quality_tier, score DESC, collected_at DESC)
            """
        )


def candidate_key(candidate: DiscoveryCandidate) -> str:
    url_key = candidate.url.lower().rstrip("/")
    if url_key:
        return url_key
    return re.sub(r"\W+", "", candidate.name.lower())


def save_candidates_sqlite(
    candidates: list[DiscoveryCandidate],
    db_path: str | Path = DEFAULT_DISCOVERY_DB_PATH,
    replace: bool = False,
) -> int:
    initialize_discovery_store(db_path)
    now = utc_now_iso()
    with sqlite3.connect(db_path) as connection:
        if replace:
            connection.execute("DELETE FROM discovery_candidates")
        for candidate in candidates:
            connection.execute(
                """
                INSERT INTO discovery_candidates (
                    candidate_key, name, url, company_website, source_domain,
                    source_type, source_query, title, snippet, evidence_excerpt,
                    score, quality_tier, recommended_action, analysis_query,
                    nvidia_signals_json, ai_framework_signals_json,
                    competitor_stack_signals_json, maturity_signals_json,
                    wrapper_risk_signals_json, location, team_size, collected_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_key) DO UPDATE SET
                    name=excluded.name,
                    company_website=excluded.company_website,
                    source_domain=excluded.source_domain,
                    source_type=excluded.source_type,
                    source_query=excluded.source_query,
                    title=excluded.title,
                    snippet=excluded.snippet,
                    evidence_excerpt=excluded.evidence_excerpt,
                    score=excluded.score,
                    quality_tier=excluded.quality_tier,
                    recommended_action=excluded.recommended_action,
                    analysis_query=excluded.analysis_query,
                    nvidia_signals_json=excluded.nvidia_signals_json,
                    ai_framework_signals_json=excluded.ai_framework_signals_json,
                    competitor_stack_signals_json=excluded.competitor_stack_signals_json,
                    maturity_signals_json=excluded.maturity_signals_json,
                    wrapper_risk_signals_json=excluded.wrapper_risk_signals_json,
                    location=excluded.location,
                    team_size=excluded.team_size,
                    updated_at=excluded.updated_at
                """,
                (
                    candidate_key(candidate),
                    candidate.name,
                    candidate.url,
                    candidate.company_website,
                    candidate.source_domain,
                    candidate.source_type,
                    candidate.source_query,
                    candidate.title,
                    candidate.snippet,
                    candidate.evidence_excerpt,
                    candidate.score,
                    candidate.quality_tier,
                    candidate.recommended_action,
                    candidate.analysis_query,
                    json.dumps(candidate.nvidia_signals, ensure_ascii=False),
                    json.dumps(candidate.ai_framework_signals, ensure_ascii=False),
                    json.dumps(candidate.competitor_stack_signals, ensure_ascii=False),
                    json.dumps(candidate.maturity_signals, ensure_ascii=False),
                    json.dumps(candidate.wrapper_risk_signals, ensure_ascii=False),
                    candidate.location,
                    candidate.team_size,
                    candidate.collected_at,
                    now,
                ),
            )
    return len(candidates)


def list_discovery_candidates(
    db_path: str | Path = DEFAULT_DISCOVERY_DB_PATH,
    limit: int = 100,
    quality_tier: str | None = None,
) -> list[dict[str, object]]:
    initialize_discovery_store(db_path)
    query = "SELECT * FROM discovery_candidates"
    params: list[object] = []
    if quality_tier:
        query += " WHERE quality_tier = ?"
        params.append(quality_tier)
    query += " ORDER BY score DESC, updated_at DESC LIMIT ?"
    params.append(limit)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()
    decoded: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        for key in [
            "nvidia_signals_json",
            "ai_framework_signals_json",
            "competitor_stack_signals_json",
            "maturity_signals_json",
            "wrapper_risk_signals_json",
        ]:
            item[key.removesuffix("_json")] = json.loads(str(item.pop(key) or "[]"))
        result = SearchResult(
            query=str(item.get("source_query") or ""),
            title=str(item.get("title") or item.get("name") or ""),
            url=str(item.get("url") or ""),
            snippet=str(item.get("snippet") or ""),
            source_type=str(item.get("source_type") or "web_search"),
            company_website=str(item.get("company_website") or ""),
            location=str(item.get("location") or ""),
            team_size=int(item["team_size"]) if item.get("team_size") else None,
        )
        context = " ".join(
            str(item.get(key) or "")
            for key in ["name", "title", "snippet", "evidence_excerpt", "company_website"]
        )
        if is_non_startup_search_result(result, context):
            continue
        if not has_startup_context(context, result.source_type, result.url) and not has_company_identity(
            str(item.get("name") or item.get("title") or ""),
            result.url,
            result.source_type,
        ):
            continue
        decoded.append(item)
    return decoded


def candidates_as_dicts(candidates: list[DiscoveryCandidate]) -> list[dict[str, object]]:
    return [asdict(candidate) for candidate in candidates]
