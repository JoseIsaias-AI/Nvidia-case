"""Public web fetching helpers with layered fallbacks and cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from nvidia_startup_ai_radar.schemas import RawPage, utc_now_iso


logger = logging.getLogger(__name__)

USER_AGENT = "NVIDIA-Startup-AI-Radar/0.1 (+research prototype)"
DEFAULT_CACHE_DIR = Path("data") / "scrape_cache"
DEFAULT_MIN_TEXT_CHARS = 500
DEFAULT_CACHE_TTL_HOURS = 24
MAX_TEXT_CHARS = 15000

ScrapeMethod = Literal["requests", "playwright", "firecrawl"]


@dataclass(frozen=True)
class ScrapeAttempt:
    method: ScrapeMethod
    text: str = ""
    title: str | None = None
    success: bool = False
    failure_reason: str | None = None
    robots_allowed: bool | None = True


def is_probably_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _cache_dir() -> Path:
    return Path(os.getenv("SCRAPER_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def _minimum_text_chars() -> int:
    return _int_env("SCRAPER_MIN_TEXT_CHARS", DEFAULT_MIN_TEXT_CHARS)


def _cache_ttl_hours() -> float:
    return _float_env("SCRAPER_CACHE_TTL_HOURS", float(DEFAULT_CACHE_TTL_HOURS))


def _clean_text(value: str | None) -> str:
    text = value or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _record_hash(url: str, collected_at: str) -> str:
    return hashlib.sha256(f"{url}|{collected_at}".encode("utf-8")).hexdigest()


def _cache_path(url: str) -> Path:
    return _cache_dir() / f"{_cache_key(url)}.json"


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _raw_page(
    *,
    url: str,
    title: str | None,
    text: str,
    scrape_method: str,
    scrape_success: bool,
    served_from_cache: bool = False,
    failure_reason: str | None = None,
    robots_allowed: bool | None = None,
    collected_at: str | None = None,
    cache_key: str | None = None,
) -> RawPage:
    text = _clean_text(text)[:MAX_TEXT_CHARS]
    return RawPage(
        url=url,
        title=title or urlparse(url).netloc,
        text=text,
        collected_at=collected_at or utc_now_iso(),
        scrape_method=scrape_method,
        scrape_success=scrape_success,
        served_from_cache=served_from_cache,
        failure_reason=failure_reason,
        robots_allowed=robots_allowed,
        cache_key=cache_key,
    )


def _load_cached_page(url: str) -> RawPage | None:
    ttl_hours = _cache_ttl_hours()
    if ttl_hours <= 0:
        return None
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid scrape cache for %s: %s", url, exc)
        return None

    collected_at = _parse_iso(str(payload.get("collected_at", "")))
    if collected_at is None:
        return None
    age_hours = (datetime.now(timezone.utc) - collected_at.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours > ttl_hours:
        return None

    logger.info("Scrape cache hit for %s (age %.2fh).", url, age_hours)
    return _raw_page(
        url=url,
        title=payload.get("title"),
        text=payload.get("text", ""),
        scrape_method=str(payload.get("scrape_method") or "cache"),
        scrape_success=True,
        served_from_cache=True,
        failure_reason=None,
        robots_allowed=payload.get("robots_allowed"),
        collected_at=payload.get("collected_at"),
        cache_key=payload.get("cache_key") or _cache_key(url),
    )


def _save_cached_page(page: RawPage) -> None:
    if not page.scrape_success or not page.text:
        return
    path = _cache_path(page.url)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "collected_at": page.collected_at,
        "scrape_method": page.scrape_method,
        "scrape_success": page.scrape_success,
        "robots_allowed": page.robots_allowed,
        "cache_key": page.cache_key or _cache_key(page.url),
        "record_hash": _record_hash(page.url, page.collected_at),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _robots_url(url: str) -> str:
    parsed = urlparse(url)
    return urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")


def _robots_allowed(url: str, timeout: int) -> tuple[bool, str | None]:
    """Check robots.txt before scraping. Network errors are logged as inconclusive."""

    import requests

    robots_url = _robots_url(url)
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = requests.get(robots_url, timeout=min(timeout, 8), headers={"User-Agent": USER_AGENT})
    except Exception as exc:
        logger.warning("robots.txt indisponivel para %s; seguindo como inconclusivo: %s", url, exc)
        return True, f"robots.txt indisponivel: {exc}"
    if response.status_code == 404:
        return True, None
    if response.status_code >= 400:
        return True, f"robots.txt retornou HTTP {response.status_code}"
    parser.parse(response.text.splitlines())
    allowed = parser.can_fetch(USER_AGENT, url)
    return allowed, None if allowed else f"Bloqueado por robots.txt: {robots_url}"


def _extract_with_trafilatura(html: str) -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_links=False, include_tables=False)
        return _clean_text(extracted)
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)
        return ""


def _html_to_text(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean_text(text)


def _fetch_with_requests(url: str, timeout: int) -> ScrapeAttempt:
    try:
        import requests

        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        text = _extract_with_trafilatura(response.text) or _html_to_text(response.text)
        return ScrapeAttempt(method="requests", text=text, title=urlparse(url).netloc, success=bool(text))
    except Exception as exc:
        logger.warning("Falha no scraping requests para %s: %s", url, exc)
        return ScrapeAttempt(method="requests", success=False, failure_reason=str(exc))


def _fetch_with_playwright(url: str, timeout: int) -> ScrapeAttempt:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        logger.warning("Playwright indisponivel para %s; pulando fallback JS: %s", url, exc)
        return ScrapeAttempt(
            method="playwright",
            success=False,
            failure_reason=f"Playwright indisponivel: {exc}",
        )

    browser = None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            title = page.title() or urlparse(url).netloc
            try:
                text = page.inner_text("body", timeout=5000)
            except Exception:
                text = ""
            if len(_clean_text(text)) < _minimum_text_chars():
                text = _extract_with_trafilatura(page.content()) or text
            browser.close()
            browser = None
            return ScrapeAttempt(method="playwright", text=_clean_text(text), title=title, success=bool(text))
    except Exception as exc:
        logger.warning("Falha no scraping Playwright para %s: %s", url, exc)
        try:
            if browser is not None:
                browser.close()
        except Exception:
            pass
        return ScrapeAttempt(method="playwright", success=False, failure_reason=str(exc))


def _fetch_with_firecrawl(url: str, timeout: int) -> ScrapeAttempt:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return ScrapeAttempt(
            method="firecrawl",
            success=False,
            failure_reason="FIRECRAWL_API_KEY ausente; fallback Firecrawl pulado.",
        )
    endpoint = os.getenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v2/scrape")
    try:
        import requests

        response = requests.post(
            endpoint,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else {}
        if not isinstance(data, dict):
            data = {}
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        text = data.get("markdown") or data.get("content") or data.get("html") or ""
        return ScrapeAttempt(
            method="firecrawl",
            text=_clean_text(text),
            title=metadata.get("title") or urlparse(url).netloc,
            success=bool(_clean_text(text)),
        )
    except Exception as exc:
        logger.warning("Falha no scraping Firecrawl para %s: %s", url, exc)
        return ScrapeAttempt(method="firecrawl", success=False, failure_reason=str(exc))


def _attempt_to_page(url: str, attempt: ScrapeAttempt) -> RawPage:
    return _raw_page(
        url=url,
        title=attempt.title,
        text=attempt.text,
        scrape_method=attempt.method,
        scrape_success=attempt.success,
        failure_reason=attempt.failure_reason,
        robots_allowed=attempt.robots_allowed,
        cache_key=_cache_key(url),
    )


def fetch_public_page(url: str, timeout: int = 20) -> RawPage:
    """Fetch a public page with cache, robots.txt, Playwright and Firecrawl fallbacks."""

    if not is_probably_url(url):
        return _raw_page(
            url=url,
            title="invalid url",
            text="",
            scrape_method="invalid",
            scrape_success=False,
            failure_reason="URL invalida para scraping.",
            robots_allowed=None,
        )

    cached = _load_cached_page(url)
    if cached is not None:
        return cached

    allowed, robots_reason = _robots_allowed(url, timeout)
    if not allowed:
        logger.info("Skipping %s because robots.txt disallows it.", url)
        return _raw_page(
            url=url,
            title=urlparse(url).netloc,
            text="",
            scrape_method="robots",
            scrape_success=False,
            failure_reason=robots_reason,
            robots_allowed=False,
            cache_key=_cache_key(url),
        )
    if robots_reason:
        logger.info("Proceeding with %s after inconclusive robots.txt check: %s", url, robots_reason)

    min_chars = _minimum_text_chars()
    attempts: list[ScrapeAttempt] = []
    for fetcher in (_fetch_with_requests, _fetch_with_playwright, _fetch_with_firecrawl):
        attempt = fetcher(url, timeout)
        attempts.append(attempt)
        text_len = len(_clean_text(attempt.text))
        logger.info(
            "Scrape attempt url=%s method=%s success=%s chars=%s",
            url,
            attempt.method,
            attempt.success,
            text_len,
        )
        if attempt.success and text_len >= min_chars:
            page = _attempt_to_page(url, ScrapeAttempt(**{**attempt.__dict__, "robots_allowed": True}))
            _save_cached_page(page)
            return page

    best = max(attempts, key=lambda item: len(_clean_text(item.text)), default=None)
    reasons = "; ".join(
        f"{attempt.method}: {attempt.failure_reason or 'conteudo insuficiente'}" for attempt in attempts
    )
    best_text = _clean_text(best.text) if best else ""
    if best_text:
        reasons = (
            f"Fonte nao coletada: melhor conteudo abaixo do minimo configurado "
            f"({len(best_text)}/{min_chars}). {reasons}"
        )
    return _raw_page(
        url=url,
        title=urlparse(url).netloc,
        text=best_text,
        scrape_method=attempts[-1].method if attempts else "requests",
        scrape_success=False,
        failure_reason=reasons or "Fonte nao coletada.",
        robots_allowed=True,
        cache_key=_cache_key(url),
    )
