"""Lightweight public web fetching helpers."""

from __future__ import annotations

from urllib.parse import urlparse

from nvidia_startup_ai_radar.schemas import RawPage


def is_probably_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def fetch_public_page(url: str, timeout: int = 20) -> RawPage:
    """Fetch and extract a public page.

    This is a thin MVP layer. For production, add robots.txt checks, per-domain
    rate limits, Playwright fallback and Firecrawl fallback as described in docs.
    """

    import requests
    import trafilatura

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "NVIDIA-Startup-AI-Radar/0.1 (+research prototype)"},
    )
    response.raise_for_status()
    extracted = trafilatura.extract(response.text, include_links=False, include_tables=False)
    text = extracted or response.text
    return RawPage(url=url, title=urlparse(url).netloc, text=text[:15000])
