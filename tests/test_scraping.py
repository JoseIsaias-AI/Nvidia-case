from nvidia_startup_ai_radar.scraping import ScrapeAttempt, fetch_public_page


def test_fetch_public_page_uses_playwright_when_requests_text_is_too_short(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRAPER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SCRAPER_MIN_TEXT_CHARS", "80")
    monkeypatch.setenv("SCRAPER_CACHE_TTL_HOURS", "24")
    monkeypatch.setattr("nvidia_startup_ai_radar.scraping._robots_allowed", lambda url, timeout: (True, None))
    monkeypatch.setattr(
        "nvidia_startup_ai_radar.scraping._fetch_with_requests",
        lambda url, timeout: ScrapeAttempt(method="requests", text="short", title="Short", success=True),
    )
    monkeypatch.setattr(
        "nvidia_startup_ai_radar.scraping._fetch_with_playwright",
        lambda url, timeout: ScrapeAttempt(
            method="playwright",
            text="Rendered React careers page with ML engineer GPU inference roles and useful content." * 2,
            title="Rendered",
            success=True,
        ),
    )

    def fail_firecrawl(url, timeout):
        raise AssertionError("Firecrawl should not run after a useful Playwright result.")

    monkeypatch.setattr("nvidia_startup_ai_radar.scraping._fetch_with_firecrawl", fail_firecrawl)

    page = fetch_public_page("https://example.com/careers")

    assert page.scrape_success is True
    assert page.scrape_method == "playwright"
    assert page.served_from_cache is False
    assert "ML engineer" in page.text


def test_fetch_public_page_serves_recent_cache_without_new_network(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRAPER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SCRAPER_MIN_TEXT_CHARS", "40")
    monkeypatch.setenv("SCRAPER_CACHE_TTL_HOURS", "24")
    calls = {"robots": 0, "requests": 0}

    def fake_robots(url, timeout):
        calls["robots"] += 1
        return True, None

    def fake_requests(url, timeout):
        calls["requests"] += 1
        return ScrapeAttempt(
            method="requests",
            text="Useful static startup page with NVIDIA GPU inference signals.",
            title="Cached",
            success=True,
        )

    monkeypatch.setattr("nvidia_startup_ai_radar.scraping._robots_allowed", fake_robots)
    monkeypatch.setattr("nvidia_startup_ai_radar.scraping._fetch_with_requests", fake_requests)

    first = fetch_public_page("https://example.com/startup")
    second = fetch_public_page("https://example.com/startup")

    assert first.scrape_success is True
    assert second.scrape_success is True
    assert second.served_from_cache is True
    assert second.scrape_method == "requests"
    assert calls == {"robots": 1, "requests": 1}


def test_fetch_public_page_respects_robots_before_scraping(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRAPER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SCRAPER_CACHE_TTL_HOURS", "0")
    monkeypatch.setattr(
        "nvidia_startup_ai_radar.scraping._robots_allowed",
        lambda url, timeout: (False, "Bloqueado por robots.txt"),
    )

    def fail_requests(url, timeout):
        raise AssertionError("No network scraper should run when robots.txt blocks the URL.")

    monkeypatch.setattr("nvidia_startup_ai_radar.scraping._fetch_with_requests", fail_requests)

    page = fetch_public_page("https://example.com/private")

    assert page.scrape_success is False
    assert page.scrape_method == "robots"
    assert page.robots_allowed is False
    assert "robots" in (page.failure_reason or "").lower()


def test_fetch_public_page_records_firecrawl_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRAPER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SCRAPER_MIN_TEXT_CHARS", "100")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr("nvidia_startup_ai_radar.scraping._robots_allowed", lambda url, timeout: (True, None))
    monkeypatch.setattr(
        "nvidia_startup_ai_radar.scraping._fetch_with_requests",
        lambda url, timeout: ScrapeAttempt(method="requests", text="short", success=True),
    )
    monkeypatch.setattr(
        "nvidia_startup_ai_radar.scraping._fetch_with_playwright",
        lambda url, timeout: ScrapeAttempt(method="playwright", text="still short", success=True),
    )

    page = fetch_public_page("https://example.com/js-app")

    assert page.scrape_success is False
    assert page.scrape_method == "firecrawl"
    assert "FIRECRAWL_API_KEY" in (page.failure_reason or "")
