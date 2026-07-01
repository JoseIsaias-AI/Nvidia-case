import re
from pathlib import Path

import pytest

from nvidia_startup_ai_radar.config import Settings, get_settings
from nvidia_startup_ai_radar.llm import build_chat_model
from nvidia_startup_ai_radar.storage import SQLiteProfileRepository, get_profile_repository


def _state(name: str) -> dict:
    return {
        "output_language": "pt",
        "human_review_required": False,
        "errors": [],
        "briefing_pt": f"# Briefing NVIDIA Startup AI Radar: {name}",
        "profile": {
            "id": name.lower(),
            "nome": name,
            "setor": "Fintech",
            "origem": "outbound",
            "classificacao": "AI-enabled",
            "score_maturidade_ia": 50,
            "score_wrapper_risco": 0,
            "evidencias": [{"fonte_url": "local://test", "trecho_resumido": name}],
        },
    }


def test_profile_repository_contract_uses_sqlite_implementation(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_REPOSITORY_BACKEND", "sqlite")
    repository = get_profile_repository(tmp_path / "profiles.sqlite")

    assert isinstance(repository, SQLiteProfileRepository)

    run_id = repository.save(_state("RepoReady"))
    assert repository.get(run_id)["nome"] == "RepoReady"
    assert repository.list(search="repo")[0]["run_id"] == run_id
    assert repository.filter(classificacao="AI-enabled")[0]["nome"] == "RepoReady"
    assert "Fintech" in repository.list_distinct_values("setor")


def test_profile_repository_rejects_unknown_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_REPOSITORY_BACKEND", "postgres")

    with pytest.raises(ValueError, match="ProfileRepository"):
        get_profile_repository(tmp_path / "profiles.sqlite")


def test_env_example_documents_all_runtime_environment_variables():
    project_root = Path(__file__).resolve().parents[1]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (project_root / "src").rglob("*.py"))
    env_names = set(re.findall(r"os\.getenv\([\"']([^\"']+)[\"']", source_text))

    env_example = project_root / ".env.example"
    documented = {
        line.split("=", 1)[0].strip()
        for line in env_example.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }
    expected_runtime_vars = {
        "LLM_PROVIDER",
        "RADAR_OUTPUT_LANGUAGE",
        "RADAR_ENABLE_WEB_FETCH",
        "RADAR_WEB_HOST",
        "RADAR_WEB_PORT",
        "PROFILE_REPOSITORY_BACKEND",
        "NVIDIA_API_KEY",
        "NVIDIA_MODEL",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "GROQ_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "SCRAPER_MIN_TEXT_CHARS",
        "SCRAPER_CACHE_TTL_HOURS",
        "SCRAPER_CACHE_DIR",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "RAG_EMBEDDING_BACKEND",
        "RAG_EMBEDDING_MODEL",
        "RAG_EMBEDDING_CACHE_DB",
        "RAG_RERANKER",
        "RAG_RERANKER_MODEL",
        "COHERE_API_KEY",
        "RAG_COHERE_API_KEY",
        "RAG_COHERE_MODEL",
    }

    assert env_names - documented == set()
    assert expected_runtime_vars - documented == set()
    for secret_name in [
        "NVIDIA_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "FIRECRAWL_API_KEY",
        "COHERE_API_KEY",
    ]:
        assert re.search(rf"^{secret_name}=$", env_example.read_text(encoding="utf-8"), flags=re.MULTILINE)


def test_settings_support_groq_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    settings = get_settings()

    assert settings.llm_provider == "groq"
    assert settings.groq_api_key == "fake-groq-key"
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.groq_base_url == "https://api.groq.com/openai/v1"


def test_build_chat_model_supports_groq_openai_compatible_client():
    settings = Settings(llm_provider="groq", groq_api_key="fake-groq-key")

    llm = build_chat_model(settings)

    assert llm is not None
