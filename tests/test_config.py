"""Tests for forge_bot.config."""

import pytest
from pydantic import ValidationError

from forge_bot.config import Settings


def _required_env() -> dict[str, str]:
    """Minimal required env vars for a valid Settings."""
    return {
        "FORGE_INSTANCE_URL": "https://gitea.example.com",
        "FORGE_API_TOKEN": "test-token",
        "FORGE_WEBHOOK_SECRET": "test-secret",
        "LLM_API_KEY": "test-llm-key",
    }


def test_settings_with_required_only(monkeypatch: pytest.MonkeyPatch):
    for k, v in _required_env().items():
        monkeypatch.setenv(k, v)

    s = Settings()
    assert s.forge_instance_url == "https://gitea.example.com"
    assert s.forge_api_token == "test-token"
    assert s.forge_webhook_secret == "test-secret"
    assert s.llm_api_key == "test-llm-key"


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch):
    for k, v in _required_env().items():
        monkeypatch.setenv(k, v)
    # Clear optional env vars that may leak from the host environment.
    for var in (
        "LLM_BASE_URL", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
        "LLM_TIMEOUT", "LLM_MAX_CONCURRENT", "LLM_CONTEXT_WINDOW",
        "SANDBOX_ENABLED", "SANDBOX_TIMEOUT", "RAG_ENABLED",
        "LOG_LEVEL", "BOT_COMMAND_PREFIX",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings()
    assert s.llm_base_url == "https://api.openai.com/v1"
    assert s.llm_model == "gpt-4o"
    assert s.llm_temperature == 0.2
    assert s.llm_max_tokens == 4096
    assert s.llm_timeout == 120
    assert s.llm_max_concurrent == 3
    assert s.sandbox_enabled is True
    assert s.sandbox_timeout == 60
    assert s.rag_enabled is False
    assert s.log_level == "INFO"
    assert s.bot_command_prefix == "/"


def test_settings_override_optionals(monkeypatch: pytest.MonkeyPatch):
    env = _required_env()
    env["LLM_MODEL"] = "llama3.2"
    env["LLM_TEMPERATURE"] = "0.7"
    env["SANDBOX_ENABLED"] = "false"
    env["RAG_ENABLED"] = "true"
    env["LOG_LEVEL"] = "DEBUG"
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    s = Settings()
    assert s.llm_model == "llama3.2"
    assert s.llm_temperature == 0.7
    assert s.sandbox_enabled is False
    assert s.rag_enabled is True
    assert s.log_level == "DEBUG"


def test_settings_missing_required_raises(monkeypatch: pytest.MonkeyPatch):
    # Only set some required vars, omit FORGE_INSTANCE_URL
    monkeypatch.setenv("FORGE_API_TOKEN", "token")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "secret")
    monkeypatch.setenv("LLM_API_KEY", "key")
    # Clear any .env file influence
    monkeypatch.delenv("FORGE_INSTANCE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_tool_mode_default(monkeypatch: pytest.MonkeyPatch):
    for k, v in _required_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("LLM_TOOL_MODE", raising=False)

    s = Settings()
    assert s.llm_tool_mode == "auto"


def test_settings_tool_mode_override(monkeypatch: pytest.MonkeyPatch):
    env = _required_env()
    env["LLM_TOOL_MODE"] = "prompt"
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    s = Settings()
    assert s.llm_tool_mode == "prompt"
