import pytest

from src.llm_client import load_deepseek_config


def test_load_deepseek_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    config = load_deepseek_config()

    assert config["api_key"] == "test-api-key"
    assert config["base_url"] == "https://example.com"
    assert config["model"] == "test-model"


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    with pytest.raises(ValueError, match="未找到 DEEPSEEK_API_KEY"):
            load_deepseek_config()