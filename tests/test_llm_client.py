from unittest.mock import patch

import pytest

from src.llm_client import (
    create_deepseek_client,
    load_deepseek_config,
    parse_correction_result,
    request_json_completion,
)

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


def test_create_deepseek_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")

    with patch("src.llm_client.OpenAI") as mock_openai:
        client = create_deepseek_client()

    mock_openai.assert_called_once_with(
        api_key="test-api-key",
        base_url="https://example.com",
    )
    assert client is mock_openai.return_value


def test_request_json_completion(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    with patch("src.llm_client.create_deepseek_client") as mock_create_client:
        mock_client = mock_create_client.return_value
        mock_response = object()
        mock_client.chat.completions.create.return_value = mock_response

        result = request_json_completion(
            system_prompt="请返回 JSON",
            user_prompt="测试字幕",
        )

    mock_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "请返回 JSON"},
            {"role": "user", "content": "测试字幕"},
        ],
        response_format={"type": "json_object"},
        max_tokens=512,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    assert result is mock_response


def test_parse_correction_result():
    content = (
        '{"corrected_text": "今天天气很好", '
        '"changed": true, '
        '"reason": "修正错别字"}'
    )

    result = parse_correction_result(content)

    assert result["corrected_text"] == "今天天气很好"
    assert result["changed"] is True
    assert result["reason"] == "修正错别字"