from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.llm_client import (
    create_deepseek_client,
    load_deepseek_config,
    parse_correction_result,
    request_correction,
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


def test_request_correction_records_usage_and_time():
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"corrected_text": "你好", '
                        '"changed": false, '
                        '"reason": "无需修改"}'
                    )
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )

    with (
        patch(
            "src.llm_client.request_json_completion",
            return_value=mock_response,
        ) as mock_request,
        patch(
            "src.llm_client.time.perf_counter",
            side_effect=[10.0, 10.25],
        ),
    ):
        result = request_correction(
            system_prompt="系统提示",
            user_prompt="用户提示",
        )

    mock_request.assert_called_once_with(
        system_prompt="系统提示",
        user_prompt="用户提示",
    )
    assert result["corrected_text"] == "你好"
    assert result["changed"] is False
    assert result["reason"] == "无需修改"
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 20
    assert result["total_tokens"] == 120
    assert result["processing_time_seconds"] == 0.25