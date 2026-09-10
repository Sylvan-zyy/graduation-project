from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from openai import APITimeoutError, OpenAIError

from src.llm_client import (
    create_deepseek_client,
    load_deepseek_config,
    parse_correction_result,
    request_correction,
    request_json_completion,
)

def test_load_deepseek_config(monkeypatch):
    """测试 load_deepseek_config 能正确读取环境变量"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    config = load_deepseek_config()

    assert config["api_key"] == "test-api-key"
    assert config["base_url"] == "https://example.com"
    assert config["model"] == "test-model"


def test_missing_api_key_raises_error(monkeypatch):
    """测试 load_deepseek_config 在缺少 API Key 时抛出 ValueError"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    with pytest.raises(ValueError, match="未找到 DEEPSEEK_API_KEY"):
            load_deepseek_config()


def test_create_deepseek_client(monkeypatch):
    """测试 create_deepseek_client 能正确创建 DeepSeek 客户端"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")

    with patch("src.llm_client.OpenAI") as mock_openai:
        client = create_deepseek_client()

    mock_openai.assert_called_once_with(
        api_key="test-api-key",
        base_url="https://example.com",
        timeout=30.0,
    )
    assert client is mock_openai.return_value


def test_request_json_completion(monkeypatch):
    """测试 request_json_completion 能正确调用 DeepSeek API"""
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
    """测试 parse_correction_result 能正确解析 JSON 字符串"""
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
    """测试 request_correction 能正确记录用量和处理时间"""
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


def test_parse_correction_result_rejects_empty_content():
    """测试 parse_correction_result 遇到空内容时抛出 ValueError"""
    empty_values = (None, "", "   \n")

    for content in empty_values:
        with pytest.raises(ValueError, match="模型返回内容为空"):
            parse_correction_result(content)


def test_parse_correction_result_rejects_invalid_json():
    """测试 parse_correction_result遇到无效JSON时抛出 ValueError"""
    with pytest.raises(
        ValueError,
        match="模型返回的内容不是有效 JSON",
    ):
        parse_correction_result("这不是 JSON")


def test_request_json_completion_handles_timeout(monkeypatch):
    """测试 request_json_completion 在请求超时时抛出 RuntimeError"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    timeout_error = APITimeoutError(
    request=Mock()
    )

    with patch(
        "src.llm_client.create_deepseek_client"
    ) as mock_create_client:
        mock_client = mock_create_client.return_value
        mock_client.chat.completions.create.side_effect = timeout_error

        with pytest.raises(
            RuntimeError,
            match="DeepSeek API 请求超时",
        ):
            request_json_completion(
                system_prompt="请返回 JSON",
                user_prompt="测试字幕",
            )


def test_request_json_completion_handles_api_error(monkeypatch):
    """测试 request_json_completion 在 API 错误时抛出 RuntimeError"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    api_error = OpenAIError("模拟 API 错误")

    with patch(
        "src.llm_client.create_deepseek_client"
    ) as mock_create_client:
        mock_client = mock_create_client.return_value
        mock_client.chat.completions.create.side_effect = api_error

        with pytest.raises(
            RuntimeError,
            match="DeepSeek API 请求失败",
        ):
            request_json_completion(
                system_prompt="请返回 JSON",
                user_prompt="测试字幕",
            )            


def test_request_correction_uses_cached_result(monkeypatch):
    """测试request_correction在缓存命中时，直接返回缓存结果"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    cached_result = {
        "corrected_text": "缓存中的字幕",
        "changed": True,
        "reason": "缓存结果",
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
        "processing_time_seconds": 0.4,
        "from_cache": False,
    }

    with (
        patch(
            "src.llm_client.build_cache_key",
            # 当这个函数被调用时，返回"test-cache-key"这个字符串
            return_value="test-cache-key",
        ),
        patch(
            "src.llm_client.load_cache",
            # 当这个函数被调用时，永远返回这个字典
            return_value={"test-cache-key": cached_result},
        ),
        patch(
            "src.llm_client.request_json_completion"
        ) as mock_request,
    ):
        result = request_correction(
            system_prompt="系统提示",
            user_prompt="用户提示",
            cache_path="cache/test.json",
        )

    mock_request.assert_not_called()
    assert result["corrected_text"] == "缓存中的字幕"
    assert result["from_cache"] is True          


def test_request_correction_saves_new_result(monkeypatch):
    """测试request_correction在缓存未命中时，调用API并保存新结果"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    # SimpleNamespace创建一个简单的对象，让关键字参数变成属性
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"corrected_text": "新校正字幕", '
                        '"changed": true, '
                        '"reason": "修正错别字"}'
                    )
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=90,
            completion_tokens=20,
            total_tokens=110,
        ),
    )

    with (
        patch(
            "src.llm_client.build_cache_key",
            return_value="test-cache-key",
        ),
        patch(
            "src.llm_client.load_cache",
            return_value={},
        ),
        patch(
            "src.llm_client.save_cache"
        ) as mock_save,
        patch(
            "src.llm_client.request_json_completion",
            return_value=mock_response,
        ) as mock_request,
        patch(
            "src.llm_client.time.perf_counter",
            side_effect=[10.0, 10.5],
        ),
    ):
        result = request_correction(
            system_prompt="系统提示",
            user_prompt="用户提示",
            cache_path="cache/test.json",
        )

    mock_request.assert_called_once()
    mock_save.assert_called_once_with(
        cache={"test-cache-key": result},
        file_path="cache/test.json",
    )
    assert result["from_cache"] is False
    assert result["corrected_text"] == "新校正字幕"
    assert result["total_tokens"] == 110
    assert result["processing_time_seconds"] == 0.5  