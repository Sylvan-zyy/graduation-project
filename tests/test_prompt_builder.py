import pytest

from src.context_manager import get_context
from src.prompt_builder import format_context_items, build_user_prompt
from src.subtitle_io import load_srt


def test_format_context_items() -> None:
    """测试多条字幕的格式化。"""
    items = ["第一条字幕", "第二条字幕"]

    result = format_context_items(items)

    assert result == "- 第一条字幕\n- 第二条字幕"


def test_format_empty_context() -> None:
    """测试没有上下文时的显示内容。"""
    result = format_context_items([])

    assert result == "（无）"


def test_normal_prompt() -> None:
    """测试普通校正提示。"""
    subtitles = load_srt("data/input/sample.srt")
    context = get_context(subtitles, current_index=2, window_size=1)

    prompt = build_user_prompt(context, prompt_type="normal")

    assert "我们准备去公园散步。" in prompt
    assert "回来以后继续学习派森编程。" in prompt
    assert "这个课程主要介绍Python的基础知识。" in prompt
    assert "使表达准确、通顺" in prompt
    assert "corrected_text" in prompt


def test_minimal_prompt() -> None:
    """测试最小修改提示。"""
    subtitles = load_srt("data/input/sample.srt")
    context = get_context(subtitles, current_index=2, window_size=1)

    prompt = build_user_prompt(context, prompt_type="minimal")

    assert "不要进行润色或改写" in prompt
    assert "证据不足" in prompt
    assert "原样保留当前字幕" in prompt


def test_prompt_without_context() -> None:
    """测试窗口为0时的提示词。"""
    subtitles = load_srt("data/input/sample.srt")
    context = get_context(subtitles, current_index=2, window_size=0)

    prompt = build_user_prompt(context, prompt_type="normal")

    assert prompt.count("（无）") == 2
    assert "回来以后继续学习派森编程。" in prompt


def test_invalid_prompt_type() -> None:
    """测试不支持的提示类型。"""
    subtitles = load_srt("data/input/sample.srt")
    context = get_context(subtitles, current_index=2, window_size=1)

    with pytest.raises(ValueError):
        build_user_prompt(context, prompt_type="unknown")