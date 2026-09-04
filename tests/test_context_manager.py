import pytest

from src.context_manager import get_context
from src.subtitle_io import load_srt


def test_middle_subtitle_with_window_one() -> None:
    """测试中间字幕、窗口大小为1。"""
    subtitles = load_srt("data/input/sample.srt")

    context = get_context(
        subtitles=subtitles,
        current_index=2,
        window_size=1,
    )

    assert context["previous"] == ["我们准备去公园散步。"]
    assert context["current"] == "回来以后继续学习派森编程。"
    assert context["next"] == ["这个课程主要介绍Python的基础知识。"]


def test_first_subtitle() -> None:
    """测试第一条字幕没有前文。"""
    subtitles = load_srt("data/input/sample.srt")

    context = get_context(
        subtitles=subtitles,
        current_index=0,
        window_size=3,
    )

    assert context["previous"] == []
    assert context["current"] == "今天天气很好。"
    assert len(context["next"]) == 3


def test_last_subtitle() -> None:
    """测试最后一条字幕没有后文。"""
    subtitles = load_srt("data/input/sample.srt")

    context = get_context(
        subtitles=subtitles,
        current_index=4,
        window_size=3,
    )

    assert len(context["previous"]) == 3
    assert context["current"] == "学习的时候要结合前后的内容。"
    assert context["next"] == []


def test_zero_window() -> None:
    """测试窗口为0时不提供上下文。"""
    subtitles = load_srt("data/input/sample.srt")

    context = get_context(
        subtitles=subtitles,
        current_index=2,
        window_size=0,
    )

    assert context["previous"] == []
    assert context["current"] == "回来以后继续学习派森编程。"
    assert context["next"] == []


def test_negative_window() -> None:
    """测试负数窗口会引发错误。"""
    subtitles = load_srt("data/input/sample.srt")

    with pytest.raises(ValueError):
        get_context(
            subtitles=subtitles,
            current_index=2,
            window_size=-1,
        )


def test_invalid_index() -> None:
    """测试不存在的字幕位置会引发错误。"""
    subtitles = load_srt("data/input/sample.srt")

    with pytest.raises(IndexError):
        get_context(
            subtitles=subtitles,
            current_index=5,
            window_size=1,
        )