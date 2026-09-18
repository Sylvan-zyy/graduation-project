from datetime import timedelta
from unittest.mock import patch

from srt import Subtitle

from src.correction_engine import (
    correct_all_subtitles,
    correct_single_subtitle,
    correct_srt_file,
)

from src.prompt_builder import SYSTEM_PROMPT


def test_correct_single_subtitle():
    """测试 correct_single_subtitle 函数的基本功能"""
    subtitles = [
        Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=2),
            content="今天天气不错",
        ),
        Subtitle(
            index=2,
            start=timedelta(seconds=2),
            end=timedelta(seconds=4),
            content="我们出去散不",
        ),
        Subtitle(
            index=3,
            start=timedelta(seconds=4),
            end=timedelta(seconds=6),
            content="记得带上雨伞",
        ),
    ]

    mock_result = {
        "corrected_text": "我们出去散步",
        "changed": True,
        "reason": "修正同音错别字",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "processing_time_seconds": 0.5,
    }

    with patch(
        "src.correction_engine.request_correction",
        return_value=mock_result,
    ) as mock_request:
        result = correct_single_subtitle(
            subtitles=subtitles,
            current_index=1,
            window_size=1,
            prompt_type="minimal",
            cache_path="cache/test.json",
        )

    request_arguments = mock_request.call_args.kwargs

    assert request_arguments["cache_path"] == "cache/test.json"
    assert request_arguments["system_prompt"] == SYSTEM_PROMPT
    assert "今天天气不错" in request_arguments["user_prompt"]
    assert "我们出去散不" in request_arguments["user_prompt"]
    assert "记得带上雨伞" in request_arguments["user_prompt"]

    assert result["subtitle_index"] == 2
    assert result["original_text"] == "我们出去散不"
    assert result["corrected_text"] == "我们出去散步"
    assert result["changed"] is True
    assert result["total_tokens"] == 120

    assert result["window_size"] == 1
    assert result["prompt_type"] == "minimal"


def test_correct_all_subtitles():
    """测试 correct_all_subtitles 函数在正常情况下的行为"""
    subtitles = [
        Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=2),
            content="第一句原文",
        ),
        Subtitle(
            index=2,
            start=timedelta(seconds=2),
            end=timedelta(seconds=4),
            content="第二句原闻",
        ),
    ]

    mock_results = [
        {
            "subtitle_index": 1,
            "original_text": "第一句原文",
            "corrected_text": "第一句原文",
            "changed": False,
            "reason": "无需修改",
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "processing_time_seconds": 0.4,
        },
        {
            "subtitle_index": 2,
            "original_text": "第二句原闻",
            "corrected_text": "第二句原文",
            "changed": True,
            "reason": "修正错别字",
            "prompt_tokens": 90,
            "completion_tokens": 20,
            "total_tokens": 110,
            "processing_time_seconds": 0.5,
        },
    ]

    with (
        patch(
            "src.correction_engine.correct_single_subtitle",
            side_effect=mock_results,
        ) as mock_correct,
        patch(
            "src.correction_engine.append_result_log"
        ) as mock_log,
    ):
        corrected_subtitles, results = correct_all_subtitles(
            subtitles=subtitles,
            window_size=1,
            prompt_type="minimal",
            cache_path="cache/test.json",
            log_path="logs/test.jsonl",
        )

    assert mock_correct.call_count == 2
    assert mock_log.call_count == 2
    # assert_any_call检查模拟对象是否至少被调用过一次且参数匹配
    mock_log.assert_any_call(
        result=mock_results[0],
        file_path="logs/test.jsonl",
    )
    mock_log.assert_any_call(
        result=mock_results[1],
        file_path="logs/test.jsonl",
    )
    

    # call_args_list是Mock对象的属性,记录了该模拟对象被调用的所有历史
    for mock_call in mock_correct.call_args_list:
        assert mock_call.kwargs["cache_path"] == "cache/test.json"

    assert subtitles[0].content == "第一句原文"
    assert subtitles[1].content == "第二句原闻"
    assert corrected_subtitles[0].content == "第一句原文"
    assert corrected_subtitles[1].content == "第二句原文"
    assert corrected_subtitles is not subtitles
    assert results == mock_results


def test_correct_all_subtitles_resumes_from_log():
    """确认第一条字幕复用日志结果，只有第二条字幕进入模型校正流程"""
    subtitles = [
        Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=2),
            content="第一句原闻",  # 已校正过
        ),
        Subtitle(
            index=2,
            start=timedelta(seconds=2),
            end=timedelta(seconds=4),
            content="第二句原闻",  # 未校正过
        ),
    ]

    completed_result = {
        "subtitle_index": 1,
        "corrected_text": "第一句原文",  # 模拟日志中已有的结果
    }
    new_result = {
        "subtitle_index": 2,
        "corrected_text": "第二句原文",  # 模拟 API 新返回的结果
    }

    with (
        patch(
            "src.correction_engine.load_completed_results",
            return_value={1: completed_result},
        ) as mock_load,  # 模拟已有字幕 1 的结果
        patch(
            "src.correction_engine.correct_single_subtitle",
            return_value=new_result,
        ) as mock_correct,  # 模拟 API 返回字幕 2 的结果
        patch(
            "src.correction_engine.append_result_log",
        ) as mock_log,  # 验证日志写入
    ):
        corrected_subtitles, results = correct_all_subtitles(
            subtitles=subtitles,
            window_size=1,
            prompt_type="minimal",
            cache_path="cache/test.json",
            log_path="logs/test.jsonl",
        )

    mock_load.assert_called_once_with("logs/test.jsonl")
    mock_correct.assert_called_once_with(
        subtitles=subtitles,
        current_index=1,
        window_size=1,
        prompt_type="minimal",
        cache_path="cache/test.json",
    )
    mock_log.assert_called_once_with(
        result=new_result,
        file_path="logs/test.jsonl",
    )

    assert corrected_subtitles[0].content == "第一句原文"
    assert corrected_subtitles[1].content == "第二句原文"
    assert results == [completed_result, new_result]


def test_correct_all_subtitles_with_empty_list():
    with patch(
        "src.correction_engine.correct_single_subtitle"
    ) as mock_correct:
        corrected_subtitles, results = correct_all_subtitles(
            subtitles=[],
            window_size=3,
            prompt_type="normal",
        )

    mock_correct.assert_not_called()
    assert corrected_subtitles == []
    assert results == []


def test_correct_srt_file():
    original_subtitles = [
        Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=2),
            content="原闻",
        )
    ]
    corrected_subtitles = [
        Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=2),
            content="原文",
        )
    ]
    mock_results = [
        {
            "subtitle_index": 1,
            "original_text": "原闻",
            "corrected_text": "原文",
            "changed": True,
            "reason": "修正错别字",
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "processing_time_seconds": 0.4,
        }
    ]

    with (
        patch(
            "src.correction_engine.load_srt",
            return_value=original_subtitles,
        ) as mock_load,
        patch(
            "src.correction_engine.correct_all_subtitles",
            return_value=(corrected_subtitles, mock_results),
        ) as mock_correct_all,
        patch(
            "src.correction_engine.save_srt"
        ) as mock_save,
    ):
        results = correct_srt_file(
            input_path="input.srt",
            output_path="output.srt",
            window_size=1,
            prompt_type="minimal",
            cache_path="cache/test.json",
            log_path="logs/test.jsonl",
        )

    mock_load.assert_called_once_with("input.srt")
    # assert_called_once_with检查模拟对象是否只被调用过一次且参数匹配
    mock_correct_all.assert_called_once_with(
        subtitles=original_subtitles,
        window_size=1,
        prompt_type="minimal",
        cache_path="cache/test.json",
        log_path="logs/test.jsonl",
    )
    mock_save.assert_called_once_with(  
        subtitles=corrected_subtitles,
        file_path="output.srt",
    )
    assert results == mock_results