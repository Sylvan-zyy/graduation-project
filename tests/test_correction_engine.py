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
        )

    request_arguments = mock_request.call_args.kwargs

    assert request_arguments["system_prompt"] == SYSTEM_PROMPT
    assert "今天天气不错" in request_arguments["user_prompt"]
    assert "我们出去散不" in request_arguments["user_prompt"]
    assert "记得带上雨伞" in request_arguments["user_prompt"]

    assert result["subtitle_index"] == 2
    assert result["original_text"] == "我们出去散不"
    assert result["corrected_text"] == "我们出去散步"
    assert result["changed"] is True
    assert result["total_tokens"] == 120


def test_correct_all_subtitles():
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

    with patch(
        "src.correction_engine.correct_single_subtitle",
        side_effect=mock_results,
    ) as mock_correct:
        corrected_subtitles, results = correct_all_subtitles(
            subtitles=subtitles,
            window_size=1,
            prompt_type="minimal",
        )

    assert mock_correct.call_count == 2
    assert subtitles[0].content == "第一句原文"
    assert subtitles[1].content == "第二句原闻"
    assert corrected_subtitles[0].content == "第一句原文"
    assert corrected_subtitles[1].content == "第二句原文"
    assert corrected_subtitles is not subtitles
    assert results == mock_results


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
        )

    mock_load.assert_called_once_with("input.srt")
    mock_correct_all.assert_called_once_with(
        subtitles=original_subtitles,
        window_size=1,
        prompt_type="minimal",
    )
    mock_save.assert_called_once_with(
        subtitles=corrected_subtitles,
        file_path="output.srt",
    )
    assert results == mock_results