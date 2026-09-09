from datetime import timedelta
from unittest.mock import patch

from srt import Subtitle

from src.correction_engine import correct_single_subtitle
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