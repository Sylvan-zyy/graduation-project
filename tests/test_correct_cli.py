import pytest

import json
from unittest.mock import patch


def test_correct_cli_passes_options_and_reports_usage(capsys):  # capsys 用来读取程序打印的 JSON
    from correct import main

    results = [
        {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "processing_time_seconds": 0.25,
        },
        {
            "failed": True,
            "usage_unknown": True,
            "processing_time_seconds": 0.5,
        },
    ]

    with patch("correct.correct_srt_file", return_value=results) as mock_correct:
        # 把命令行参数传递给 main 函数
        main([
            "--input", "input.srt",
            "--output", "output.srt",
            "--window-size", "1",
            "--prompt-type", "minimal",
        ])

    mock_correct.assert_called_once_with(
        input_path="input.srt",
        output_path="output.srt",
        window_size=1,
        prompt_type="minimal",
    )

    # capsys 用来捕获终端输出，.readouterr().out取出 main() 打印到标准输出的文本，json.loads()再把 JSON 文本解析成字典
    report = json.loads(capsys.readouterr().out)
    assert report["subtitle_count"] == 2
    assert report["failed_subtitle_count"] == 1
    assert report["api_usage"]["total_tokens"] == 100  # 两次字典取值，先取出 api_usage子字典，再从子字典取 total_tokens
    assert report["api_usage"]["unknown_usage_count"] == 1


def test_correct_cli_rejects_same_input_and_output(tmp_path, capsys):
    from correct import main

    subtitle_path = str(tmp_path / "original.srt")

    with patch("correct.correct_srt_file") as mock_correct:
        with pytest.raises(SystemExit) as error:
            main(["--input", subtitle_path, "--output", subtitle_path])

    assert error.value.code == 2
    assert "输入和输出必须使用不同文件" in capsys.readouterr().err
    mock_correct.assert_not_called()