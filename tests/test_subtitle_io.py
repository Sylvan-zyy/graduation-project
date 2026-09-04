from datetime import timedelta

from src.subtitle_io import load_srt, save_srt


def test_load_srt() -> None:
    """测试能否正确读取示例字幕。"""
    subtitles = load_srt("data/input/sample.srt")

    assert len(subtitles) == 5
    assert subtitles[0].content == "今天天气很好。"
    assert subtitles[2].content == "回来以后继续学习派森编程。"
    assert subtitles[2].start == timedelta(seconds=8.5)
    assert subtitles[2].end == timedelta(seconds=12)


def test_save_and_reload_srt(tmp_path) -> None:
    """测试字幕保存后能否再次正确读取。"""
    subtitles = load_srt("data/input/sample.srt")
    output_path = tmp_path / "output.srt"

    save_srt(subtitles, str(output_path))
    reloaded_subtitles = load_srt(str(output_path))

    assert output_path.exists()
    assert len(reloaded_subtitles) == 5
    assert reloaded_subtitles[2].content == "回来以后继续学习派森编程。"
    assert reloaded_subtitles[2].start == timedelta(seconds=8.5)
    assert reloaded_subtitles[2].end == timedelta(seconds=12)