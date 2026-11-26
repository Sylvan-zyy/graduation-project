from srt import Subtitle
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


def test_save_srt_preserves_subtitle_numbers(tmp_path):
    subtitles = [
        Subtitle(4, timedelta(seconds=0), timedelta(seconds=1), "你好"),  # 字幕编号从 4 开始
        Subtitle(9, timedelta(seconds=1), timedelta(seconds=2), "再见"),
    ]
    output_path = str(tmp_path / "numbering.srt")  # str()把路径转成字符串

    save_srt(subtitles, output_path)
    reloaded = load_srt(output_path)

    assert [subtitle.index for subtitle in reloaded] == [4, 9]