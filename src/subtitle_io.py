from pathlib import Path

import srt


def load_srt(file_path: str) -> list[srt.Subtitle]:
    """读取SRT文件并返回字幕列表。"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到字幕文件：{path}")

    content = path.read_text(encoding="utf-8-sig")
    subtitles = list(srt.parse(content))

    return subtitles


def save_srt(subtitles: list[srt.Subtitle], file_path: str) -> None:
    """将字幕列表保存为SRT文件。"""
    path = Path(file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    content = srt.compose(subtitles)
    path.write_text(content, encoding="utf-8")