import srt


def get_context(
    subtitles: list[srt.Subtitle],
    current_index: int,
    window_size: int = 3,
) -> dict[str, object]:
    """提取当前字幕前后的上下文。"""
    if window_size < 0:
        raise ValueError("窗口大小不能小于0")

    if current_index < 0 or current_index >= len(subtitles):
        raise IndexError("当前字幕位置超出范围")

    start_index = max(0, current_index - window_size)
    end_index = min(len(subtitles), current_index + window_size + 1)

    previous_items = subtitles[start_index:current_index]
    next_items = subtitles[current_index + 1:end_index]

    return {
        "previous": [item.content for item in previous_items],
        "current": subtitles[current_index].content,
        "next": [item.content for item in next_items],
    }