from srt import Subtitle

from .context_manager import get_context
from .llm_client import request_correction
from .prompt_builder import SYSTEM_PROMPT, build_user_prompt


def correct_single_subtitle(
    subtitles: list[Subtitle],
    current_index: int,
    window_size: int = 3,
    prompt_type: str = "normal",
) -> dict[str, str | bool | int | float]:
    """结合上下文校正指定位置的一条字幕。"""
    context = get_context(
        subtitles=subtitles,
        current_index=current_index,
        window_size=window_size,
    )

    user_prompt = build_user_prompt(
        context=context,
        prompt_type=prompt_type,
    )

    result = request_correction(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return {
        "subtitle_index": subtitles[current_index].index,
        "original_text": context["current"],
        **result,
    }