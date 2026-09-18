from copy import deepcopy

from srt import Subtitle

from .context_manager import get_context
from .llm_client import request_correction
from .prompt_builder import SYSTEM_PROMPT, build_user_prompt
from .result_logger import append_result_log
from .subtitle_io import load_srt, save_srt


def correct_single_subtitle(
    subtitles: list[Subtitle],
    current_index: int,
    window_size: int = 3,
    prompt_type: str = "normal",
    cache_path: str | None = None,
) -> dict[str, str | bool | int | float]:
    """结合上下文校正单条字幕，并返回校正结果字典"""
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
        cache_path=cache_path,
    )

    return {
    "subtitle_index": subtitles[current_index].index,
    "original_text": context["current"],
    "window_size": window_size,
    "prompt_type": prompt_type,
    **result,
}


def correct_all_subtitles(
    subtitles: list[Subtitle],
    window_size: int = 3,
    prompt_type: str = "normal",
    cache_path: str | None = None,
    log_path: str | None = None,
) -> tuple[
    list[Subtitle],
    list[dict[str, str | bool | int | float]],
]:
    """结合上下文校正所有字幕，并返回校正后的字幕列表和结果列表"""
    corrected_subtitles = deepcopy(subtitles)
    results = []

    for current_index in range(len(subtitles)):
        result = correct_single_subtitle(
            subtitles=subtitles,
            current_index=current_index,
            window_size=window_size,
            prompt_type=prompt_type,
            cache_path=cache_path,
        )

        corrected_text = result["corrected_text"]
        if not isinstance(corrected_text, str):
            raise TypeError("corrected_text 必须是字符串")

        corrected_subtitles[current_index].content = corrected_text
        results.append(result)
        if log_path is not None:
            append_result_log(
                result=result,
                file_path=log_path,
            )

    return corrected_subtitles, results


def correct_srt_file(
    input_path: str,
    output_path: str,
    window_size: int = 3,
    prompt_type: str = "normal",
    cache_path: str | None = None,
    log_path: str | None = None,
) -> list[dict[str, str | bool | int | float]]:
    """读取、校正并保存一个完整的 SRT 文件。"""
    subtitles = load_srt(input_path)

    corrected_subtitles, results = correct_all_subtitles(
        subtitles=subtitles,
        window_size=window_size,
        prompt_type=prompt_type,
        cache_path=cache_path,
        log_path=log_path,
    )

    save_srt(
        subtitles=corrected_subtitles,
        file_path=output_path,
    )

    return results