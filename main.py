from src.context_manager import get_context
from src.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.subtitle_io import load_srt


def main() -> None:
    """程序入口。"""
    input_path = "data/input/sample.srt"
    subtitles = load_srt(input_path)

    context = get_context(
        subtitles=subtitles,
        current_index=2,
        window_size=1,
    )

    normal_prompt = build_user_prompt(
        context=context,
        prompt_type="normal",
    )

    minimal_prompt = build_user_prompt(
        context=context,
        prompt_type="minimal",
    )

    print("========== 系统提示词 ==========")
    print(SYSTEM_PROMPT)

    print("\n========== 普通校正提示 ==========")
    print(normal_prompt)

    print("\n========== 最小修改提示 ==========")
    print(minimal_prompt)


if __name__ == "__main__":
    main()