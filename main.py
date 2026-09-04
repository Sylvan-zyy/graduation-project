from src.context_manager import get_context
from src.subtitle_io import load_srt, save_srt


def main() -> None:
    """程序入口。"""
    input_path = "data/input/sample.srt"
    output_path = "data/output/sample_corrected.srt"

    subtitles = load_srt(input_path)

    print(f"成功读取 {len(subtitles)} 条字幕：")

    for item in subtitles:
        print(f"{item.index}: {item.content}")

    context = get_context(
        subtitles=subtitles,
        current_index=2,
        window_size=1,
    )

    print("\n第3条字幕的上下文：")
    print(f"前文：{context['previous']}")
    print(f"当前：{context['current']}")
    print(f"后文：{context['next']}")

    subtitles[2].content = "回来以后继续学习Python编程。"

    save_srt(subtitles, output_path)
    print(f"\n校正后的字幕已保存到：{output_path}")


if __name__ == "__main__":
    main()