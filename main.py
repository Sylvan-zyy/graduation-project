from src.subtitle_io import load_srt, save_srt


def main() -> None:
    """程序入口。"""
    input_path = "data/input/sample.srt"
    output_path = "data/output/sample_corrected.srt"

    subtitles = load_srt(input_path)

    print(f"成功读取 {len(subtitles)} 条字幕：")

    for item in subtitles:
        print(f"{item.index}: {item.content}")

    subtitles[2].content = "回来以后继续学习Python编程。"

    save_srt(subtitles, output_path)
    print(f"校正后的字幕已保存到：{output_path}")


if __name__ == "__main__":
    main()