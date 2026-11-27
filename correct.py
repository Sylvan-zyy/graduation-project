import argparse
import json
from pathlib import Path

from srt import SRTParseError

from src.correction_engine import correct_srt_file
from src.evaluator import summarize_api_usage


# main(argv) 让测试可以直接传入模拟参数
def main(argv: list[str] | None = None) -> None:
    # 如果 argv 是 None，就从命令行读取参数；否则就使用传入的参数列表
    # 在终端运行时，argv会读取并解析命令行参数；在测试时，argv可以传入模拟参数列表
    parser = argparse.ArgumentParser(description="批量校正 SRT 字幕")
    parser.add_argument("--input", required=True, help="原始 SRT 文件")
    parser.add_argument("--output", required=True, help="校正后的 SRT 文件")
    parser.add_argument("--window-size", type=int, default=3, help="前后各取几条字幕")
    parser.add_argument("--prompt-type", choices=("normal", "minimal"), default="normal")
    args = parser.parse_args(argv)

    if args.window_size < 0:
        parser.error("--window-size 不能小于 0")

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        if input_path.resolve() == output_path.resolve() or (  # .resolve() 把路径转成绝对路径并解析符号链接
            input_path.exists()
            and output_path.exists()
            and input_path.samefile(output_path)  # samefile() 检查两个路径是否指向同一个文件，前提是两个文件都存在
        ):
            parser.error("输入和输出必须使用不同文件")

        results = correct_srt_file(
            input_path=args.input,
            output_path=args.output,
            window_size=args.window_size,
            prompt_type=args.prompt_type,
        )
        report = {
            "subtitle_count": len(results),
            "failed_subtitle_count": sum(
                result.get("failed") is True for result in results
            ),
            "api_usage": summarize_api_usage(results),
        }
    except (OSError, ValueError, RuntimeError, SRTParseError) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()