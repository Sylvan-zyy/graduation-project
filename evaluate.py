import argparse
import json

from src.evaluator import evaluate_srt_files


# add_argument() 定义终端能接收的参数，required=True表示这一项必须提供
def main() -> None:
    parser = argparse.ArgumentParser(description="评估三份已对齐的 SRT 字幕")
    parser.add_argument("--reference", required=True, help="人工参考字幕")
    parser.add_argument("--original", required=True, help="校正前的原始字幕")
    parser.add_argument("--corrected", required=True, help="校正后的字幕")
    args = parser.parse_args()  # 解析终端输入的参数，返回一个命名空间对象，属性名就是 add_argument() 里定义的参数名

    try:
        result = evaluate_srt_files(
            reference_path=args.reference,
            original_path=args.original,
            corrected_path=args.corrected,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))  # parser.error() 会打印错误信息并退出程序，返回码为 2，是argparse的标准错误出口

    # 以 JSON 格式打印评估结果，ensure_ascii=False 保留中文，indent=2 每层缩紧2格美化输出
    print(json.dumps(result, ensure_ascii=False, indent=2))


# 保证直接运行脚本时执行，被 import 时不执行
if __name__ == "__main__":
    main()