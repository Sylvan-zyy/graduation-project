import json
from pathlib import Path


def append_result_log(
    result: dict,
    file_path: str,
) -> None:
    """把一条校正结果追加到 JSON Lines 日志。"""
    log_path = Path(file_path)
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_line = json.dumps(
        result,
        ensure_ascii=False,  # 保留中文，不转成\uXXXX
    )

    # with ... as ... 自动管理文件打开/关闭
    with log_path.open(
        mode="a",  # 追加模式（append）在文件末尾写入，不覆盖已有内容
        encoding="utf-8",
    ) as log_file:
        log_file.write(json_line + "\n")


def load_result_log(file_path: str) -> list[dict]:
    """读取 JSON Lines 日志；文件不存在时返回空列表。"""
    log_path = Path(file_path)

    if not log_path.exists():
        return []

    records = []

    with log_path.open(
        mode="r",
        encoding="utf-8",
    ) as log_file:
        for line in log_file:
            if line.strip():  # 跳过空行（只有空格或换行）
                records.append(
                    json.loads(line)  # 把每行JSON字符串解析成字典
                )

    return records        


def load_completed_results(
    file_path: str,
) -> dict[int, dict]:
    """按字幕编号读取已经完成的校正结果。"""
    completed_results = {}

    for record in load_result_log(file_path):
        subtitle_index = record.get("subtitle_index")

    # isinstance是Python内置函数，用来判断一个对象是否是某个类的实例
    # isinstance(object, classinfo)，object是要检查的对象，classinfo是类或类型元组
        if isinstance(subtitle_index, int):  # 确保字幕编号是整数
            completed_results[subtitle_index] = record

    return completed_results