from src.result_logger import (
    append_result_log,
    load_completed_results,
    load_result_log,
)

def test_append_result_log_creates_and_appends(tmp_path):
    """测试 append_result_log 函数创建日志文件并追加记录"""
    log_path = tmp_path / "logs" / "results.jsonl"

    first_result = {
        "subtitle_index": 1,
        "original_text": "今天天汽很好",
        "corrected_text": "今天天气很好",
        "changed": True,
        "reason": "修正错别字",
    }
    second_result = {
        "subtitle_index": 2,
        "original_text": "我们出去散步",
        "corrected_text": "我们出去散步",
        "changed": False,
        "reason": "无需修改",
    }

    append_result_log(
        result=first_result,
        file_path=str(log_path),
    )
    append_result_log(
        result=second_result,
        file_path=str(log_path),
    )

    records = load_result_log(
        str(log_path)
    )

    assert len(records) == 2
    assert records[0] == first_result
    assert records[1] == second_result


def test_load_result_log_when_file_does_not_exist(tmp_path):
    """测试 load_result_log 函数在日志文件不存在时返回空列表"""
    log_path = tmp_path / "missing.jsonl"

    records = load_result_log(
        str(log_path)
    )

    assert records == []    


def test_load_completed_results_keeps_latest_record(tmp_path):
    """测试 load_completed_results 函数仅保留每个字幕编号的最新记录"""
    log_path = tmp_path / "results.jsonl"

    records = [
        {
            "subtitle_index": 1,
            "corrected_text": "第一次结果",
        },
        {
            "subtitle_index": 2,
            "corrected_text": "第二条结果",
        },
        {
            "subtitle_index": 1,
            "corrected_text": "最新结果",
        },
        {
            "corrected_text": "缺少字幕编号",
        },
    ]

    for record in records:
        append_result_log(
            result=record,
            file_path=str(log_path),
        )

    completed = load_completed_results(
        str(log_path)
    )

    assert set(completed) == {1, 2}
    assert completed[1]["corrected_text"] == "最新结果"
    assert completed[2]["corrected_text"] == "第二条结果"