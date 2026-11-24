import pytest

from src.evaluator import (
    calculate_cer,
    calculate_corpus_cer,
    calculate_correct_modification_rate,
    calculate_overcorrection_rate,
    summarize_api_usage,
    evaluate_srt_files,
)

def test_calculate_cer_with_identical_text():
    result = calculate_cer(
        reference_text="今天天气好",
        predicted_text="今天天气好",
    )

    assert result == 0.0


def test_calculate_cer_with_one_substitution():
    result = calculate_cer(
        reference_text="天气好",
        predicted_text="天气号",
    )

    assert result == pytest.approx(1 / 3)  #  approx浮点数近似比较工具


def test_calculate_cer_with_one_insertion():
    result = calculate_cer(
        reference_text="天气好",
        predicted_text="天气很好",
    )

    assert result == pytest.approx(1 / 3)


def test_calculate_cer_with_one_deletion():
    result = calculate_cer(
        reference_text="今天天气",
        predicted_text="今天天",
    )

    assert result == pytest.approx(1 / 4)


def test_calculate_cer_rejects_empty_reference():
    with pytest.raises(
        ValueError,
        match="reference_text 不能为空",
    ):
        calculate_cer(
            reference_text="",
            predicted_text="测试",
        )


def test_calculate_cer_rejects_non_string_reference():
    with pytest.raises(
        TypeError,
        match="reference_text 必须是字符串",
    ):
        calculate_cer(
            reference_text=None,
            predicted_text="测试",
        )


def test_calculate_cer_rejects_non_string_prediction():
    with pytest.raises(
        TypeError,
        match="predicted_text 必须是字符串",
    ):
        calculate_cer(
            reference_text="测试",
            predicted_text=None,
        )

# 测试语料库 CER 计算使用总参考长度
def test_calculate_corpus_cer_uses_total_reference_length():
    result = calculate_corpus_cer(
        reference_texts=["你好", "今天天气真好"],
        predicted_texts=["你号", "今天天气真好"],
    )

    assert result == pytest.approx(1 / 8)


def test_calculate_corpus_cer_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="条数必须一致"):
        calculate_corpus_cer(
            reference_texts=["你好", "再见"],
            predicted_texts=["你好"],
        )


def test_calculate_corpus_cer_rejects_empty_corpus():
    with pytest.raises(ValueError, match="总字符数不能为零"):
        calculate_corpus_cer(
            reference_texts=[],
            predicted_texts=[],
        )


def test_calculate_corpus_cer_counts_insertions_for_empty_reference():
    result = calculate_corpus_cer(
        reference_texts=["", "你好"],
        predicted_texts=["啊", "你好"],
    )

    assert result == pytest.approx(1 / 2)


def test_calculate_corpus_cer_preserves_subtitle_boundaries():
    result = calculate_corpus_cer(
        reference_texts=["你", "好"],
        predicted_texts=["你好", ""],
    )

    assert result == pytest.approx(1.0)

# 测试正确修改率计算只统计改进
def test_calculate_correct_modification_rate_counts_only_improvements():
    from src.evaluator import calculate_correct_modification_rate

    result = calculate_correct_modification_rate(
        reference_texts=["今天天气好", "我们去看戏", "你好", "早上好"],
        original_texts=["今夭天汽好", "我们去看细", "你好", "早上号"],
        corrected_texts=["今天天汽好", "我们去看系", "你号", "早上号"],
    )

    assert result == pytest.approx(1 / 3)


def test_calculate_correct_modification_rate_returns_none_without_changes():
    result = calculate_correct_modification_rate(
        reference_texts=["你好"],
        original_texts=["你号"],
        corrected_texts=["你号"],
    )

    assert result is None  # 返回None表示没有修改的字幕条数


def test_calculate_correct_modification_rate_returns_zero_without_improvements():
    result = calculate_correct_modification_rate(
        reference_texts=["你好"],
        original_texts=["你好"],
        corrected_texts=["你号"],
    )

    assert result == 0.0  # 返回0.0表示有修改的字幕条数，但没有改进的字幕条数

# 测试确保校正结果少了一条时会报错，避免 zip() 悄悄跳过配对不上的字幕
def test_calculate_correct_modification_rate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="条数必须一致"):
        calculate_correct_modification_rate(
            reference_texts=["你好", "再见"],
            original_texts=["你号", "再见"],
            corrected_texts=["你好"],
        )


def test_calculate_overcorrection_rate_counts_only_originally_correct_subtitles():
    result = calculate_overcorrection_rate(
        reference_texts=["你好", "再见", "早上好", "天气好"],
        original_texts=["你好", "再见", "早上好", "天气号"],
        corrected_texts=["你号", "再见", "早上好", "天汽号"],
    )

    assert result == pytest.approx(1 / 3)


# 测试过度修改率计算只统计原本正确的字幕条数
def test_calculate_overcorrection_rate_returns_none_without_correct_originals():
    result = calculate_overcorrection_rate(
        reference_texts=["你好"],
        original_texts=["你号"],
        corrected_texts=["你好"],
    )

    assert result is None  # None表示没有原本正确的字幕条数


# 测试过度修改率计算在原本正确的字幕条数中没有被改错的情况
def test_calculate_overcorrection_rate_returns_zero_when_correct_text_is_preserved():
    result = calculate_overcorrection_rate(
        reference_texts=["你好", "再见"],
        original_texts=["你好", "再件"],
        corrected_texts=["你好", "再见"],
    )

    assert result == 0.0  # 0.0表示有原本正确的字幕条数，但没有被改错的字幕条数


# 只累加本轮新API请求的Token和耗时
# 只有日志里没有已完成结果，而且请求缓存也没命中时，校正流程才需要发起新的 API 请求
def test_summarize_api_usage_counts_only_new_api_results():
    first_api_result = {
        "from_cache": False,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
        "processing_time_seconds": 0.25,
    }
    second_api_result = {
        "from_cache": False,
        "prompt_tokens": 60,
        "completion_tokens": 10,
        "total_tokens": 70,
        "processing_time_seconds": 0.5,
    }

    results = [
        first_api_result,  # 本轮新请求
        {**first_api_result, "from_cache": True},
        {**second_api_result, "from_log": True},  # from_log 表示是从日志中读取的历史结果，不计入本轮统计
        second_api_result,
    ]

    # 比较两个字典的键和对应的值，字典中字段的书写顺序不影响结果
    assert summarize_api_usage(results) == {
        "api_result_count": 2,
        "prompt_tokens": 140,
        "completion_tokens": 30,
        "total_tokens": 170,
        "api_processing_time_seconds": 0.75,
    }


def test_summarize_api_usage_returns_zero_without_new_api_results():
    results = [{"from_cache": True}, {"from_log": True}]

    assert summarize_api_usage(results) == {
        "api_result_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_processing_time_seconds": 0.0,
    }


# 测试当某条结果缺少必要的 Token 计数字段时，summarize_api_usage() 会抛出 ValueError 异常
def test_summarize_api_usage_rejects_missing_token_count():
    results = [
        {
            "from_cache": False,
            "prompt_tokens": 80,
            "total_tokens": 100,
            "processing_time_seconds": 0.25,
        },
    ]

    with pytest.raises(ValueError, match="completion_tokens"):
        summarize_api_usage(results)

# 把前面写的几个评估指标接到了 SRT 文件入口上，校验CER、正确修改率和过度修改率的计算结果是否符合预期
def test_evaluate_srt_files_compares_original_and_corrected_subtitles(tmp_path):  # tmp_path 是 pytest 自动提供的临时目录
    template = (
        "1\n00:00:00,000 --> 00:00:01,000\n{first}\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n{second}\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\n{third}\n\n"
    )
    reference_path = tmp_path / "reference.srt"  # / 用于拼接路径
    original_path = tmp_path / "original.srt"
    corrected_path = tmp_path / "corrected.srt"

    for path, texts in [
        (reference_path, ("你好", "再见", "早上好")),
        (original_path, ("你号", "再件", "早上好")),
        (corrected_path, ("你好", "再见", "早上号")),
    ]:
        first, second, third = texts
        path.write_text(
            template.format(first=first, second=second, third=third),  # 把模板字符串里的 {first}、{second}、{third} 替换成实际文本
            encoding="utf-8",
        )

    result = evaluate_srt_files(
        reference_path=str(reference_path),
        original_path=str(original_path),
        corrected_path=str(corrected_path),
    )

    assert result == pytest.approx({
        "original_cer": 2 / 7,
        "corrected_cer": 1 / 7,
        "correct_modification_rate": 2 / 3,
        "overcorrection_rate": 1.0,
    })


# @pytest.mark.parametrize把一个测试函数自动展开成多组用例，每组用不同的参数运行
@pytest.mark.parametrize(
    "number, end_time",
    [
        (2, "00:00:01,000"),
        (1, "00:00:02,000"),
    ],
)
def test_evaluate_srt_files_rejects_unaligned_subtitles(tmp_path, number, end_time):
    reference_path = tmp_path / "reference.srt"
    original_path = tmp_path / "original.srt"
    corrected_path = tmp_path / "corrected.srt"

    matching_entry = "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n"
    reference_path.write_text(matching_entry, encoding="utf-8")
    original_path.write_text(matching_entry, encoding="utf-8")
    corrected_path.write_text(
        f"{number}\n00:00:00,000 --> {end_time}\n你好\n\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="编号或时间轴不一致"):
        evaluate_srt_files(
            reference_path=str(reference_path),
            original_path=str(original_path),
            corrected_path=str(corrected_path),
        )