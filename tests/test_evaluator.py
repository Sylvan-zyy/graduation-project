import pytest

from src.evaluator import calculate_cer, calculate_corpus_cer


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