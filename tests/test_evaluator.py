import pytest

from src.evaluator import calculate_cer


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

    assert result == pytest.approx(1 / 3)


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