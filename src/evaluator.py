"""CER(字符错误率) = 插入、删除和替换的最少次数(编辑距离) ÷ 参考文本字符数(参考文本长度)"""
"""编辑距离：参考文本前 i 个字符 → 预测文本前 j 个字符，最少需要多少插入、删除、替换操作"""
"""动态规划表：二维表，记录参考前 i 个字符 → 预测前 j 个字符的最小编辑距离"""

from .subtitle_io import load_srt

#  私有辅助函数（下划线开头）
def _calculate_edit_distance(
    reference_text: str,
    predicted_text: str,
) -> int:
    """计算两个字符串之间的字符编辑距离。"""
    if not isinstance(reference_text, str):
        raise TypeError("reference_text 必须是字符串")

    if not isinstance(predicted_text, str):
        raise TypeError("predicted_text 必须是字符串")


    # 编辑距离算法的核心，表示参考文本为空字符串时，变成预测文本需要多少次操作
    # 初始化动态规划表
    previous_row = list(range(len(predicted_text) + 1))  

    # enumerate()用于在遍历可迭代对象时，每次产出一个(索引, 元素)元组
    # 外层循环：遍历参考文本中的每个字符，并计算与预测文本的编辑距离
    for reference_index, reference_char in enumerate(
        reference_text,
        start=1,  # 默认为0，这里从1开始，把0行作为参考文本为空字符串的情况
    ): 
        # 当前行中第一个元素 = 参考文本索引，初始化为一个只包含一个元素的列表
        current_row = [reference_index]

        # 内层：遍历预测文本中的每个字符，并计算插入、删除和替换的成本
        for predicted_index, predicted_char in enumerate(
            predicted_text,
            start=1,
        ):
            # 插入成本 = 当前行的前一个元素 + 1
            # (dp[i][j]=dp[i][j-1]+1)，在预测文本中插入一个字符
            insertion_cost = current_row[predicted_index - 1] + 1  

            # 删除成本 = 上一行的当前索引元素 + 1，在预测文本中删除一个字符
            # dp[i][j]=dp[i-1][j]+1，previous_row是上一行i-1的编辑距离，predicted_index是j
            deletion_cost = previous_row[predicted_index] + 1

            # 替换成本 = 上一行的前一个元素 + (参考字符 != 预测字符)
            # (dp[i-1][j-1]) + cost，如果字符相同，成本为 0；否则为 1
            substitution_cost = (
                previous_row[predicted_index - 1]
                + (reference_char != predicted_char)  # 布尔值，字符相同 → False → 0
            )

            # 选择成本最小值，并添加到当前行
            # 动态规划的核心：当前状态的最优解 = 三种操作中成本最小的那个
            current_row.append(
                min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
            )

        previous_row = current_row  # 把当前行作为下一轮的上一行，继续迭代

    # 动态规划表的最后一个元素 = 编辑距离
    # previous_row[-1] 是最后一行的最后一个元素，列表从右往左取第一个元素
    edit_distance = previous_row[-1]
    return edit_distance  # 返回编辑距离


def calculate_cer(
    reference_text: str,
    predicted_text: str,
) -> float:
    """计算单条字幕的字符错误率。"""
    edit_distance = _calculate_edit_distance(
        reference_text,
        predicted_text,
    )

    if reference_text == "":
        raise ValueError("reference_text 不能为空")

    return edit_distance / len(reference_text)


def calculate_corpus_cer(
    reference_texts: list[str],
    predicted_texts: list[str],
) -> float:
    """按位置配对，计算总编辑距离除以总参考字符数。"""
    if not isinstance(reference_texts, list):
        raise TypeError("reference_texts 必须是列表")

    if not isinstance(predicted_texts, list):
        raise TypeError("predicted_texts 必须是列表")

    if len(reference_texts) != len(predicted_texts):
        raise ValueError("参考字幕与预测字幕的条数必须一致")

    total_edit_distance = 0
    total_reference_chars = 0

# reference_texts是整个参考字幕列表，reference_text是其中的一条字幕，predicted同理
# zip()每轮取一对同位置的元素，再把这一对分别赋给两个变量
    for reference_text, predicted_text in zip(
        reference_texts,
        predicted_texts,
    ):
        total_edit_distance += _calculate_edit_distance(
            reference_text,
            predicted_text,
        )
        total_reference_chars += len(reference_text)

    if total_reference_chars == 0:
        raise ValueError("参考字幕的总字符数不能为零")  # 作用在字符串上，加的是当前这一轮字幕的字符数

    return total_edit_distance / total_reference_chars  # 循环结束后才做一次除法


# 计算正确修改率
# 正确率 = 校正后编辑距离严格变小的字幕条数 / 校正文本与原始文本不同的字幕条数
# 正确率 = 已修改字幕中更接近参考的比例 = 改进的字幕数 / 修改的字幕数
def calculate_correct_modification_rate(
    reference_texts: list[str],  # 人工核对过的参考字幕
    original_texts: list[str],  # 原始字幕
    corrected_texts: list[str],  # 大模型校正后的字幕
) -> float | None:
    """返回已修改字幕中更接近参考的比例。"""
    subtitle_lists = (reference_texts, original_texts, corrected_texts)
    if not all(isinstance(texts, list) for texts in subtitle_lists):
        raise TypeError("参考、原始和校正字幕都必须是列表")

    if not (len(reference_texts) == len(original_texts) == len(corrected_texts)):
        raise ValueError("参考、原始和校正字幕的条数必须一致")

    # 统计已修改字幕数和改进的字幕数
    modified_count = 0
    improved_count = 0  # 只有校正后的距离严格小于校正前的距离，count才加1

    # zip()每次取出同一位置的三个元素，再分别赋给三个变量
    # 带 s 的变量是整个列表，循环里的三个变量是本轮处理的字幕文本
    for reference, original, corrected in zip(
        reference_texts, original_texts, corrected_texts
    ):
        if not all(isinstance(text, str) for text in (reference, original, corrected)):
            raise TypeError("每条字幕都必须是字符串")

        if corrected == original:
            continue  # 比较校正前后的文本，如果两者相同，跳过本轮剩余代码，直接处理下一条字幕

        modified_count += 1
        original_distance = _calculate_edit_distance(reference, original)
        corrected_distance = _calculate_edit_distance(reference, corrected)

        if corrected_distance < original_distance:
            improved_count += 1

    if modified_count == 0:
        return None

    return improved_count / modified_count


# 计算过度修改率
# 先找出原本就和人工参考字幕完全一致的条目，再看其中有多少条被模型改得和参考字幕不一致
def calculate_overcorrection_rate(
    reference_texts: list[str],
    original_texts: list[str],
    corrected_texts: list[str],
) -> float | None:
    """返回原本正确的字幕中被改错的比例。"""
    subtitle_lists = (reference_texts, original_texts, corrected_texts)
    if not all(isinstance(texts, list) for texts in subtitle_lists):
        raise TypeError("参考、原始和校正字幕都必须是列表")

    if not (len(reference_texts) == len(original_texts) == len(corrected_texts)):
        raise ValueError("参考、原始和校正字幕的条数必须一致")

    originally_correct_count = 0
    overcorrected_count = 0

    for reference, original, corrected in zip(
        reference_texts, original_texts, corrected_texts
    ):
        if not all(isinstance(text, str) for text in (reference, original, corrected)):
            raise TypeError("每条字幕都必须是字符串")

        if original != reference:  # 如果原始文本和参考文本不同，说明它原本就有错，此时进入下一轮循环
            continue

        originally_correct_count += 1
        if corrected != reference:  # 第二个if只在原本正确的字幕上执行，此时如果校正后的字幕和参考字幕不同，说明它被改错了
            overcorrected_count += 1

    if originally_correct_count == 0:
        return None

    # 过度修改率 = 原本正确的字幕中被改错的比例 = 改错的字幕数 / 原本正确的字幕数
    return overcorrected_count / originally_correct_count


def summarize_api_usage(
    results: list[dict[str, str | bool | int | float]],
) -> dict[str, int | float]:
    """汇总本轮请求的耗时、已知 Token 和用量未知的次数。"""
    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    api_result_count = 0
    unknown_usage_count = 0
    api_processing_time_seconds = 0.0

    for result in results:
        if result.get("from_log") or result.get("from_cache"):
            continue

        seconds = result.get("processing_time_seconds")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or seconds < 0
        ):
            raise ValueError("processing_time_seconds 必须是非负数")

        api_result_count += 1
        api_processing_time_seconds += seconds

        if result.get("usage_unknown") is True:
            unknown_usage_count += 1
            continue

        for field in token_totals:
            tokens = result.get(field)
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise ValueError(f"{field} 必须是非负整数")
            token_totals[field] += tokens

    return {
        "api_result_count": api_result_count,
        "unknown_usage_count": unknown_usage_count,
        **token_totals,
        "api_processing_time_seconds": api_processing_time_seconds,
    }


# 评估 SRT 文件
def evaluate_srt_files(
    reference_path: str,
    original_path: str,
    corrected_path: str,
) -> dict[str, float | None]:
    """读取三份对齐的 SRT，并计算字幕校正的质量指标。"""
    # load_srt() 会把 SRT 文件解析成 Subtitle 对象列表，字幕条数、编号和时间轴(start, end)必须一致
    # 对齐检查用前三个属性，计算指标用 content
    reference_subtitles = load_srt(reference_path)
    original_subtitles = load_srt(original_path)
    corrected_subtitles = load_srt(corrected_path)

    # label 用于报错时指出哪一份文件有问题，每轮先检查条数,若不同则抛出异常
    for label, subtitles in (
        ("原始字幕", original_subtitles),
        ("校正字幕", corrected_subtitles),
    ):
        if len(subtitles) != len(reference_subtitles):
            raise ValueError(f"{label}与参考字幕的条数必须一致")

        # position 是这次比较到列表里的第几条，`subtitle.index` 是 SRT 文件里写的字幕编号
        for position, (reference, subtitle) in enumerate(
            zip(reference_subtitles, subtitles), start=1
        ):
            # zip() 负责按现有顺序配对，若值相等则字幕对齐。元组比较负责验证配对是否正确
            reference_key = (reference.index, reference.start, reference.end)
            subtitle_key = (subtitle.index, subtitle.start, subtitle.end)
            if subtitle_key != reference_key:
                raise ValueError(
                    f"{label}第 {position} 条与参考字幕的编号或时间轴不一致"
                )

    reference_texts = [subtitle.content for subtitle in reference_subtitles]
    original_texts = [subtitle.content for subtitle in original_subtitles]
    corrected_texts = [subtitle.content for subtitle in corrected_subtitles]

    return {
        "original_cer": calculate_corpus_cer(reference_texts, original_texts),
        "corrected_cer": calculate_corpus_cer(reference_texts, corrected_texts),
        "correct_modification_rate": calculate_correct_modification_rate(
            reference_texts, original_texts, corrected_texts
        ),
        "overcorrection_rate": calculate_overcorrection_rate(
            reference_texts, original_texts, corrected_texts
        ),
    }