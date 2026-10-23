"""CER(字符错误率) = 插入、删除和替换的最少次数(编辑距离) ÷ 参考文本字符数(参考文本长度)"""
"""编辑距离：参考文本前 i 个字符 → 预测文本前 j 个字符，最少需要多少插入、删除、替换操作"""
"""动态规划表：二维表，记录参考前 i 个字符 → 预测前 j 个字符的最小编辑距离"""


def calculate_cer(
    reference_text: str,
    predicted_text: str,
) -> float:
    """计算预测文本相对于参考文本的字符错误率。"""
    if not isinstance(reference_text, str):
        raise TypeError("reference_text 必须是字符串")

    if not isinstance(predicted_text, str):
        raise TypeError("predicted_text 必须是字符串")

    if reference_text == "":
        raise ValueError("reference_text 不能为空")

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
    return edit_distance / len(reference_text)  # CER = 编辑距离 / 参考文本长度