SYSTEM_PROMPT = """
你是一名影视字幕校正助手。
你的任务是纠正当前字幕中的语音识别错误、错别字和明显的标点错误。
参考字幕只用于理解上下文，不得修改参考字幕。
请保持说话人的原意和表达风格。
""".strip()


def format_context_items(items: list[str]) -> str:
    """将多条上下文字幕转换成适合提示词的文本。"""
    if not items:
        return "（无）"

    return "\n".join(f"- {item}" for item in items)


def build_user_prompt(
    context: dict[str, object],
    prompt_type: str = "normal",
) -> str:
    """根据上下文和提示类型构造用户提示词。"""
    if prompt_type not in ("normal", "minimal"):
        raise ValueError(f"不支持的提示类型：{prompt_type}")

    previous_items = context["previous"]
    current_text = context["current"]
    next_items = context["next"]

    if not isinstance(previous_items, list):
        raise TypeError("previous必须是列表")

    if not isinstance(current_text, str):
        raise TypeError("current必须是字符串")

    if not isinstance(next_items, list):
        raise TypeError("next必须是列表")

    if prompt_type == "normal":
        correction_instruction = (
            "请根据上下文校正当前字幕中的错误，使表达准确、通顺。"
        )
    else:
        correction_instruction = (
            "只修改能够根据上下文明确判断的错误，不要进行润色或改写；"
            "如果证据不足，请原样保留当前字幕。"
        )

    previous_text = format_context_items(previous_items)
    next_text = format_context_items(next_items)

    return f"""
【前文字幕】
{previous_text}

【当前字幕】
{current_text}

【后文字幕】
{next_text}

【校正要求】
{correction_instruction}

请只返回以下JSON格式，不要输出其他内容：
{{
  "corrected_text": "校正后的当前字幕",
  "changed": true,
  "reason": "修改原因"
}}
""".strip()