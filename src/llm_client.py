import json,os,time

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI, OpenAIError

from .cache_manager import build_cache_key, load_cache, save_cache


def load_deepseek_config() -> dict[str, str]:
    """从 .env 文件中读取 DeepSeek API 配置。"""
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")

    return {
        "api_key": api_key,
        "base_url": os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
        "model": os.getenv(
            "DEEPSEEK_MODEL",
            "deepseek-v4-flash",
        ),
    }


def create_deepseek_client() -> OpenAI:
    """根据环境配置创建 DeepSeek 客户端。"""
    config = load_deepseek_config()

    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=30.0,
    )


def request_json_completion(system_prompt: str, user_prompt: str):
    """请求 DeepSeek，并要求返回 JSON 格式。"""
    config = load_deepseek_config()
    client = create_deepseek_client()

    try:
        return client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=512,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    except APITimeoutError as exc:
        raise RuntimeError("DeepSeek API 请求超时") from exc
    except OpenAIError as exc:
        raise RuntimeError("DeepSeek API 请求失败") from exc
    

def parse_correction_result(
    content: str | None,
) -> dict[str, str | bool]:
    """解析模型返回的字幕校正 JSON。"""
    if not content or not content.strip():  # strip()去掉首尾空白字符
        raise ValueError("模型返回内容为空")

    try:
        data = json.loads(content)  # dumps把字典变成JSON字符串，解析函数用loads把JSON字符串读回Python对象
    except json.JSONDecodeError as exc:
        raise ValueError("模型返回的内容不是有效 JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("模型返回的 JSON 必须是对象")

    for field in ("corrected_text", "changed", "reason"):
        if field not in data:
            raise ValueError(f"模型返回缺少字段：{field}")

    corrected_text = data["corrected_text"]
    changed = data["changed"]
    reason = data["reason"]

    if not isinstance(corrected_text, str) or not corrected_text.strip():
        raise ValueError("corrected_text 必须是非空字符串")
    if not isinstance(changed, bool):
        raise ValueError("changed 必须是布尔值")
    if not isinstance(reason, str):
        raise ValueError("reason 必须是字符串")

    return {
        "corrected_text": corrected_text,
        "changed": changed,
        "reason": reason,
    }


def request_correction(
    system_prompt: str,
    user_prompt: str,
    cache_path: str | None = None,
) -> dict[str, str | bool | int | float]:
    """请求字幕校正，并返回校正结果与用量指标。"""
    cache: dict[str, dict] = {}
    cache_key = None

    if cache_path is not None:
        config = load_deepseek_config()
        cache = load_cache(cache_path)
        cache_key = build_cache_key(
            model=config["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return {
                **cached_result,
                "from_cache": True,
            }

        
    start_time = time.perf_counter()

    response = request_json_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    processing_time = time.perf_counter() - start_time
    content = response.choices[0].message.content if response.choices else None
    usage = response.usage

    result: dict[str, str | bool | int | float] = {
        "processing_time_seconds": processing_time,
        "from_cache": False,
    }
    if usage is not None:
        result.update(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    try:
        correction = parse_correction_result(content)
    except ValueError as error:
        return {**result, "failed": True, "reason": str(error)}

    result.update(correction)

    if cache_path is not None and cache_key is not None:
        cache[cache_key] = result
        save_cache(
            cache=cache,
            file_path=cache_path,
    )

    return result