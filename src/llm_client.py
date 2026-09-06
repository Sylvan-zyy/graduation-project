import json
import os

from dotenv import load_dotenv
from openai import OpenAI


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
    )


def request_json_completion(system_prompt: str, user_prompt: str):
    """请求 DeepSeek，并要求返回 JSON 格式。"""
    config = load_deepseek_config()
    client = create_deepseek_client()

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


def parse_correction_result(content: str) -> dict[str, str | bool]:
    """解析模型返回的字幕校正 JSON。"""
    data = json.loads(content)

    return {
        "corrected_text": data["corrected_text"],
        "changed": data["changed"],
        "reason": data["reason"],
    }