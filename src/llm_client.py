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