import hashlib
import json
from pathlib import Path


def build_cache_key(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """根据模型和完整提示词生成唯一缓存键。"""
    cache_source = json.dumps(  # 把 Python 字典变成 JSON 字符串
        {
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(  # 用 SHA-256 算法计算哈希值
        cache_source.encode("utf-8")  # 把 JSON 字符串转换成字节序列（哈希函数需要）
    ).hexdigest()  # 把哈希值转换成 64 位的十六进制字符串


def load_cache(file_path: str) -> dict[str, dict]:
    """从 JSON 文件读取缓存；文件不存在时返回空字典。"""
    cache_path = Path(file_path)

    if not cache_path.exists():
        return {}

    content = cache_path.read_text(encoding="utf-8")
    if not content.strip():
        return {}

    # 把 JSON 字符串解析成 Python 字典并返回
    return json.loads(content)


def save_cache(
    cache: dict[str, dict],
    file_path: str,
) -> None:
    """把缓存字典保存为 JSON 文件。"""
    cache_path = Path(file_path)
    cache_path.parent.mkdir(
        parents=True,  # 创建所有缺失的父目录
        exist_ok=True,  # 如果目录已存在，不报错
    )

    cache_path.write_text(
        json.dumps(
            cache,  # 要序列化的Python字典
            ensure_ascii=False,  # 保留中文
            indent=2,  # 缩进2个空格，格式化输出
        ),
        encoding="utf-8",
    )