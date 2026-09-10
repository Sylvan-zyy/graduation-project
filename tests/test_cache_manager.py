from src.cache_manager import (
    build_cache_key,
    load_cache,
    save_cache,
) 


def test_build_cache_key_is_stable():
    first_key = build_cache_key(
        model="test-model",
        system_prompt="系统提示",
        user_prompt="用户提示",
    )
    second_key = build_cache_key(
        model="test-model",
        system_prompt="系统提示",
        user_prompt="用户提示",
    )

    assert first_key == second_key
    assert len(first_key) == 64


def test_build_cache_key_changes_with_context():
    first_key = build_cache_key(
        model="test-model",
        system_prompt="系统提示",
        user_prompt="前文一，当前字幕，后文一",
    )
    second_key = build_cache_key(
        model="test-model",
        system_prompt="系统提示",
        user_prompt="前文二，当前字幕，后文二",
    )

    assert first_key != second_key


def test_load_cache_when_file_does_not_exist(tmp_path):
    # tmp_path它都会自动创建一个临时文件夹，测试结束后自动删除
    # 在 tmp_path 这个目录下，创建一个名为 missing.json 的文件
    cache_path = tmp_path / "missing.json"

    result = load_cache(str(cache_path))

    assert result == {}


def test_load_cache_from_json_file(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        '{"test-key": {"corrected_text": "校正文本"}}',
        encoding="utf-8",
    )

    result = load_cache(str(cache_path))

    assert result == {
        "test-key": {
            "corrected_text": "校正文本",
        }
    }    


def test_save_cache_creates_directory_and_file(tmp_path):
    cache_path = tmp_path / "nested" / "cache.json"
    cache = {
        "test-key": {
            "corrected_text": "校正后的字幕",
            "changed": True,
        }
    }

    save_cache(
        cache=cache,
        file_path=str(cache_path),
    )

    assert cache_path.exists()
    assert load_cache(str(cache_path)) == cache    