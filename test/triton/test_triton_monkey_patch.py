import secrets

import pytest


def generate_random_hex(length=16):
    return secrets.token_hex(length // 2)


def test_import_from_root():
    try:
        from liger_kernel.triton import apply_liger_triton_cache_manager  # noqa: F401
    except Exception:
        pytest.fail("Import kernel patch from root fails")


def test_import_custom_cache_manager():
    from triton.runtime.cache import get_cache_manager

    from liger_kernel.triton import apply_liger_triton_cache_manager

    apply_liger_triton_cache_manager()
    random_hex_key = generate_random_hex(16)
    cache_manager = get_cache_manager(key=random_hex_key)
    from liger_kernel.triton.monkey_patch import LigerTritonFileCacheManager

    assert isinstance(cache_manager, LigerTritonFileCacheManager), (
        "Cache manager should have been LigerTritonFileCacheManager"
    )
