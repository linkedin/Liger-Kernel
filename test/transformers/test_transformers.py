import pytest


def test_import_from_root():
    try:
        from liger_kernel.transformers import LigerBlockSparseTop2MLP  # noqa: F401
        from liger_kernel.transformers import LigerCrossEntropyLoss  # noqa: F401
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # noqa: F401
        from liger_kernel.transformers import LigerGEGLUMLP  # noqa: F401
        from liger_kernel.transformers import LigerLayerNorm  # noqa: F401
        from liger_kernel.transformers import LigerPhi3SwiGLUMLP  # noqa: F401
        from liger_kernel.transformers import LigerRMSNorm  # noqa: F401
        from liger_kernel.transformers import LigerSwiGLUMLP  # noqa: F401
        from liger_kernel.transformers import liger_rotary_pos_emb  # noqa: F401
    except Exception:
        pytest.fail("Import kernels from root fails")
