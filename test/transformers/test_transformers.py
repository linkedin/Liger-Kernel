import pytest


def test_import_from_root():
    try:
        from liger_kernel.transformers import (  # noqa: F401
            LigerBlockSparseTop2MLP,
            LigerCrossEntropyLoss,
            LigerFusedLinearCrossEntropyLoss,
            LigerGEGLUMLP,
            LigerLayerNorm,
            LigerPhi3SwiGLUMLP,
            LigerRMSNorm,
            LigerSwiGLUMLP,
            liger_rotary_pos_emb,
        )
    except Exception:
        pytest.fail("Import kernels from root fails")
