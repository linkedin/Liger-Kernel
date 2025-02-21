import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import create_mask
from torch.nn.attention.flex_attention import flex_attention

from liger_kernel.utils import infer_device


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def prefix_mask(b, h, q_idx, kv_idx, rejected_index, chosen_index):
    return (~((q_idx >= rejected_index[b]) & (chosen_index[b] <= kv_idx) & (kv_idx < rejected_index[b]))) & (
        q_idx >= kv_idx
    )


device = infer_device()
set_seed(42)


def _test_correctness_flex(B, H, S, D, mask_func, dtype, atol, rtol, device=infer_device()):
    """
    Test attention mechanisms with various implementations.

    Parameters:
        B (int): Batch size
        H (int): Number of attention heads
        S (int): Sequence length
        D (int): Hidden dimension per head
        mask_func: A function that generates custom attention mask
        dtype: Data type for computation
        atol (float): Absolute tolerance for comparison
        rtol (float): Relative tolerance for comparison
    """
    torch.manual_seed(0)

    # Initialize input tensors, i.e. the tensors after q, k, and v projections of hidden states (attention head input)
    query_torch = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    key_torch = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    value_torch = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)

    query_flex = query_torch.clone().detach().requires_grad_(True)
    key_flex = key_torch.clone().detach().requires_grad_(True)
    value_flex = value_torch.clone().detach().requires_grad_(True)

    block_mask = create_block_mask(mask_func, B, H, S, S, device=device)  # Sparsity block mask
    mask = create_mask(mask_func, B, H, S, S, device=device)  # Regular mask

    # If you are using a causal mask with FA2, you can enable `is_causal`."
    # e.g.,
    # F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

    torch_out = F.scaled_dot_product_attention(query_torch, key_torch, value_torch, attn_mask=mask)

    flex_out = flex_attention(query_flex, key_flex, value_flex, block_mask=block_mask)

    # Check forward pass
    assert_verbose_allclose(flex_out, torch_out, atol=atol, rtol=rtol)

    grad_out = torch.ones_like(torch_out)
    torch_out.backward(grad_out)
    flex_out.backward(grad_out)

    # Check gradients
    assert_verbose_allclose(query_flex.grad, query_torch.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(key_flex.grad, key_torch.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(value_flex.grad, value_torch.grad, atol=atol, rtol=rtol)


def _is_flex_attention_supported():
    """Check if flex attention is supported on the current device"""
    device = infer_device()
    return device in ["cuda"]


@pytest.mark.skipif(not _is_flex_attention_supported(), reason="FlexAttention is only supported on CUDA or CPU devices")
@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (2, 8, 1024, 32),
        (3, 12, 2048, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            3e-2,
            5e-1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e-2, 5e-3),
        (torch.float32, 1e-3, 5e-4),
    ],
)
def test_correctness_flex(B, H, S, D, dtype, atol, rtol):
    _test_correctness_flex(B, H, S, D, causal_mask, dtype, atol, rtol)

    # Roughly generate custom rejected and chosen indices for each batch
    chosen_index = torch.randint(0, S // 2, (B,), device=infer_device())
    rejected_index = torch.randint(S // 2, S, (B,), device=infer_device())

    def wrapped_prefix_mask(b, h, q_idx, kv_idx):
        return prefix_mask(b, h, q_idx, kv_idx, rejected_index, chosen_index)

    _test_correctness_flex(B, H, S, D, wrapped_prefix_mask, dtype, atol, rtol)


def _test_correctness_prefix(
    B=2,
    H=8,
    P=512,
    C=256,
    R=256,
    D=32,
    dtype=torch.float32,
    atol=1e-3,
    rtol=5e-4,
    device=infer_device(),
):
    """
    Test that prefix sharing attention matches separate computations (i.e. two separate casual masked attention, prefix+chosen and prefix+rejected).
    The mental model is:

    A. prefix + chosen
    P
    P P
    P P P
    P P P C
    P P P C C
    P P P C C C

    B. prefix + rejected
    P
    P P
    P P P
    P P P R
    P P P R R
    P P P R R R

    C. shared prefix + chosen + rejected
    P
    P P
    P P P
    P P P C
    P P P C C
    P P P C C C
    P P P       R
    P P P       R R
    P P P       R R R


    We test them as below to ensure attention value equivalence:
    1. prefix of shared attn (upper of C.) == prefix of chosen attn (upper of A.)
    2. prefix of shared attn (upper of C.) == prefix of rejected attn (upper of B.)
    P       P
    P P   = P P
    P P P   P P P

    3. prefix of shared attn (middle right of C.) == prefix of chosen attn (lower right of A.)
    C       C
    C C   = C C
    C C C   C C C

    4. prefix of shared attn (lower right of C.) == prefix of rejected attn (lower right of B.)
    R       R
    R R   = R R
    R R R   R R R

    Args:
        B: batch size
        H: number of heads
        P: prefix length
        C: chosen response length
        R: rejected response length
        D: hidden dimension per head
    """
    torch.manual_seed(0)

    # Total sequence length for shared version
    S = P + C + R

    # Initialize input tensors, i.e. the tensors after q, k, and v projections of hidden states (attention head input)
    query = torch.randn(B, H, S, D, device=device, dtype=dtype)
    key = torch.randn(B, H, S, D, device=device, dtype=dtype)
    value = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # Split tensors for separate computation
    query_prefix = query[:, :, :P, :]
    key_prefix = key[:, :, :P, :]
    value_prefix = value[:, :, :P, :]

    query_chosen = query[:, :, P : P + C, :]
    key_chosen = key[:, :, P : P + C, :]
    value_chosen = value[:, :, P : P + C, :]

    query_rejected = query[:, :, P + C :, :]
    key_rejected = key[:, :, P + C :, :]
    value_rejected = value[:, :, P + C :, :]

    chosen_index = torch.full((B,), P + C, device=device)
    rejected_index = torch.full((B,), S, device=device)

    def wrapped_prefix_mask(b, h, q_idx, kv_idx):
        return prefix_mask(b, h, q_idx, kv_idx, rejected_index, chosen_index)

    block_mask = create_block_mask(wrapped_prefix_mask, B, H, S, S, device=device)
    shared_out = flex_attention(query, key, value, block_mask=block_mask)

    # Compute attention for prefix + chosen separately
    PC = P + C
    query_pc = torch.cat([query_prefix, query_chosen], dim=2)
    key_pc = torch.cat([key_prefix, key_chosen], dim=2)
    value_pc = torch.cat([value_prefix, value_chosen], dim=2)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    pc_block_mask = create_block_mask(causal_mask, B, H, PC, PC, device=device)
    pc_out = flex_attention(query_pc, key_pc, value_pc, block_mask=pc_block_mask)

    # Compute attention for prefix + rejected separately
    PR = P + R
    query_pr = torch.cat([query_prefix, query_rejected], dim=2)
    key_pr = torch.cat([key_prefix, key_rejected], dim=2)
    value_pr = torch.cat([value_prefix, value_rejected], dim=2)

    pr_block_mask = create_block_mask(causal_mask, B, H, PR, PR, device=device)
    pr_out = flex_attention(query_pr, key_pr, value_pr, block_mask=pr_block_mask)

    shared_prefix = shared_out[:, :, :P, :P]
    shared_chosen = shared_out[:, :, P : P + C, P : P + C]
    shared_rejected = shared_out[:, :, P + C :, P + C :]

    separate_prefix_c = pc_out[:, :, :P, :P]
    separate_chosen = pc_out[:, :, P:, P:]
    separate_prefix_r = pr_out[:, :, :P, :P]
    separate_rejected = pr_out[:, :, P:, P:]

    # Verify prefix outputs are identical
    assert torch.allclose(shared_prefix, separate_prefix_c, atol=atol, rtol=rtol), (
        "Prefix attention from shared computation doesn't match prefix+chosen computation"
    )
    assert torch.allclose(shared_prefix, separate_prefix_r, atol=atol, rtol=rtol), (
        "Prefix attention from shared computation doesn't match prefix+rejected computation"
    )

    # Verify chosen and rejected outputs
    assert torch.allclose(shared_chosen, separate_chosen, atol=atol, rtol=rtol), (
        "Chosen response attention doesn't match between shared and separate computation"
    )
    assert torch.allclose(shared_rejected, separate_rejected, atol=atol, rtol=rtol), (
        "Rejected response attention doesn't match between shared and separate computation"
    )

    print("All attention values match between shared and separate computations!")


@pytest.mark.skipif(not _is_flex_attention_supported(), reason="FlexAttention is only supported on CUDA or CPU devices")
@pytest.mark.parametrize(
    "B, H, P, C, R, D",
    [
        (2, 8, 512, 256, 256, 32),
        (3, 12, 1024, 512, 512, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            3e-2,
            5e-1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e-2, 5e-3),
        (torch.float32, 1e-3, 5e-4),
    ],
)
def test_correctness_prefix(B, H, P, C, R, D, dtype, atol, rtol):
    """Parametrized test for different configurations"""
    _test_correctness_prefix(B=B, H=H, P=P, C=C, R=R, D=D, dtype=dtype, atol=atol, rtol=rtol)
