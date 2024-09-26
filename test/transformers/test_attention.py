import pytest
import torch
from torch import Tensor
from typing import Optional, Tuple

from liger_kernel.ops.flash_attention import flash_attn_func, flash_attn_reference
from test.utils import set_seed

set_seed()
DEVICE = "cuda"


def compare_numerical_errors(
    x_ref: Tensor,
    x_pt: Tensor,
    x_triton: Tensor,
    error_mul: float,
    error_atol: float,
    tensor_name: str,
) -> None:
    max_pt_error = (x_pt - x_ref).abs().max().item()
    max_triton_error = (x_triton - x_ref).abs().max().item()
    assert max_triton_error <= max(error_mul * max_pt_error, error_atol), tensor_name


def _test_attention(
    batch_size: int,
    nheads_q: int,
    nheads_kv: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    causal: bool,
    dropout_p: float,
    use_attention: bool,
    use_bias: bool,
    dtype: torch.dtype,
) -> Optional[Tuple[Tensor, ...]]:

    # Prepare data
    q = torch.normal(0, 0.5, (batch_size, seqlen_q, nheads_q, head_dim), dtype=dtype, device=DEVICE).requires_grad_()
    k = torch.normal(0, 0.5, (batch_size, seqlen_k, nheads_kv, head_dim), dtype=dtype, device=DEVICE).requires_grad_()
    v = torch.normal(0, 0.5, (batch_size, seqlen_k, nheads_kv, head_dim), dtype=dtype, device=DEVICE).requires_grad_()
    do = torch.randn_like(q)
    attn_bias = torch.rand(size=(1, 1, seqlen_q, seqlen_k), dtype=dtype, device=q.device) if use_bias else None

    # Compute the outputs of the forward pass
    ref_output = flash_attn_reference(q, k, v, attn_bias=attn_bias, causal=causal, upcast=True, reorder_ops=False)
    pt_output = flash_attn_reference(q, k, v, attn_bias=attn_bias, causal=causal, upcast=False, reorder_ops=True)
    liger_output = flash_attn_func(q, k, v, attention_bias=attn_bias, causal=causal)
    compare_numerical_errors(ref_output, pt_output, liger_output, 1, 1e-4, "output")

    # Compare the gradients after the backward pass
    ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_output, (q, k, v), do, retain_graph=True)
    pt_dq, pt_dk, pt_dv = torch.autograd.grad(pt_output, (q, k, v), do, retain_graph=True)
    liger_dq, liger_dk, liger_dv = torch.autograd.grad(liger_output, (q, k, v), do, retain_graph=True)
    compare_numerical_errors(ref_dq, pt_dq, liger_dq, 2, 1e-4, "dq")
    compare_numerical_errors(ref_dk, pt_dk, liger_dk, 2, 1e-4, "dk")
    compare_numerical_errors(ref_dv, pt_dv, liger_dv, 2, 1e-4, "dv")


@pytest.mark.parametrize("dtype", [(torch.float16), (torch.bfloat16)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "head_dim, nheads_q, nheads_kv",
    [(32, 9, 9), (40, 9, 3), (64, 8, 8), (128, 8, 2), (256, 4, 2)],
)
@pytest.mark.parametrize("swap_seqlens", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (113, 203),
        (127, 512),
        (128, 217),
        (256, 512),
    ],
)
@pytest.mark.parametrize("batch_size", [4])
def test_fwd_bwd(
    batch_size: int,
    nheads_q: int,
    nheads_kv: int,
    seqlen_q: int,
    seqlen_k: int,
    swap_seqlens: bool,
    head_dim: int,
    causal: bool,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    if swap_seqlens:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    _test_attention(
        batch_size=batch_size,
        nheads_q=nheads_q,
        nheads_kv=nheads_kv,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        causal=causal,
        dropout_p=0,
        use_attention=False,
        use_bias=use_bias,
        dtype=dtype,
    )
