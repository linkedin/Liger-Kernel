from test.utils import assert_verbose_allclose, set_seed

import pytest
import torch

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)
from liger_kernel.utils import infer_device

device = infer_device()

# set random seed globally
set_seed()


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),  # typical shape
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-1),
        (1.0, torch.float32, 2e-2, 5e-1),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ref_bias", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
def test_ref_input(
    B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias, ignore_index, beta
):
    """Test that using ref_input gives different results than using input_chunk for reference model."""
    B = 2 * B  # requires B to be even

    # Create input tensors
    input_chunk = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    ref_input = (
        torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    )  # Different input for reference model
    target_chunk = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

    # Assign some random elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target_chunk.view(-1)[indices_to_assign] = ignore_index

    # Create weights and biases
    weight = torch.randn(V, H, device=device, dtype=dtype)
    ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    _bias = torch.randn(V, device=device, dtype=dtype) if bias else None
    _ref_bias = torch.randn(V, device=device, dtype=dtype) if ref_bias else None

    # Mock loss function that returns the difference between policy and reference logits
    def mock_loss_fn(
        chosen_logps,
        rejected_logps,
        full_target,
        beta=0.1,
        ref_chosen_logps=None,
        ref_rejected_logps=None,
    ):
        # Return the mean difference between policy and reference logits
        diff = (chosen_logps - ref_chosen_logps).mean() + (
            rejected_logps - ref_rejected_logps
        ).mean()
        return diff, (diff,)  # Return an aux output to test that too

    # Forward pass without ref_input (using input_chunk for reference model)
    outputs1 = LigerFusedLinearPreferenceBase._compute_loss(
        input_chunk=input_chunk,
        weight=weight,
        target_chunk=target_chunk,
        bias=_bias,
        preference_loss_fn=mock_loss_fn,
        full_target=target_chunk,
        ignore_index=ignore_index,
        alpha=1.0,
        beta=beta,
        compute_nll_loss=True,
        use_ref_model=True,
        ref_weight=ref_weight,
        ref_bias=_ref_bias,
    )

    # Forward pass with ref_input
    outputs2 = LigerFusedLinearPreferenceBase._compute_loss(
        input_chunk=input_chunk,
        weight=weight,
        target_chunk=target_chunk,
        bias=_bias,
        preference_loss_fn=mock_loss_fn,
        full_target=target_chunk,
        ignore_index=ignore_index,
        alpha=1.0,
        beta=beta,
        compute_nll_loss=True,
        use_ref_model=True,
        ref_weight=ref_weight,
        ref_bias=_ref_bias,
        ref_input=ref_input,  # Use different input for reference model
    )

    # The outputs should be different since we used different inputs
    loss1, (chosen_logps1, rejected_logps1, _, _, _, aux1) = outputs1
    loss2, (chosen_logps2, rejected_logps2, _, _, _, aux2) = outputs2

    # The chosen/rejected logps from the policy model should be identical
    assert_verbose_allclose(chosen_logps1, chosen_logps2, atol=atol, rtol=rtol)
    assert_verbose_allclose(rejected_logps1, rejected_logps2, atol=atol, rtol=rtol)

    # But the losses and aux outputs should be different since ref_input is different
    with pytest.raises(AssertionError):
        assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        assert_verbose_allclose(aux1, aux2, atol=atol, rtol=rtol)
