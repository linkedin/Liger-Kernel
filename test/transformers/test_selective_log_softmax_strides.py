import pytest
import torch

from liger_kernel.ops.grpo_loss import fused_selective_log_softmax


def _layout_inputs(ids, mask, layout):
    batch, length = ids.shape
    if layout == "dense":
        return ids, mask
    if layout == "different_prefixes":
        # Both backing allocations are large enough for the old indexing too:
        # this reproduces a wrong result, not an illegal memory access.
        id_storage = ids.new_zeros(batch, length + 3)
        mask_storage = mask.new_zeros(batch, length + 9)
        id_storage[:, -length:] = ids
        mask_storage[:, -length:] = mask
        return id_storage, mask_storage
    if layout == "strided_ids":
        storage = ids.new_zeros(batch, 2 * length)
        storage[:, ::2] = ids
        # Use a larger independent mask allocation to keep baseline reads safe.
        mask_storage = mask.new_zeros(batch, 2 * length + 3)
        mask_storage[:, -length:] = mask
        return storage[:, ::2], mask_storage
    if layout == "strided_mask":
        storage = mask.new_zeros(batch, 2 * length)
        storage[:, ::2] = mask
        return ids, storage[:, ::2]
    if layout == "broadcast_ids":
        return ids[:1].expand(batch, -1), mask
    if layout == "broadcast_mask":
        # The logical view broadcasts one row. Extra backing rows keep incorrect
        # non-broadcast reads by the baseline inside the allocation.
        storage = mask.new_zeros(batch, length)
        storage[0] = mask[0]
        return ids, storage[:1].expand(batch, -1)
    raise AssertionError(f"Unknown test layout: {layout}")


@pytest.mark.parametrize(
    "layout", ["dense", "different_prefixes", "strided_ids", "strided_mask", "broadcast_ids", "broadcast_mask"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [17, 2053])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_selective_log_softmax_respects_independent_strides(layout, dtype, vocab_size, temperature):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the Triton kernel")
    batch, length = 3, 7
    generator = torch.Generator(device="cuda").manual_seed(83)
    logits = torch.randn(batch, length + 1, vocab_size, generator=generator, dtype=dtype, device="cuda")
    ids = torch.randint(vocab_size, (batch, length), generator=generator, device="cuda")
    mask = torch.tensor(
        [[1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1]],
        dtype=torch.bool,
        device="cuda",
    )
    input_ids, attention_mask = _layout_inputs(ids, mask, layout)
    before_ids, before_mask = input_ids.clone(), attention_mask.clone()
    actual = fused_selective_log_softmax(logits, input_ids, temperature, attention_mask)
    selected = input_ids[:, -length:]
    active = attention_mask[:, -length:]
    expected = (logits[:, :-1].float() / temperature).log_softmax(-1).gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    expected = expected.masked_fill(~active, 0)

    assert actual.dtype == torch.float32
    assert actual.shape == (batch, length)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(actual[~active]) == 0
    assert torch.equal(input_ids, before_ids)
    assert torch.equal(attention_mask, before_mask)


@pytest.mark.parametrize("layout", ["dense", "strided_ids", "broadcast_ids"])
def test_selective_log_softmax_without_mask_preserves_id_layout(layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the Triton kernel")
    generator = torch.Generator(device="cuda").manual_seed(89)
    logits = torch.randn(3, 8, 17, generator=generator, device="cuda")
    ids = torch.randint(17, (3, 7), generator=generator, device="cuda")
    ids, _ = _layout_inputs(ids, torch.ones_like(ids, dtype=torch.bool), layout)
    actual = fused_selective_log_softmax(logits, ids, temperature=1.0)
    expected = logits[:, :-1].log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
