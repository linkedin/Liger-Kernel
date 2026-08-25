import importlib

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.ops import LigerCCEFunction
from liger_kernel.ops import liger_cce as ops_liger_cce
from liger_kernel.transformers import LigerCCELoss
from liger_kernel.transformers.functional import liger_cce
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()
cce_module = importlib.import_module("liger_kernel.ops.cce")


def _make_cce_inputs(
    shape,
    dtype=torch.float32,
    tensor_device=device,
    bias=False,
    requires_grad=False,
    target_dtype=torch.long,
):
    n_tokens, hidden_size, vocab_size = shape
    hidden = torch.randn(n_tokens, hidden_size, device=tensor_device, dtype=dtype, requires_grad=requires_grad)
    weight = (torch.randn(vocab_size, hidden_size, device=tensor_device, dtype=dtype) * 0.1).requires_grad_(
        requires_grad
    )
    bias_value = (
        (torch.randn(vocab_size, device=tensor_device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
        if bias
        else None
    )
    targets = torch.randint(vocab_size, (n_tokens,), device=tensor_device, dtype=target_dtype)
    return hidden, weight, targets, bias_value


def _clone_with_grad(*tensors):
    return tuple(tensor.detach().clone().requires_grad_(True) if tensor is not None else None for tensor in tensors)


def _torch_cce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    logit_scale: float = 1.0,
    softcap: Optional[float] = None,
    reduction: str = "mean",
):
    logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
    logits = logits * logit_scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, targets.long(), ignore_index=ignore_index, reduction=reduction)


@pytest.mark.parametrize("shape", [(17, 33, 71), (257, 64, 503)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("bias", [False, True])
def test_cce_forward_backward(shape, dtype, reduction, bias):
    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (7e-2, 7e-2)

    hidden, weight, targets, bias_value = _make_cce_inputs(shape, dtype=dtype, bias=bias)
    targets[::7] = -100

    torch_hidden, torch_weight, torch_bias = _clone_with_grad(hidden, weight, bias_value)
    liger_hidden, liger_weight, liger_bias = _clone_with_grad(hidden, weight, bias_value)

    torch_loss = _torch_cce(
        torch_hidden,
        torch_weight,
        targets,
        torch_bias,
        logit_scale=0.7,
        softcap=4.0,
        reduction=reduction,
    )
    liger_loss = liger_cce(
        liger_hidden,
        liger_weight,
        targets,
        liger_bias,
        logit_scale=0.7,
        softcap=4.0,
        reduction=reduction,
    )
    assert_verbose_allclose(torch_loss, liger_loss, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(torch_loss) * 0.37
    torch_loss.backward(grad_output)
    liger_loss.backward(grad_output)
    assert_verbose_allclose(torch_hidden.grad, liger_hidden.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(torch_weight.grad, liger_weight.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(torch_bias.grad, liger_bias.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cce_metrics(dtype):
    hidden, weight, targets, _ = _make_cce_inputs((39, 47, 101), dtype=dtype, requires_grad=True)
    targets[::5] = -100

    loss, metrics = liger_cce(hidden, weight, targets, return_metrics=True)
    logits = F.linear(hidden.float(), weight.float())
    valid = targets != -100
    expected_correct = ((logits.argmax(-1) == targets) & valid).sum()
    probabilities = logits.softmax(-1)
    entropy = -(probabilities * logits.log_softmax(-1)).sum(-1)
    expected_entropy_sum = entropy[valid].sum()

    assert torch.equal(metrics["num_correct_tokens"], expected_correct)
    assert_verbose_allclose(metrics["entropy_sum"], expected_entropy_sum, atol=5e-2, rtol=5e-2)
    assert not metrics["num_correct_tokens"].requires_grad
    assert not metrics["entropy_sum"].requires_grad

    loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cce_all_ignored_has_connected_zero_gradients(reduction):
    hidden, weight, targets, bias = _make_cce_inputs((13, 29, 61), dtype=torch.bfloat16, bias=True, requires_grad=True)
    targets.fill_(-100)

    loss = liger_cce(hidden, weight, targets, bias, reduction=reduction)
    assert torch.count_nonzero(loss) == 0
    loss.backward(torch.randn_like(loss))

    for gradient in (hidden.grad, weight.grad, bias.grad):
        assert gradient is not None
        assert torch.count_nonzero(gradient) == 0


def test_cce_int32_targets_and_noncontiguous_inputs():
    hidden, weight, targets, _ = _make_cce_inputs((23, 38, 89), target_dtype=torch.int32)

    torch_hidden, torch_weight = _clone_with_grad(hidden, weight)
    liger_hidden, liger_weight = _clone_with_grad(hidden, weight)
    expected = _torch_cce(torch_hidden[:, ::2], torch_weight[:, ::2], targets)
    actual = liger_cce(liger_hidden[:, ::2], liger_weight[:, ::2], targets)
    assert_verbose_allclose(expected, actual, atol=2e-5, rtol=2e-4)

    expected.backward()
    actual.backward()
    assert_verbose_allclose(torch_hidden.grad, liger_hidden.grad, atol=2e-5, rtol=2e-4)
    assert_verbose_allclose(torch_weight.grad, liger_weight.grad, atol=2e-5, rtol=2e-4)


def test_cce_entrypoints():
    hidden, weight, targets, _ = _make_cce_inputs((11, 17, 43))

    expected = liger_cce(hidden, weight, targets)
    assert_verbose_allclose(ops_liger_cce(hidden, weight, targets), expected)
    assert_verbose_allclose(LigerCCELoss()(weight, hidden, targets), expected)
    function_loss, entropy, correct = LigerCCEFunction.apply(
        hidden, weight, targets, None, -100, 1.0, None, "mean", False
    )
    assert_verbose_allclose(function_loss, expected)
    assert entropy is None
    assert correct is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_cce_noncurrent_device():
    torch.cuda.set_device(0)
    noncurrent_device = torch.device("cuda:1")
    hidden, weight, targets, _ = _make_cce_inputs(
        (19, 31, 67), dtype=torch.bfloat16, tensor_device=noncurrent_device, requires_grad=True
    )

    loss = liger_cce(hidden, weight, targets)
    loss.backward()

    assert loss.device == noncurrent_device
    assert hidden.grad is not None
    assert weight.grad is not None


def test_cce_uses_portable_defaults_for_non_nvidia_backends():
    xpu_device = torch.device("xpu")
    config = cce_module._cce_config(257, 64, 503, torch.bfloat16, xpu_device)
    splits, split_size = cce_module._cce_num_splits(3, 503, 64, torch.device("meta"))

    assert config == {"BLOCK_N": 64, "BLOCK_V": 64, "BLOCK_H": 32, "num_warps": 4, "num_stages": 1}
    assert splits == 1
    assert split_size == 512


@pytest.mark.parametrize(
    "mutation, error_type, match",
    [
        (lambda h, w, t, b: (h.unsqueeze(0), w, t, b), ValueError, "hidden"),
        (lambda h, w, t, b: (h, w[:, :-1], t, b), ValueError, "weight"),
        (lambda h, w, t, b: (h, w, t[:-1], b), ValueError, "targets"),
        (lambda h, w, t, b: (h, w.double(), t, b), TypeError, "weight dtype"),
        (lambda h, w, t, b: (h, w, t.float(), b), TypeError, "targets"),
        (lambda h, w, t, b: (h, w, t, b[:-1]), ValueError, "bias"),
    ],
)
def test_cce_input_validation(mutation, error_type, match):
    hidden, weight, targets, bias = _make_cce_inputs((7, 13, 31), bias=True)
    args = mutation(hidden, weight, targets, bias)
    with pytest.raises(error_type, match=match):
        liger_cce(*args)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"reduction": "batchmean"}, "reduction"),
        ({"softcap": 0.0}, "softcap"),
        ({"logit_scale": "one"}, "logit_scale"),
    ],
)
def test_cce_option_validation(kwargs, match):
    hidden, weight, targets, _ = _make_cce_inputs((7, 13, 31))
    with pytest.raises((TypeError, ValueError), match=match):
        liger_cce(hidden, weight, targets, **kwargs)
