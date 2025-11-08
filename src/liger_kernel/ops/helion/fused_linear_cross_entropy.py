import argparse

import helion
import helion.language as hl
import torch

from helion._testing import run_example

from liger_kernel.utils import infer_device

# Best config for default llama3Config(hidden_size=4096, vocab_size=32000) with 4096 bs*seqlen input
h100_fwd_config = helion.Config(
    block_sizes=[64, 32, 512],
    indexing=["pointer", "pointer", "tensor_descriptor", "pointer", "tensor_descriptor", "tensor_descriptor"],
    load_eviction_policies=["", "last", "last", "last"],
    num_stages=8,
    num_warps=16,
    pid_type="flat",
    range_flattens=[None, False, None],
    range_multi_buffers=[None, True, False],
    range_num_stages=[0, 3, 3],
    range_unroll_factors=[0, 0, 1],
    range_warp_specializes=[],
)


# @helion.kernel(config=h100_fwd_config, static_shapes=True)
@helion.kernel(autotune_effort="none", ignore_warnings=[helion.exc.TensorOperationInWrapper])
def fused_linear_cross_entropy_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Performs matrix multiplication followed by cross entropy loss.
    Args:
        x: input tensor of shape [BT, H]
        weight: weight tensor of shape [V, H]
        target: target tensor of shape [BT,]
        ignore_index: index to ignore in the target
        reduction: reduction to apply to the loss
    Returns:
        loss: loss tensor of shape [1] if reduction is "mean" or "sum", [BT] otherwise
    """
    BT, H = x.size()
    V = weight.size(0)
    block_size_bt = hl.register_block_size(BT)
    block_size_h = hl.register_block_size(H)
    block_size_v = hl.register_block_size(V)

    nll = torch.zeros(BT, device=x.device, dtype=torch.float32)
    lse = torch.zeros(BT, device=x.device, dtype=torch.float32)

    # May be useful for splitting fwd and bwd
    # lse = torch.full((BT,), fill_value=-torch.inf, device=x.device, dtype=torch.float32)
    # neg_target_logits = torch.zeros(BT, device=x.device, dtype=torch.float32)

    n_non_ignore = (target != ignore_index).sum().unsqueeze(0)

    # forward
    for tile_bt in hl.tile(BT, block_size=block_size_bt):
        m_i = hl.zeros([tile_bt], dtype=torch.float32) - float("inf")
        d_i = hl.zeros([tile_bt], dtype=torch.float32)
        nll_tile = hl.zeros([tile_bt], dtype=torch.float32)
        if reduction == "mean":
            n_non_ignore_value = hl.load(n_non_ignore, [0])

        # target_indices = target[tile_bt][:, None]  # ERROR: it introduces a new size, which is not broadcastable
        target_indices = target[tile_bt].unsqueeze(1)  # [tile_bt, 1]
        for tile_v in hl.tile(V, block_size=block_size_v):
            # logits computation
            acc = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
            for tile_h in hl.tile(H, block_size=block_size_h):
                x_tile = x[tile_bt, tile_h]
                weight_tile = weight[tile_v, tile_h]
                acc = hl.dot(x_tile, weight_tile.T, acc=acc, out_dtype=torch.float32)

            # online softmax statistics
            m_ij = torch.maximum(m_i, torch.amax(acc, dim=-1))
            d_i = d_i * torch.exp(m_i - m_ij) + torch.exp(acc - m_ij[:, None]).sum(dim=-1)
            m_i = m_ij

            # offset = tile_v.index[None, :]  # ERROR: it introduces a new size, which is not broadcastable
            offset = tile_v.index.unsqueeze(0)  # [1, tile_v]
            mask = target_indices == offset  # [tile_bt, tile_v]
            nll_tile += torch.sum(-acc * mask, dim=-1)  # [tile_bt]

        # loss computation: -logsoftmax(x_y) = -log(exp(x_y) / sum(exp(x_i))) = -x_y + log(sum(exp(x_i)))
        lse_tile = m_i + torch.log(d_i)
        nll_tile = nll_tile + lse_tile

        # handle ignore index
        nll_tile = nll_tile * (target_indices.ravel() != ignore_index)

        if reduction == "mean":
            nll_tile /= n_non_ignore_value

        nll[tile_bt] = nll_tile
        lse[tile_bt] = lse_tile

    if reduction != "none":
        loss = nll.sum()
    else:
        loss = nll

    return loss.to(x.dtype), lse


h100_bwd_config = helion.Config(
    block_sizes=[128, 64, 128],
    indexing=[
        "pointer",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
    ],
    l2_groupings=[64],
    load_eviction_policies=["", "first", "last", "", "", "last", "first", "first", ""],
    loop_orders=[[0, 1]],
    num_stages=7,
    num_warps=8,
    pid_type="flat",
    range_flattens=[None, True],
    range_multi_buffers=[None, None],
    range_num_stages=[0, 3],
    range_unroll_factors=[0, 1],
    range_warp_specializes=[],
)


# @helion.kernel(config=h100_bwd_config, static_shapes=True)
@helion.kernel(autotune_effort="none", ignore_warnings=[helion.exc.TensorOperationInWrapper])
def fused_linear_cross_entropy_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    BT, H = x.size()
    V = weight.size(0)
    block_size_bt = hl.register_block_size(BT)
    block_size_h = hl.register_block_size(H)
    block_size_v = hl.register_block_size(V)
    grad_x = torch.zeros_like(x, dtype=torch.float32)
    grad_w = torch.zeros_like(weight, dtype=torch.float32)
    n_non_ignore = (target != ignore_index).sum().unsqueeze(0)

    num_block_bt = (BT + block_size_bt - 1) // block_size_bt
    num_block_h = (H + block_size_h - 1) // block_size_h
    num_block_v = (V + block_size_v - 1) // block_size_v
    grad_x_lock = torch.zeros((num_block_bt, num_block_h), dtype=torch.int32, device=x.device)
    grad_w_lock = torch.zeros((num_block_v, num_block_h), dtype=torch.int32, device=x.device)
    # backward
    for tile_bt, tile_v in hl.tile([BT, V], block_size=(block_size_bt, block_size_v)):
        if reduction == "mean":
            n_non_ignore_value = hl.load(n_non_ignore, [0], eviction_policy="evict_last")
        # Restore logits
        acc2 = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
        for tile_h in hl.tile(H, block_size=block_size_h):
            x_tile = x[tile_bt, tile_h]
            weight_tile = weight[tile_v, tile_h]
            acc2 = hl.dot(x_tile, weight_tile.T, acc=acc2, out_dtype=torch.float32)

        # softmax(x_i) = exp(x_i) / sum(exp(x_i))
        #              = exp(x_i) / log(exp(sum(x_i)))
        #              = exp(x_i) / lse = exp(x_i - lse)
        lse_tile = lse[tile_bt]
        target_indices = target[tile_bt].unsqueeze(1)  # [tile_bt, 1]

        grad_logits_tile = torch.exp(acc2 - lse_tile[:, None])
        offset = tile_v.index.unsqueeze(0)  # [1, tile_v]
        mask = target_indices == offset  # [tile_bt, tile_v]
        grad_logits_tile = grad_logits_tile - mask.float()
        # handle out of bound values in grad_logits_tile
        grad_logits_tile = grad_logits_tile * ((tile_bt.index < BT)[:, None] & (tile_v.index < V)[None, :])

        # handle ignore index
        grad_logits_tile = grad_logits_tile * (target_indices != ignore_index)

        if reduction == "mean":
            grad_logits_tile /= n_non_ignore_value

        for tile_h in hl.tile(H, block_size=block_size_h):
            # grad_x = grad_logits @ weight
            rhs_tile = weight[tile_v, tile_h]
            partial_grad_x = hl.dot(grad_logits_tile, rhs_tile, out_dtype=torch.float32)
            while hl.atomic_cas(grad_x_lock, [tile_bt.id, tile_h.id], 0, 1, sem="acquire") == 1:
                pass
            grad_x[tile_bt, tile_h] += partial_grad_x
            hl.atomic_xchg(grad_x_lock, [tile_bt.id, tile_h.id], 0, sem="release")
            # hl.atomic_add(grad_x, [tile_bt, tile_h], partial_grad_x)

            # for tile_h in hl.tile(H, block_size=block_size_h):
            # grad_w = grad_logits.T[tile_v, tile_bt] @ x[tile_bt, tile_h]
            rhs_tile = x[tile_bt, tile_h]
            partial_grad_w = hl.dot(grad_logits_tile.T, rhs_tile, out_dtype=torch.float32)
            while hl.atomic_cas(grad_w_lock, [tile_v.id, tile_h.id], 0, 1, sem="acquire") == 1:
                pass
            grad_w[tile_v, tile_h] += partial_grad_w
            hl.atomic_xchg(grad_w_lock, [tile_v.id, tile_h.id], 0, sem="release")
            # hl.atomic_add(grad_w, [tile_v, tile_h], partial_grad_w)

    return grad_x.to(x.dtype), grad_w.to(x.dtype)


@helion.kernel(autotune_effort="none", ignore_warnings=[helion.exc.TensorOperationInWrapper])
def _grad_logit_compute(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    n_non_ignore: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    BT, H = x.size()
    V = weight.size(0)

    block_size_bt = hl.register_block_size(BT)
    block_size_h = hl.register_block_size(H)
    block_size_v = hl.register_block_size(V)
    grad_logits = torch.zeros((BT, V), dtype=torch.float32, device=x.device)
    for tile_bt, tile_v in hl.tile([BT, V], block_size=(block_size_bt, block_size_v)):
        if reduction == "mean":
            n_non_ignore_value = hl.load(n_non_ignore, [0], eviction_policy="evict_last")
        # Restore logits
        acc2 = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
        for tile_h in hl.tile(H, block_size=block_size_h):
            x_tile = x[tile_bt, tile_h]
            weight_tile = weight[tile_v, tile_h]
            acc2 = hl.dot(x_tile, weight_tile.T, acc=acc2, out_dtype=torch.float32)

        # softmax(x_i) = exp(x_i) / sum(exp(x_i))
        #              = exp(x_i) / log(exp(sum(x_i)))
        #              = exp(x_i) / lse = exp(x_i - lse)
        lse_tile = lse[tile_bt]
        target_indices = target[tile_bt].unsqueeze(1)  # [tile_bt, 1]

        grad_logits_tile = torch.exp(acc2 - lse_tile[:, None])
        offset = tile_v.index.unsqueeze(0)  # [1, tile_v]
        mask = target_indices == offset  # [tile_bt, tile_v]
        grad_logits_tile = grad_logits_tile - mask.float()
        # handle out of bound values in grad_logits_tile
        grad_logits_tile = grad_logits_tile * ((tile_bt.index < BT)[:, None] & (tile_v.index < V)[None, :])

        # handle ignore index
        grad_logits_tile = grad_logits_tile * (target_indices != ignore_index)

        if reduction == "mean":
            grad_logits_tile /= n_non_ignore_value

        grad_logits[tile_bt, tile_v] = grad_logits_tile
    return grad_logits


def fused_linear_cross_entropy_bwd_chunk(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    BT, H = x.size()
    V = weight.size(0)
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    num_chunks = (V + H - 1) // H
    chunk_size = (BT + num_chunks - 1) // num_chunks
    grad_x = torch.zeros_like(x, dtype=torch.float32)
    grad_w = torch.zeros_like(weight, dtype=torch.float32)
    n_non_ignore = (target != ignore_index).sum().unsqueeze(0)

    x_chunks = torch.chunk(x, chunks=num_chunks, dim=0)
    lse_chunks = torch.chunk(lse, chunks=num_chunks, dim=0)
    target_chunks = torch.chunk(target, chunks=num_chunks, dim=0)

    for chunk_id, (x_chunk, target_chunk, lse_chunk) in enumerate(zip(x_chunks, target_chunks, lse_chunks)):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        grad_logits_chunk = _grad_logit_compute(
            x_chunk,
            weight,
            target_chunk,
            lse_chunk,
            n_non_ignore,
            reduction,
        )

        grad_x[start_idx:end_idx] = grad_logits_chunk @ weight
        grad_w += torch.mm(grad_logits_chunk.T, x_chunk).float()

    return grad_x.to(x.dtype), grad_w.to(x.dtype)


class LigerFusedLinearCrossEntropyHelionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, ignore_index=-100, reduction="mean", bwd_impl="chunk"):
        assert bwd_impl in ["chunk", "cce"]
        loss, lse = fused_linear_cross_entropy_fwd(
            _input,
            weight,
            target,
            ignore_index,
            reduction,
        )
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.bwd_impl = bwd_impl
        ctx.save_for_backward(_input, lse, weight, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.ndim == 0, "token_scaling is not supported. grad_output must be a scalar"
        _input, lse, weight, target = ctx.saved_tensors
        if ctx.bwd_impl == "cce":
            bwd_fn = fused_linear_cross_entropy_bwd
        elif ctx.bwd_impl == "chunk":
            bwd_fn = fused_linear_cross_entropy_bwd_chunk
        grad_input, grad_weight = bwd_fn(
            _input,
            weight,
            target,
            lse,
            ctx.ignore_index,
            ctx.reduction,
        )
        return grad_input * grad_output, grad_weight * grad_output, None, None, None, None


class LigerFusedLinearCrossEntropyHelion(torch.nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean", bwd_impl="chunk"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.bwd_impl = bwd_impl

    def forward(self, _input, weight, target):
        return LigerFusedLinearCrossEntropyHelionFunction.apply(
            _input, weight, target, self.ignore_index, self.reduction, self.bwd_impl
        )


class TorchLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.logits = None

    def forward(self, x, target):
        self.logits = self.lm_head(x).to(torch.float32)
        return self.ce_loss(self.logits, target)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
        bwd_impl: str = "cce",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.flce = LigerFusedLinearCrossEntropyHelion(
            ignore_index=ignore_index, reduction=reduction, bwd_impl=bwd_impl
        )

    def forward(self, x, target):
        return self.flce(x, self.lm_head.weight, target)


from functools import partial

from cut_cross_entropy import linear_cross_entropy


class CutLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.flce = partial(linear_cross_entropy, ignore_index=ignore_index, reduction=reduction, return_lse=False)

    def forward(self, x, target):
        return self.flce(x, self.lm_head.weight, target)


from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


class TritonLigerLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.flce = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, target):
        return self.flce(self.lm_head.weight, x, target, None)


def generate_flce_fwd_input(BT, V, H, dtype, device):
    x = torch.randn(BT, H, device=device, dtype=dtype)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device)
    return (x, weight, target)


def generate_flce_bwd_input(BT, V, H, dtype, device):
    x = torch.randn(BT, H, device=device, dtype=dtype)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device)
    lse = torch.randn(BT, device=device, dtype=torch.float32)
    return (x, weight, target, lse)


def generate_grad_logits_compute_input(BT, V, H, dtype, device):
    x = torch.randn(BT, H, device=device, dtype=dtype)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device)
    lse = torch.randn(BT, device=device, dtype=torch.float32)
    n_non_ignore = (target != -100).sum().unsqueeze(0)
    return (x, weight, target, lse, n_non_ignore)


def autotune_kernels(model_config_dataset):
    device = infer_device()
    torch_device = getattr(torch, device)
    gpu_name = torch_device.get_device_name(torch_device.current_device())

    for model_name, model_config in model_config_dataset.items():
        for dtype in [torch.bfloat16, torch.float32]:
            BT = 4096
            args = generate_flce_fwd_input(
                BT,
                model_config["hidden_size"],
                model_config["vocab_size"],
                dtype=dtype,
                device=device,
            )
            config = fused_linear_cross_entropy_fwd.autotune(args)
            if dtype == torch.bfloat16:
                dtype_str = "bf16"
            elif dtype == torch.float32:
                dtype_str = "fp32"
            config.save(f"configs/fused_linear_cross_entropy_fwd_{gpu_name}_{model_name}_{dtype_str}.json")

    for model_name, model_config in model_config_dataset.items():
        for dtype in [torch.bfloat16, torch.float32]:
            BT = 4096
            args = generate_flce_bwd_input(
                BT,
                model_config["hidden_size"],
                model_config["vocab_size"],
                dtype=dtype,
                device=device,
            )
            config = fused_linear_cross_entropy_bwd.autotune(args)
            if dtype == torch.bfloat16:
                dtype_str = "bf16"
            elif dtype == torch.float32:
                dtype_str = "fp32"
            config.save(f"configs/fused_linear_cross_entropy_bwd_{gpu_name}_{model_name}_{dtype_str}.json")

    for model_name, model_config in model_config_dataset.items():
        for dtype in [torch.bfloat16, torch.float32]:
            BT = 4096
            args = generate_grad_logits_compute_input(
                BT,
                model_config["hidden_size"],
                model_config["vocab_size"],
                dtype=dtype,
                device=device,
            )
            config = _grad_logit_compute.autotune(args)
            if dtype == torch.bfloat16:
                dtype_str = "bf16"
            elif dtype == torch.float32:
                dtype_str = "fp32"
            config.save(f"configs/_grad_logit_compute_{gpu_name}_{model_name}_{dtype_str}.json")


def check():
    device = infer_device()

    batch_size = 2
    seq_len = 4096
    hidden_size = 4096
    vocab_size = 32000


    print(f"BT={batch_size * seq_len}, H={hidden_size}, V={vocab_size}")

    dtype = torch.float32
    reduction = "mean"
    ignore_index = -100
    rtol = 1e-2
    atol = 1e-1

    input = torch.randn(batch_size * seq_len, hidden_size, device=device, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device=device, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)
    # Init
    ref_lm_head_ce = TorchLMHeadCE(hidden_size, vocab_size, dtype=dtype, reduction=reduction).to(device=device)
    liger_lm_head_ce = LigerLMHeadCE(hidden_size, vocab_size, dtype=dtype, reduction=reduction, bwd_impl="cce").to(
        device=device
    )
    liger_chunk_lm_head_ce = LigerLMHeadCE(
        hidden_size, vocab_size, dtype=dtype, reduction=reduction, bwd_impl="chunk"
    ).to(device=device)
    cce_lm_head_ce = CutLMHeadCE(hidden_size, vocab_size, dtype=dtype, reduction=reduction).to(device=device)
    triton_liger_lm_head_ce = TritonLigerLMHeadCE(hidden_size, vocab_size, dtype=dtype, reduction=reduction).to(
        device=device
    )

    ref_lm_head_ce.lm_head.weight.data = weight.data
    liger_lm_head_ce.lm_head.weight.data = weight.data
    liger_chunk_lm_head_ce.lm_head.weight.data = weight.data
    cce_lm_head_ce.lm_head.weight.data = weight.data
    triton_liger_lm_head_ce.lm_head.weight.data = weight.data

    def fwd_bwd_fn(input, target, fn):
        loss = fn(input, target)
        loss.backward()
        return loss

    liger_lm_head_ce_fwd_bwd = partial(fwd_bwd_fn, fn=liger_lm_head_ce)
    liger_chunk_lm_head_ce_fwd_bwd = partial(fwd_bwd_fn, fn=liger_chunk_lm_head_ce)
    ref_lm_head_ce_fwd_bwd = partial(fwd_bwd_fn, fn=ref_lm_head_ce)
    cce_lm_head_ce_fwd_bwd = partial(fwd_bwd_fn, fn=cce_lm_head_ce)
    triton_liger_lm_head_ce_fwd_bwd = partial(fwd_bwd_fn, fn=triton_liger_lm_head_ce)

    # Test and Benchmark

    run_example(
        liger_lm_head_ce,
        {
            "torch_fwd": ref_lm_head_ce,
            "cce_fwd": cce_lm_head_ce,
            "triton_flce_fwd": triton_liger_lm_head_ce,
        },
        (input, target),
        kernel_name="helion_fwd",
        rtol=rtol * 10,
        atol=atol,
    )
    if reduction != "none":
        run_example(
            {
                "helion_fwd_bwd_cce": liger_lm_head_ce_fwd_bwd,
                "helion_fwd_bwd_chunk": liger_chunk_lm_head_ce_fwd_bwd,
            },
            {
                "torch_fwd_bwd": ref_lm_head_ce_fwd_bwd,
                "cce_fwd_bwd": cce_lm_head_ce_fwd_bwd,
                "triton_flce_fwd_bwd": triton_liger_lm_head_ce_fwd_bwd,
            },
            (input, target),
            rtol=rtol,
            atol=atol,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autotune", default=False)
    parser.add_argument("--benchmark", default=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    if args.benchmark:
        check()

    model_config_dataset = {
        "llama": {
            "hidden_size": 4096,
            "vocab_size": 32000,
        },
        "gemma3": {
            "hidden_size": 2305,
            "vocab_size": 262208,
        },
        "qwen3": {
            "hidden_size": 4096,
            "vocab_size": 151936,
        },
    }

    if args.autotune:
        print("autotuning all kernels...")
        autotune_kernels(model_config_dataset=model_config_dataset)
