import helion
import helion.language as hl
import torch


# TODO: autotune and find the best configs for different devices
@helion.kernel(autotune_effort="none", ignore_warnings=[helion.exc.TensorOperationInWrapper])
def fused_linear_cross_entropy_fwd_bwd(
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
    grad_x = torch.zeros_like(x, dtype=torch.float32)
    grad_w = torch.zeros_like(weight, dtype=torch.float32)

    # May be useful for splitting fwd and bwd
    # lse = torch.full((BT,), fill_value=-torch.inf, device=x.device, dtype=torch.float32)
    # neg_target_logits = torch.zeros(BT, device=x.device, dtype=torch.float32)

    n_non_ignore = (target != ignore_index).sum().unsqueeze(0)

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

        if reduction == "mean":
            nll_tile /= n_non_ignore_value

        nll[tile_bt] = nll_tile
        # gradients computation
        for tile_v in hl.tile(V, block_size=block_size_v):
            # Restore logits
            acc = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
            for tile_h in hl.tile(H, block_size=block_size_h):
                x_tile = x[tile_bt, tile_h]
                weight_tile = weight[tile_v, tile_h]
                acc = hl.dot(x_tile, weight_tile.T, acc=acc, out_dtype=torch.float32)

            # softmax(x_i) = exp(x_i) / sum(exp(x_i))
            #              = exp(x_i) / log(exp(sum(x_i)))
            #              = exp(x_i) / lse = exp(x_i - lse)
            grad_logits_tile = torch.exp(acc - lse_tile[:, None])
            offset = tile_v.index.unsqueeze(0)  # [1, tile_v]
            mask = target_indices == offset  # [tile_bt, tile_v]
            grad_logits_tile = grad_logits_tile - mask.float()
            # handle out of bound values in grad_logits_tile
            grad_logits_tile = grad_logits_tile * ((tile_bt.index < BT)[:, None] & (tile_v.index < V)[None, :])

            if reduction == "mean":
                grad_logits_tile /= n_non_ignore_value

            for tile_h in hl.tile(H, block_size=block_size_h):
                # grad_x = grad_logits @ weight
                rhs_tile = weight[tile_v, tile_h]
                partial_grad_x = hl.dot(grad_logits_tile, rhs_tile, out_dtype=torch.float32)
                hl.atomic_add(grad_x, [tile_bt, tile_h], partial_grad_x)
                # grad_w = grad_logits.T[tile_v, tile_bt] @ x[tile_bt, tile_h]
                rhs_tile = x[tile_bt, tile_h]
                partial_grad_w = hl.dot(grad_logits_tile.T, rhs_tile, out_dtype=torch.float32)
                hl.atomic_add(grad_w, [tile_v, tile_h], partial_grad_w)

    if reduction != "none":
        loss = nll.sum() 
    else:
        loss = nll

    # return format is not determined yet
    return loss, dict(
        {
            "grad_x": grad_x,
            "grad_w": grad_w,
        }
    )


class LigerFusedLinearCrossEntropyHelionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        ignore_index=-100,
        reduction="mean",
    ):
        loss, aux_output = fused_linear_cross_entropy_fwd_bwd(
            _input,
            weight,
            target,
            ignore_index,
            reduction,
        )
        ctx.save_for_backward(aux_output["grad_x"], aux_output["grad_w"])
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.ndim == 0, "token_scaling is not supported. grad_output must be a scalar"
        grad_input, grad_weight = ctx.saved_tensors
        return grad_input * grad_output, grad_weight * grad_output, None, None, None


class LigerFusedLinearCrossEntropyHelion(torch.nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, _input, weight, target):
        return LigerFusedLinearCrossEntropyHelionFunction.apply(
            _input, weight, target, self.ignore_index, self.reduction
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
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.flce = LigerFusedLinearCrossEntropyHelion(ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, target):
        return self.flce(x, self.lm_head.weight, target)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    device = "cuda"

    batch_size = 2
    seq_len = 1024
    hidden_size = 1024
    vocab_size = 2048
    dtype = torch.float32
    reduction = "sum"
    ignore_index = -100

    input = torch.randn(batch_size * seq_len, hidden_size, device=device, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device=device, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)

    # Init
    ref_lm_head_ce = TorchLMHeadCE(hidden_size, vocab_size, dtype=dtype, reduction=reduction).to(device=device)
    liger_lm_head_ce = LigerLMHeadCE(hidden_size, vocab_size, dtype=dtype, reduction=reduction).to(device=device)

    ref_lm_head_ce.lm_head.weight.data = weight.data
    liger_lm_head_ce.lm_head.weight.data = weight.data

    ref_input = input.detach().clone().requires_grad_(True)
    liger_input = input.detach().clone().requires_grad_(True)

    # Forward pass
    ref_loss: torch.Tensor = ref_lm_head_ce(ref_input, target)
    liger_loss: torch.Tensor = liger_lm_head_ce(liger_input, target)

    torch.testing.assert_close(liger_loss, ref_loss, rtol=1e-1, atol=1e-1)

    # Backward pass (backward() with reduction=="none" is not supported yet)
    if reduction == "none":
        pass
    else:
        liger_loss.backward()
        ref_loss.backward()

        torch.testing.assert_close(liger_input.grad, ref_input.grad, rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(
            liger_lm_head_ce.lm_head.weight.grad, ref_lm_head_ce.lm_head.weight.grad, rtol=1e-1, atol=1e-1
        )
