import helion
import helion.language as hl
import torch


@helion.kernel(autotune_effort="none")
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
        loss: loss tensor of shape [1]
    """
    BT, H = x.size()
    V = weight.size(0)
    block_size_bt = hl.register_block_size(BT)
    block_size_h = hl.register_block_size(H)
    block_size_v = hl.register_block_size(V)

    logits = torch.empty(BT, V, device=x.device, dtype=torch.float32)
    lse = torch.full((BT,), fill_value=-torch.inf, device=x.device, dtype=torch.float32)  # DEBUG
    nll = torch.zeros(BT, device=x.device, dtype=torch.float32)
    neg_target_logits = torch.zeros(BT, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(BT, H, device=x.device, dtype=torch.float32)


    for tile_bt in hl.tile(BT, block_size=block_size_bt):
        m_i = hl.zeros([tile_bt], dtype=torch.float32) - float("inf")
        d_i = hl.zeros([tile_bt], dtype=torch.float32)
        nll_tile = hl.zeros([tile_bt], dtype=torch.float32)
        # target_indices = target[tile_bt][:, None]  # [tile_bt, 1] # ERROR
        target_indices = target[tile_bt].unsqueeze(1)  # [tile_bt, 1]
        for tile_v in hl.tile(V, block_size=block_size_v):
            # logits computation
            acc = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
            for tile_h in hl.tile(H, block_size=block_size_h):
                x_tile = x[tile_bt, tile_h]
                weight_tile = weight[tile_v, tile_h]
                acc = hl.dot(x_tile, weight_tile.T, acc=acc, out_dtype=torch.float32)


            logits[tile_bt, tile_v] = acc  # DEBUG

            # online softmax statistics
            m_ij = torch.maximum(m_i, torch.amax(acc, dim=-1))
            d_i = d_i * torch.exp(m_i - m_ij) + torch.exp(acc - m_ij[:, None]).sum(dim=-1)
            m_i = m_ij

            # offset = tile_v.index[None, :]  # [1, tile_v] # ERROR
            offset = tile_v.index.unsqueeze(0)  # [1, tile_v]
            mask = target_indices == offset  # [tile_bt, tile_v]
            nll_tile += torch.sum(-acc * mask, dim=-1)  # [tile_bt]

        # loss computation: -logsoftmax(x_y) = -log(exp(x_y) / sum(exp(x_i))) = -x_y + log(sum(exp(x_i)))
        lse_tile = m_i + torch.log(d_i)
        lse[tile_bt] = lse_tile

        neg_target_logits[tile_bt] = nll_tile

        nll_tile = nll_tile + lse_tile
        nll[tile_bt] = nll_tile

        # gradients computation
        for tile_v in hl.tile(V, block_size=block_size_v):
            # Restore logits
            # acc = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
            # for tile_h in hl.tile(H, block_size=block_size_h):
            #     x_tile = x[tile_bt, tile_h]
            #     weight_tile = weight[tile_v, tile_h]
            #     acc = hl.dot(x_tile, weight_tile.T, acc=acc, out_dtype=torch.float32)

            logits_tile = logits[tile_bt, tile_v]

            # softmax(x_i) = exp(x_i) / sum(exp(x_i)) 
            #              = exp(x_i) / log(exp(sum(x_i)))
            #              = exp(x_i) / lse = exp(x_i - lse)
            grad_logits_tile = torch.exp(logits_tile - lse_tile[:, None]) 
            
            # grad_x = grad_logits @ weight
            for tile_h in hl.tile(H, block_size=block_size_h):
                weight_tile = weight[tile_v, tile_h]
                partial_grad_x = hl.dot(grad_logits_tile, weight_tile, out_dtype=torch.float32)
                hl.atomic_add(grad_x, [tile_bt, tile_h], partial_grad_x)

    if reduction == "mean":
        loss = nll.sum() / nll.numel()
    elif reduction == "sum":
        loss = nll.sum()
    else:
        loss = nll

    return loss, logits.to(x.dtype), lse, neg_target_logits, grad_x




# class LigerFusedLinearCrossEntropyHelionFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         _input,
#         weight,
#         target,
#         ignore_index=-100,
#         reduction="mean",
#     ):
#         assert _input.ndim == weight.ndim
#         loss, grad_input, grad_weight = fused_linear_cross_entropy_fwd_bwd(
#             _input,
#             weight,
#             target,
#             ignore_index,
#             reduction,
#         )
#         ctx.save_for_backward(grad_input, grad_weight)
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input, grad_weight = ctx.saved_tensors
#         return grad_input * grad_output, grad_weight * grad_output, None, None, None


class LigerFusedLinearCrossEntropyHelion(torch.nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, _input, weight, target):
        # return LigerFusedLinearCrossEntropyHelionFunction.apply(
        #     _input,
        #     weight,
        #     target,
        #     self.ignore_index,
        #     self.reduction
        # )
        return fused_linear_cross_entropy_fwd(_input, weight, target, self.ignore_index, self.reduction)


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

    def forward(self, x, target):
        logits = self.lm_head(x).to(torch.float32)
        return self.ce_loss(logits, target)


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
    hidden_size = 4096
    vocab_size = 32000
    dtype = torch.float32
    reduction = "none"
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
    ref_loss = ref_lm_head_ce(ref_input, target)
    ref_logits = input @ weight.T
    liger_loss, liger_logits, liger_lse, liger_neg_target_logits, liger_grad_x = liger_lm_head_ce(liger_input, target)

    liger_logprobs = torch.nn.functional.log_softmax(liger_logits, dim=-1)
    ref_logprobs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

    ref_lse = torch.logsumexp(ref_logits, dim=-1)
    ref_neg_target_logits = torch.nn.functional.nll_loss(ref_logits, target, reduction="none")
    ref_neg_target_logits2 = torch.masked_select(
        ref_logits, mask=target[:, None] == torch.arange(vocab_size, device=ref_logits.device)[None, :]
    )


    for i in range(5):
        print("=" * 30 + f"(i = {i})" + "=" * 30)
        print(f"{ref_lse[i]=}")
        print(f"{ref_neg_target_logits[i]=}")
        print(f"{ref_neg_target_logits[i] + ref_lse[i]=}")
        print(f"{ref_loss[i]=}")
        print(f"{liger_loss[i]=}")
        print("=" * 64)

    torch.testing.assert_close(liger_logprobs, ref_logprobs, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(liger_lse, ref_lse, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(liger_neg_target_logits, ref_neg_target_logits, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(ref_loss, liger_loss, rtol=1e-1, atol=1e-1)


    # Backward pass
    ref_loss.backward()
    torch.testing.assert_close(liger_grad_x, ref_input.grad, rtol=1e-1, atol=1e-1)

    # ref_loss.backward()
    # liger_loss.backward()

    # torch.testing.assert_close(ref_input.grad, liger_input.grad)
    # torch.testing.assert_close(ref_lm_head_ce.lm_head.weight.grad, liger_lm_head_ce.lm_head.weight.grad)
