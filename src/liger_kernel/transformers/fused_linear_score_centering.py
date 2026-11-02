from typing import Optional

import torch

from liger_kernel.ops import ISCorrection
from liger_kernel.ops import fused_linear_score_centering

__all__ = ["ISCorrection", "LigerFusedLinearScoreCenteringLoss"]


class LigerFusedLinearScoreCenteringLoss(torch.nn.Module):
    """Fused lm_head + score centering loss, optionally on top of a token-level IS correction (TIS, MIS, PPO, ...).

    Returns (loss, metrics) with the masked means of the policy entropy, the head KL(q || p) and the fraction of
    tokens whose IS weight is zero.
    """

    def __init__(
        self,
        score_center: bool = True,
        correction: Optional[ISCorrection] = None,
        temperature: float = 1.0,
        eps: float = 1e-6,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            score_center (bool): Subtract the expected score under q * w. False gives plain (IS-corrected) REINFORCE.
            correction (ISCorrection, optional): Token-level IS correction; None is plain REINFORCE.
            temperature (float): Sampling temperature the logits are divided by.
            eps (float): Floor on the trainer's tail mass.
            chunk_size (int, optional): Rows of logits materialized at once; None picks at least 1024.
        """
        super().__init__()
        assert temperature != 0, "Temperature cannot be 0."
        self.score_center = score_center
        self.correction = correction.check() if correction is not None else None
        self.temperature = temperature
        self.eps = eps
        self.chunk_size = chunk_size

    def forward(
        self,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        sampler_token_logps,
        sampler_topk_ids,
        sampler_topk_logps,
        bias=None,
        num_items_in_batch=None,
    ):
        return fused_linear_score_centering(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            sampler_token_logps,
            sampler_topk_ids,
            sampler_topk_logps,
            bias,
            self.score_center,
            self.correction,
            self.temperature,
            self.eps,
            num_items_in_batch,
            self.chunk_size,
        )

    def extra_repr(self) -> str:
        return f"score_center={self.score_center}, correction={self.correction}, temperature={self.temperature}"
