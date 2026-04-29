import torch

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss


def _build_wrapper(loss_fn: LigerCrossEntropyLoss):
    """Build a drop-in replacement for ``fused_vocab_parallel_cross_entropy``.

    The returned callable has exactly the same parameter list Megatron expects
    (``vocab_parallel_logits``, ``target``, ``tp_group``). Any unknown kwargs
    will raise ``TypeError`` naturally — this is intentional: if a future
    Megatron release adds new parameters to the fused-CE contract, we want to
    fail loudly rather than silently drop them.
    """

    def liger_fused_vocab_parallel_cross_entropy(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        tp_group=None,
    ) -> torch.Tensor:
        if tp_group is not None and hasattr(tp_group, "size") and tp_group.size() > 1:
            raise RuntimeError(
                f"Liger Megatron cross-entropy wrapper requires tensor_model_parallel_size=1, "
                f"got tp_group.size()={tp_group.size()}. Vocab-parallel support is tracked as "
                f"follow-up work."
            )

        s, b, v = vocab_parallel_logits.shape
        logits_2d = vocab_parallel_logits.reshape(-1, v)
        target_1d = target.reshape(-1)
        loss = loss_fn(logits_2d, target_1d)
        return loss.reshape(s, b)

    return liger_fused_vocab_parallel_cross_entropy


def _patch_fused_vocab_parallel_cross_entropy(
    reduction: str = "none",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> None:
    """Replace ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``.

    See ``apply_liger_kernel_to_megatron`` in ``monkey_patch.py`` for the public
    entry point; this helper holds the cross-entropy-specific patching logic so
    that future Megatron kernel integrations can sit alongside it without
    polluting the framework-level apply function.

    Args:
        reduction: Must be ``"none"``; Megatron's fused-CE contract returns
            per-token loss shaped ``[seq, batch]``.
        ignore_index: Target index to ignore.
        label_smoothing: Cross-entropy label smoothing factor.
    """
    assert reduction == "none", (
        f"Megatron's fused_vocab_parallel_cross_entropy contract requires per-token loss; "
        f"reduction must be 'none', got {reduction!r}."
    )

    try:
        import megatron.core.fusions.fused_cross_entropy as fce
    except ImportError as exc:
        raise ImportError(
            "apply_liger_kernel_to_megatron requires megatron-core to be installed. "
            "Expected symbol path: "
            "megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy."
        ) from exc

    if not hasattr(fce, "fused_vocab_parallel_cross_entropy"):
        raise ImportError(
            "megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy not found. "
            "The symbol path may have changed in your Megatron-LM version. Please file an issue "
            "on https://github.com/linkedin/Liger-Kernel with your megatron-core version."
        )

    loss_fn = LigerCrossEntropyLoss(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    fce.fused_vocab_parallel_cross_entropy = _build_wrapper(loss_fn)
