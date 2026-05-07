import logging

import torch

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

logger = logging.getLogger(__name__)

_ACTIVATION_LOGGED = False


def _check_tensor_parallel_size_at_patch_time() -> None:
    """Raise RuntimeError if Megatron's parallel state already reports TP>1.

    If Megatron is importable but the parallel state is not yet initialized
    (for example, ``apply_liger_kernel_to_megatron`` is called before
    ``initialize_megatron``), silently defer; the wrapper checks again at call
    time against the ``tp_group`` argument Megatron supplies.
    """
    try:
        from megatron.core import parallel_state
    except ImportError:
        return
    try:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    except (AssertionError, RuntimeError):
        return
    if tp_size > 1:
        raise RuntimeError(
            f"apply_liger_kernel_to_megatron currently requires tensor_model_parallel_size=1, "
            f"got {tp_size}. Vocab-parallel cross-entropy support is planned as follow-up work."
        )


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
        global _ACTIVATION_LOGGED
        if not _ACTIVATION_LOGGED:
            logger.info(
                "Liger cross-entropy kernel is active for Megatron-LM "
                "(replacing megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy)."
            )
            _ACTIVATION_LOGGED = True

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


def apply_liger_kernel_to_megatron(
    reduction: str = "none",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> None:
    """Replace Megatron-LM's fused_vocab_parallel_cross_entropy with Liger's Triton cross-entropy.

    This monkey-patches
    ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``
    so that Megatron training pipelines use Liger's Triton kernel (online
    softmax, in-place gradients, no full-softmax materialization) instead of
    Megatron's native fused implementation.

    Args:
        reduction: Must be ``"none"``; Megatron's fused-CE contract returns
            per-token loss shaped ``[seq, batch]`` and handles reduction itself
            downstream.
        ignore_index: Target index to ignore. Pass the value used in your
            Megatron training config.
        label_smoothing: Cross-entropy label smoothing factor. Liger does not
            auto-detect this — callers should pass
            ``cfg.label_smoothing_factor`` (or equivalent) from their
            Megatron ``TransformerConfig`` if label smoothing is enabled, to
            preserve the native behavior.

    Scope:
        Initial release supports ``tensor_model_parallel_size=1`` only. With
        TP>1, each rank holds a vocab-sharded logits slice ``[N, V/tp]`` and
        computing cross-entropy requires cross-rank all-reduces that Liger's
        kernel does not perform. A ``RuntimeError`` is raised at patch time if
        the Megatron parallel state already reports TP>1, and again at call
        time if a multi-rank ``tp_group`` is passed.

    Raises:
        AssertionError: If ``reduction != "none"``.
        ImportError: If ``megatron.core.fusions.fused_cross_entropy`` is not
            importable, or if the expected
            ``fused_vocab_parallel_cross_entropy`` symbol is missing from that
            module (indicating an incompatible Megatron version).
        RuntimeError: If tensor model parallelism > 1 is detected.

    Example:
        >>> from liger_kernel.megatron import apply_liger_kernel_to_megatron
        >>> apply_liger_kernel_to_megatron(
        ...     ignore_index=-100,
        ...     label_smoothing=cfg.label_smoothing_factor,
        ... )
        >>> # call before Megatron's forward pass reaches compute_language_model_loss
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

    _check_tensor_parallel_size_at_patch_time()

    loss_fn = LigerCrossEntropyLoss(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    fce.fused_vocab_parallel_cross_entropy = _build_wrapper(loss_fn)
