#!/usr/bin/env python
"""Compare the fused-linear-cross-entropy (FLCE) backends at matched shapes and
explicit token ``chunk_size`` overrides.

This is a *self-contained* micro-benchmark for the FLCE backends that expose an
explicit chunk-size control:

* ``triton``   — the portable Triton composed op
                 (:mod:`liger_kernel.ops.fused_linear_cross_entropy`).
* ``cutedsl``  — the native SM100 CuTe DSL implementation
                 (:mod:`liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy`).
* ``cutile``   — the cuTile implementation
                 (:mod:`liger_kernel.ops.cutile.ops.fused_linear_cross_entropy`).

It imports the ``autograd.Function`` classes **directly** (no dispatcher, no
``LIGER_KERNEL_IMPL`` env) so a single process can time whichever backends the
current GPU supports side by side.

This is a *whole-pipeline* comparison (full forward + backward), not a CE-only
DSL-isolation harness. The backends differ in which parts run in a custom
kernel vs. PyTorch / cuBLAS:

* ``triton``  — the projection (``x @ w.T``), ``dX`` and ``dW`` GEMMs all run
  through PyTorch / cuBLAS; the CE reduction runs in Triton.
* ``cutedsl`` — the projection (``x @ w.T``) runs through a CuTe-DSL SM100 GEMM;
  the ``dX`` / ``dW`` GEMMs run through PyTorch / cuBLAS.
* ``cutile``  — the projection, ``dX`` and ``dW`` GEMMs all use PyTorch BLAS;
  cuTile owns the partitioned CE statistics, the dZ scatter and the final FP32
  dW scale-and-cast.

Capability policy (imports fail loudly if the SDK is missing):

* ``cutedsl`` requires exactly SM100 (Blackwell, e.g. B200).
* ``cutile``  runs on SM90 (Hopper) **and** SM100 (Blackwell).
* On SM100 the default is all three backends; on SM90 it is ``triton`` +
  ``cutile``; elsewhere it is ``triton`` only. Explicitly requesting a backend the
  GPU cannot run raises rather than silently degrading.

Timing uses CUDA events over a fresh full forward+backward each iteration with a
fixed scalar upstream gradient of 1; both ``x`` and ``w`` are trainable and both
leaf grads are cleared before every iteration. One unconditional compile pass plus
the requested warmup are excluded from the measurement. No random tensors are
allocated inside the timed region and ``retain_graph`` is never used.

Peak-memory numbers are the *incremental* peak **allocated** bytes above the
per-backend input-only baseline (both leaf grads are cleared, then peak stats are
reset and the device synchronized, before the dedicated memory-probe iteration).

Requested vs. effective chunk size: a chunk larger than the token count ``N`` is
clamped to ``N``; the table reports the ``chunk`` you asked for and the
``eff_chunk`` actually used, so a clamp is never mistaken for the request.

Cross-chunk ``dW`` accumulator (``--accum-dtype``): ``fp32`` (the default, kept for
backward compatibility) passes ``torch.float32`` to every backend so the ``dW``
accumulation precision is matched across ``triton`` / ``cutedsl`` / ``cutile``.
``default`` passes ``None`` and lets each backend apply its own policy: with the
BF16 weights used here ``triton`` and the native ``cutedsl`` path store the
cross-chunk ``dW`` in **BF16**, while ``cutile`` **always** retains an **FP32**
accumulator even when ``None`` is requested. The header states the requested mode
and the table's ``eff_accum`` column reports the effective accumulator per backend,
so the ``default`` panel is never mislabelled as precision-matched.

Results are printed to stdout only — no CSV is written into the repository.

Example::

    .venv/bin/python benchmark/scripts/benchmark_flce_backends.py \\
        --backends all3 --tokens 4096 --hidden-size 2048 --vocab-size 32000 \\
        --chunk-sizes 256 1024 4096

    # per-backend default accumulator policy (triton/cutedsl BF16, cutile FP32):
    .venv/bin/python benchmark/scripts/benchmark_flce_backends.py \\
        --backends all3 --accum-dtype default
"""

from __future__ import annotations

import argparse
import importlib
import statistics

import torch

_ACCUM_DTYPE = torch.float32
_DTYPE = torch.bfloat16

# ``--accum-dtype`` maps a requested mode onto the value passed at Function slot 11:
# ``fp32`` -> torch.float32 (matched precision), ``default`` -> None (per-backend policy).
_ACCUM_DTYPE_CHOICES = {"fp32": torch.float32, "default": None}

_MODULE_PATHS = {
    "triton": "liger_kernel.ops.fused_linear_cross_entropy",
    "cutedsl": "liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy",
    "cutile": "liger_kernel.ops.cutile.ops.fused_linear_cross_entropy",
}

# Optional SDK each native backend imports; a missing one is surfaced loudly.
_BACKEND_DEP = {"cutedsl": "cutlass.cute", "cutile": "cuda.tile"}


def _load_backend(name):
    """Import a backend's ``LigerFusedLinearCrossEntropyFunction`` directly."""
    if name not in _MODULE_PATHS:
        raise ValueError(f"Unknown backend {name!r}. Choose from {sorted(_MODULE_PATHS)}.")
    module = importlib.import_module(_MODULE_PATHS[name])
    return module.LigerFusedLinearCrossEntropyFunction


def _capability():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to benchmark the FLCE backends.")
    return torch.cuda.get_device_capability()


def _backend_cc_ok(name, cc):
    """Pure capability predicate (no torch / SDK imports)."""
    if name == "triton":
        return True
    if name == "cutedsl":
        return cc == (10, 0)
    if name == "cutile":
        return cc in ((9, 0), (10, 0))
    return False


def _backend_requirement(name):
    if name == "cutedsl":
        return "exact SM100 (10, 0)"
    if name == "cutile":
        return "SM90 (9, 0) or SM100 (10, 0)"
    return "any CUDA GPU"


def _check_backend_supported(name, cc):
    """Raise a clear error if ``cc`` / the SDK cannot run ``name``."""
    if name not in _MODULE_PATHS:
        raise ValueError(f"Unknown backend {name!r}. Choose from {sorted(_MODULE_PATHS)}.")
    if not _backend_cc_ok(name, cc):
        raise RuntimeError(f"backend {name!r} requires {_backend_requirement(name)}; this GPU is {cc}.")
    dep = _BACKEND_DEP.get(name)
    if dep is not None:
        # Let a missing / broken SDK propagate its own ImportError verbatim.
        importlib.import_module(dep)


def _select_default_backends(cc):
    """Triton always; add the DSL backend(s) the arch supports."""
    backends = ["triton"]
    if cc == (10, 0):
        backends += ["cutedsl", "cutile"]
    elif cc == (9, 0):
        backends += ["cutile"]
    return backends


def _resolve_backends(requested, cc):
    """Expand ``all3`` / apply the arch default and reject nonsensical mixes."""
    if not requested:
        return _select_default_backends(cc)
    if "all3" in requested:
        if len(requested) != 1:
            raise ValueError("'all3' is exclusive; do not combine it with individual backends.")
        return ["triton", "cutedsl", "cutile"]
    return list(requested)


def _effective_accum(name, requested):
    """Human label of the *effective* cross-chunk ``dW`` accumulator for BF16 weights.

    ``fp32`` forces ``torch.float32`` on every backend. Under ``default`` (``None``
    passed through) the native ``triton`` / ``cutedsl`` paths store the cross-chunk
    ``dW`` in BF16 to match the BF16 weights, while ``cutile`` *always* retains an
    FP32 accumulator regardless — so the two are not precision-matched.
    """
    if requested == "fp32":
        return "fp32"
    if name == "cutile":
        return "fp32"
    return "bf16"


def _build_inputs(tokens, hidden, vocab, device, seed=0):
    """Deterministic BF16 inputs — identical values reused for every backend."""
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, hidden, device=device, dtype=_DTYPE, generator=generator)
    w = torch.randn(vocab, hidden, device=device, dtype=_DTYPE, generator=generator) / (hidden**0.5)
    target = torch.randint(0, vocab, (tokens,), device=device, generator=generator)
    return x, w, target


def _apply(fn, x, w, target, chunk_size, accum_dtype=_ACCUM_DTYPE):
    # 15 base positional args + (ce_impl, ce_mode, chunk_size). ``accum_dtype`` is the
    # cross-chunk dW accumulator at slot 11: torch.float32 (matched precision) or None
    # (per-backend policy — see _effective_accum). Threaded as a parameter, never a
    # mutable global, so one process can compare modes without state leak.
    loss, _, _, _ = fn.apply(
        x,
        w,
        target,
        None,  # bias
        None,  # ce_weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        accum_dtype,  # accum_dtype
        False,  # use_token_scaling
        False,  # return_token_accuracy
        False,  # return_predicted_tokens
        None,  # ce_impl
        None,  # ce_mode
        chunk_size,
    )
    return loss


def _one_iteration(fn, x, w, target, chunk_size, upstream, accum_dtype=_ACCUM_DTYPE):
    x.grad = None
    w.grad = None
    loss = _apply(fn, x, w, target, chunk_size, accum_dtype=accum_dtype)
    loss.backward(upstream)


def _loss_value(fn, x, w, target, chunk_size, accum_dtype=_ACCUM_DTYPE):
    """Return a representative loss as a Python float, keeping no autograd graph alive."""
    loss = _apply(fn, x, w, target, chunk_size, accum_dtype=accum_dtype)
    value = float(loss.detach())
    del loss
    x.grad = None
    w.grad = None
    return value


def _time_backend(fn, x, w, target, chunk_size, warmup, iters, accum_dtype=_ACCUM_DTYPE):
    upstream = torch.ones((), device=x.device, dtype=torch.float32)

    # One unconditional compile pass (excluded) so warmup may legitimately be 0.
    _one_iteration(fn, x, w, target, chunk_size, upstream, accum_dtype=accum_dtype)
    for _ in range(warmup):
        _one_iteration(fn, x, w, target, chunk_size, upstream, accum_dtype=accum_dtype)

    # Baseline must be input-only: drop both leaf grads before snapshotting.
    x.grad = None
    w.grad = None
    torch.cuda.synchronize()

    # Dedicated memory probe (separate from the timed loop for a precise peak).
    torch.cuda.reset_peak_memory_stats(x.device)
    baseline = torch.cuda.memory_allocated(x.device)
    _one_iteration(fn, x, w, target, chunk_size, upstream, accum_dtype=accum_dtype)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(x.device)
    incremental_peak_mb = max(0.0, (peak - baseline) / (1024**2))
    x.grad = None
    w.grad = None

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _one_iteration(fn, x, w, target, chunk_size, upstream, accum_dtype=accum_dtype)
        ends[i].record()
    torch.cuda.synchronize()

    median_ms = statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))
    return median_ms, incremental_peak_mb


def _run(args):
    device = torch.device("cuda")
    cc = _capability()
    backends = _resolve_backends(args.backends, cc)

    # Validate up front so a misconfigured run fails before any timing work.
    for name in backends:
        _check_backend_supported(name, cc)

    requested = args.accum_dtype
    accum_value = _ACCUM_DTYPE_CHOICES[requested]
    if requested == "fp32":
        accum_note = "cross-chunk dW accumulator forced to FP32 on every backend (precision matched)"
    else:
        accum_note = (
            "per-backend accumulator policy for BF16 weights: triton/cutedsl store cross-chunk dW in "
            "BF16, cutile always retains an FP32 accumulator (NOT precision matched)"
        )

    print(
        f"device={torch.cuda.get_device_name(device)} cc={cc} "
        f"tokens={args.tokens} hidden={args.hidden_size} vocab={args.vocab_size} "
        f"dtype={_DTYPE} accum_dtype_requested={requested} ({accum_note})"
    )
    header = (
        f"{'backend':<10} {'chunk':>8} {'eff_chunk':>10} {'eff_accum':>10} "
        f"{'median_ms':>12} {'peak_MiB':>12} {'loss':>12}"
    )
    print(header)
    print("-" * len(header))

    for name in backends:
        fn = _load_backend(name)
        eff_accum = _effective_accum(name, requested)
        for chunk_size in args.chunk_sizes:
            effective = min(chunk_size, args.tokens)
            x, w, target = _build_inputs(args.tokens, args.hidden_size, args.vocab_size, device, seed=args.seed)
            x.requires_grad_(True)
            w.requires_grad_(True)
            # Representative loss (recomputed cleanly, no graph retained into timing).
            loss_val = _loss_value(fn, x, w, target, chunk_size, accum_dtype=accum_value)
            median_ms, peak_mb = _time_backend(
                fn, x, w, target, chunk_size, args.warmup, args.iters, accum_dtype=accum_value
            )
            print(
                f"{name:<10} {chunk_size:>8} {effective:>10} {eff_accum:>10} "
                f"{median_ms:>12.4f} {peak_mb:>12.2f} {loss_val:>12.5f}"
            )
            # Drop this shape's tensors before the next backend/chunk allocates.
            del x, w, target


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=_positive_int, default=2048, help="number of tokens B*T (default: 2048)")
    parser.add_argument("--hidden-size", type=_positive_int, default=1024, help="hidden dim H (default: 1024)")
    parser.add_argument("--vocab-size", type=_positive_int, default=32000, help="vocab dim V (default: 32000)")
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=_positive_int,
        default=[256, 1024],
        help="explicit positive chunk sizes to sweep (default: 256 1024)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=["triton", "cutedsl", "cutile", "all3"],
        help="backends to run; 'all3' is exclusive (default: triton + the DSL(s) the arch supports)",
    )
    parser.add_argument(
        "--warmup", type=_non_negative_int, default=5, help="warmup iterations excluded from timing (default: 5)"
    )
    parser.add_argument("--iters", type=_positive_int, default=20, help="timed iterations, must be >= 1 (default: 20)")
    parser.add_argument(
        "--accum-dtype",
        choices=sorted(_ACCUM_DTYPE_CHOICES),
        default="fp32",
        help=(
            "cross-chunk dW accumulator policy: 'fp32' (default, backward-compatible) passes "
            "torch.float32 to every backend so accumulation precision matches; 'default' passes None "
            "so each backend uses its own policy — with BF16 weights triton/cutedsl accumulate dW in "
            "BF16 while cutile always keeps an FP32 accumulator (default: fp32)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the shared inputs (default: 0)")
    return parser


def main():
    args = _build_parser().parse_args()
    _run(args)


if __name__ == "__main__":
    main()
