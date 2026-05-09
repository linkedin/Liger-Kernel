import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.mhc import LigerMHC
from liger_kernel.utils import infer_device

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

device = infer_device()


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, eps: float, dtype: torch.dtype, device: str):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, *, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MiniLlamaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, *, dtype: torch.dtype, device: str):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = _build_rope_cache(seq_len, self.head_dim, device=x.device, dtype=q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn)


class MiniLlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_mult: int, *, dtype: torch.dtype, device: str):
        super().__init__()
        intermediate_size = hidden_size * intermediate_mult
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, *, dtype: torch.dtype, device: str):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.attn = MiniLlamaAttention(hidden_size, num_heads, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm(x))


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_mult: int, *, dtype: torch.dtype, device: str):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.mlp = MiniLlamaMLP(hidden_size, intermediate_mult, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class TorchMHC(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        *,
        hc: int,
        c: int,
        tmax: int,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        phi_dtype: torch.dtype,
    ):
        super().__init__()
        self.layer = layer
        self.hc = int(hc)
        self.c = int(c)
        self.tmax = int(tmax)
        self.rms_eps = float(rms_eps)
        self.pre_eps = float(pre_eps)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.post_mult = float(post_mult)

        layer_param = next(layer.parameters())
        device = layer_param.device

        m = hc * hc + 2 * hc
        k = hc * c
        self.phi = nn.Parameter(torch.randn(k, m, dtype=phi_dtype, device=device) * 0.02)
        self.b = nn.Parameter(torch.zeros(m, dtype=torch.float32, device=device))
        self.alpha_pre = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        self.alpha_post = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        self.alpha_res = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))

        self.layer_dtype = layer_param.dtype

    def _coeffs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from test.transformers.test_mhc import mhc_coeffs_ref

        return mhc_coeffs_ref(
            x,
            self.phi,
            self.b,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            tmax=self.tmax,
            rms_eps=self.rms_eps,
            pre_eps=self.pre_eps,
            sinkhorn_eps=self.sinkhorn_eps,
            post_mult=self.post_mult,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_pre, h_post, h_res = self._coeffs(x)
        x_in = (x.float() * h_pre.unsqueeze(-1)).sum(dim=-2)
        if x_in.dtype != self.layer_dtype:
            x_in = x_in.to(self.layer_dtype)
        f_out = self.layer(x_in)
        x_out = torch.einsum("...oi,...ic->...oc", h_res, x.float()) + h_post.unsqueeze(-1) * f_out.float().unsqueeze(
            -2
        )
        return x_out.to(x.dtype)


class MHCDecoderLayer(nn.Module):
    def __init__(
        self,
        mhc_cls: type[nn.Module],
        *,
        hidden_size: int,
        hc: int,
        num_heads: int,
        intermediate_mult: int,
        tmax: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        attn = AttentionBlock(hidden_size, num_heads, dtype=dtype, device=device)
        mlp = MLPBlock(hidden_size, intermediate_mult, dtype=dtype, device=device)
        mhc_kwargs = dict(
            hc=hc,
            c=hidden_size,
            tmax=tmax,
            rms_eps=1e-6,
            pre_eps=1e-4,
            sinkhorn_eps=1e-6,
            post_mult=2.0,
            phi_dtype=dtype,
        )
        self.attn = mhc_cls(attn, **mhc_kwargs)
        self.mlp = mhc_cls(mlp, **mhc_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class BenchMiniMHCLM(nn.Module):
    def __init__(
        self,
        mhc_cls: type[nn.Module],
        *,
        vocab_size: int,
        hidden_size: int,
        hc: int,
        num_layers: int,
        num_heads: int,
        intermediate_mult: int,
        tmax: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hc = hc
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hc * hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList(
            [
                MHCDecoderLayer(
                    mhc_cls,
                    hidden_size=hidden_size,
                    hc=hc,
                    num_heads=num_heads,
                    intermediate_mult=intermediate_mult,
                    tmax=tmax,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.hc, self.hidden_size)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-2)
        x = self.final_norm(x)
        return self.lm_head(x)


def _build_model(
    provider: str,
    *,
    hidden_size: int,
    hc: int,
    num_layers: int,
    num_heads: int,
    intermediate_mult: int,
    vocab_size: int,
    tmax: int,
    dtype: torch.dtype,
):
    mhc_cls = LigerMHC if provider == "liger" else TorchMHC
    return BenchMiniMHCLM(
        mhc_cls,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        hc=hc,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_mult=intermediate_mult,
        tmax=tmax,
        dtype=dtype,
        device=device,
    )


def _setup_mhc_lm(input: SingleBenchmarkRunInput):
    """Create model and inputs for MHC LM benchmark."""
    cfg = input.extra_benchmark_config
    hidden_size = cfg["hidden_size"]
    bsz = cfg["B"]
    seq_len = cfg.get("T", input.x)
    hc = cfg["HC"]
    num_layers = cfg["layers"]
    num_heads = cfg["heads"]
    vocab_size = cfg["vocab"]
    dtype = cfg["dtype"]
    tmax = cfg["tmax"]
    intermediate_mult = cfg["intermediate_mult"]

    model = _build_model(
        input.kernel_provider,
        hidden_size=hidden_size,
        hc=hc,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_mult=intermediate_mult,
        vocab_size=vocab_size,
        tmax=tmax,
        dtype=dtype,
    )

    input_ids = torch.randint(0, vocab_size, (bsz, seq_len), device=device)
    grad_to_none = list(model.parameters())

    fwd_fn = lambda: model(input_ids)
    fwd_loss_fn = lambda: fwd_fn().float().mean()
    return grad_to_none, fwd_fn, fwd_loss_fn


def bench_speed_mhc_lm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd_fn, fwd_loss = _setup_mhc_lm(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd_fn, quantiles=QUANTILES, grad_to_none=grad_to_none, rep=100)
    elif mode == "backward":
        loss = fwd_loss()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: loss.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=grad_to_none,
            rep=100,
        )
    elif mode == "full":

        def full():
            loss = fwd_loss()
            loss.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, grad_to_none=grad_to_none, rep=100)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_mhc_lm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd_fn, fwd_loss_fn = _setup_mhc_lm(input)

    def full():
        loss = fwd_loss_fn()
        loss.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


def _resolve_model_config_mhc_lm(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_mhc_lm(
        SingleBenchmarkRunInput(
            x=input.x,
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "dtype": model_info["dtype"],
                "B": cfg["B"],
                "T": cfg["T"],
                "HC": cfg["HC"],
                "layers": cfg["layers"],
                "heads": cfg["heads"],
                "vocab": cfg["vocab"],
                "tmax": cfg["tmax"],
                "intermediate_mult": cfg["intermediate_mult"],
            },
        )
    )


def bench_speed_mhc_lm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd_fn, fwd_loss = _resolve_model_config_mhc_lm(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd_fn, quantiles=QUANTILES, grad_to_none=grad_to_none, rep=100)
    elif mode == "backward":
        loss = fwd_loss()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: loss.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=grad_to_none,
            rep=100,
        )
    elif mode == "full":

        def full():
            loss = fwd_loss()
            loss.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, grad_to_none=grad_to_none, rep=100)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_mhc_lm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    grad_to_none, fwd_fn, fwd_loss = _resolve_model_config_mhc_lm(input)

    def full():
        loss = fwd_loss()
        loss.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    mhc_lm_defaults = {"HC": 4, "layers": 2, "heads": 8, "vocab": 4096, "tmax": 8, "intermediate_mult": 4}

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())
        B = 2

        def _probe_factory(model_cfg, probe_bt):
            def _probe():
                T = max(1, probe_bt // B)
                probe_input = SingleBenchmarkRunInput(
                    x=0,
                    kernel_provider="torch",
                    extra_benchmark_config={
                        "hidden_size": model_cfg.hidden_size,
                        "dtype": model_cfg.dtype,
                        "B": B,
                        "T": T,
                        **mhc_lm_defaults,
                    },
                )
                _, _, fwd_loss_fn = _setup_mhc_lm(probe_input)
                return fwd_loss_fn()

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
        model_configs_info = {
            cfg.name: {"hidden_size": cfg.hidden_size, "dtype": cfg.dtype} for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "mhc_llama_like_lm",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "B": sweep.batch_size,
                    "T": sweep.seq_len,
                    **mhc_lm_defaults,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_mhc_lm_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_mhc_lm_model_config,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        B = 2
        probe_T = 256

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=0,
                kernel_provider="torch",
                extra_benchmark_config={
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "B": B,
                    "T": probe_T,
                    **mhc_lm_defaults,
                },
            )
            _, _, fwd_loss_fn = _setup_mhc_lm(probe_input)
            return fwd_loss_fn()

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_T)

        common_configs = {
            "kernel_name": "mhc_llama_like_lm",
            "x_name": "T",
            "x_label": "sequence length",
            "x_values": [2**i for i in range(7, int(math.log2(max(128, config.seq_len))) + 1)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "hidden_size": model.hidden_size,
                    "B": B,
                    "dtype": model.dtype,
                    **mhc_lm_defaults,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_mhc_lm,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_mhc_lm,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
