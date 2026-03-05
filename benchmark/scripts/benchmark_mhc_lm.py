import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

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
        self.attn = mhc_cls(
            attn,
            hc=hc,
            c=hidden_size,
            tmax=tmax,
            rms_eps=1e-6,
            pre_eps=1e-4,
            sinkhorn_eps=1e-6,
            post_mult=2.0,
            phi_dtype=dtype,
        )
        self.mlp = mhc_cls(
            mlp,
            hc=hc,
            c=hidden_size,
            tmax=tmax,
            rms_eps=1e-6,
            pre_eps=1e-4,
            sinkhorn_eps=1e-6,
            post_mult=2.0,
            phi_dtype=dtype,
        )

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


def bench_speed_mhc_lm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    hidden_size = int(input.x)
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra = input.extra_benchmark_config
    bsz = extra["B"]
    seq_len = extra["T"]
    hc = extra["HC"]
    num_layers = extra["layers"]
    num_heads = extra["heads"]
    vocab_size = extra["vocab"]
    dtype = extra["dtype"]
    tmax = extra["tmax"]
    intermediate_mult = extra["intermediate_mult"]

    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")

    model = _build_model(
        provider,
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

    def fwd():
        return model(input_ids)

    def fwd_loss():
        return fwd().float().mean()

    grad_to_none = list(model.parameters())

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, grad_to_none=grad_to_none, rep=100)
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
        raise ValueError(f"Unknown mode: {mode}")

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_mhc_lm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    hidden_size = int(input.x)
    provider = input.kernel_provider
    extra = input.extra_benchmark_config
    bsz = extra["B"]
    seq_len = extra["T"]
    hc = extra["HC"]
    num_layers = extra["layers"]
    num_heads = extra["heads"]
    vocab_size = extra["vocab"]
    dtype = extra["dtype"]
    tmax = extra["tmax"]
    intermediate_mult = extra["intermediate_mult"]

    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")

    model = _build_model(
        provider,
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

    def fwd():
        return model(input_ids)

    def full():
        loss = fwd().float().mean()
        loss.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "mhc_llama_like_lm",
        "x_name": "hidden_size",
        "x_label": "hidden_size",
        "x_values": [256, 512, 1024],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "B": 2,
                "T": 256,
                "HC": 4,
                "layers": 2,
                "heads": 8,
                "vocab": 4096,
                "dtype": torch.bfloat16,
                "tmax": 8,
                "intermediate_mult": 4,
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
