import torch
import torch.nn as nn
from torchviz import make_dot

from liger_kernel.transformers.rms_norm import LigerRMSNorm


class BaseRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L112
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L122
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# "bs, sl, hd",
#     [
#         (2, 2, 8),
#         "dtype, atol, rtol",
#     [
#         (torch.float32, 1e-4, 1e-6),
#         (torch.bfloat16, 2e-1, 2e-2),
#     "reference, offset, casting_mode",
#     [
#         (LlamaRMSNorm, 0.0, "llama"),
#         (GemmaRMSNorm, 1.0, "gemma"),
(bs, sl, hd) = (2, 2, 8)
dtype = torch.bfloat16
offset = 1.0
casting_mode = "gemma"

_tensor = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

h1 = _tensor.clone().requires_grad_(True)
h2 = _tensor.clone().requires_grad_(True)

# do
do = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

triton_rms = (
    LigerRMSNorm(hidden_size=hd, offset=offset, casting_mode=casting_mode)
    .to("cuda")
    .to(dtype)
)
triton_o = triton_rms(h2)
make_dot(triton_o, params=dict(triton_rms.named_parameters()))
triton_o.backward(do.clone(), retain_graph=True)

# reference (llama or gemma)
ref_rms = GemmaRMSNorm(hidden_size=hd).to("cuda").to(dtype)
ref_o = ref_rms(h1)
make_dot(ref_o, params=dict(ref_rms.named_parameters()))
ref_o.backward(do.clone(), retain_graph=True)
