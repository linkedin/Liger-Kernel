# sanity_check_fused_rms_norm.py
import torch

# ⬇️ UPDATE these import paths to match your repo layout
from liger_kernel.ops.simple_rms_norm import (
    LigerRMSNormFunction,      # the Triton-fused autograd Function
)

# -----------------------------------------------------------------------------
# 🟢 Pure-PyTorch reference (no fusion, no Triton)
# -----------------------------------------------------------------------------
def reference_residual_rmsnorm(hidden, residual, weight, eps=1e-5, offset=0.0):
    """Implements:
       S  = hidden + residual
       residual_out = S                           # saved for next block
       rms = sqrt(mean(S²) + eps)
       hidden_out = (S / rms) * (weight + offset)
       returns (hidden_out, residual_out)
    """
    S = hidden + residual
    print("S", S)
    rms = torch.sqrt(S.pow(2).mean(-1, keepdim=True) + eps)
    print("rms", rms)
    hidden_out = (S / rms) * (weight + offset)
    print("hidden_out", hidden_out)
    return hidden_out, S


# -----------------------------------------------------------------------------
# 🧪 Test runner
# -----------------------------------------------------------------------------
def run_test():
    torch.manual_seed(0)

    # ---------------- Hyper-params ----------------
    B, H = 8, 4096
    eps  = 1e-5
    offset = 0.0
    mode   = "none"        # "llama" / "gemma" / "none"

    # ---------------- Inputs ----------------
    hidden = torch.randn(B, H, device="cuda", dtype=torch.float16, requires_grad=True)
    residual = torch.randn_like(hidden, requires_grad=True)
    weight = torch.randn(H,  device="cuda", dtype=torch.float16, requires_grad=True)

    print("hidden", hidden)
    print("residual", residual)
    print("weight", weight)

    #                                    ── reference ──
    h_ref   = hidden.detach().clone().float().requires_grad_(True)
    r_ref   = residual.detach().clone().float().requires_grad_(True)
    w_ref   = weight.detach().clone().float().requires_grad_(True)
    hidden_ref_out, residual_ref_out = reference_residual_rmsnorm(h_ref, r_ref, w_ref, eps, offset)
    #                                    ── Triton kernel ──
    hidden_out, residual_out = LigerRMSNormFunction.apply(
        hidden, residual, weight, eps, offset, mode
    )

    # ---------------- Forward check ----------------
    assert torch.allclose(hidden_out, hidden_ref_out.half(), atol=1e-3, rtol=1e-3), "Hidden output mismatch"
    assert torch.allclose(residual_out, residual_ref_out.half(), atol=1e-3, rtol=1e-3), "Residual output mismatch"

    # ---------------- Back-prop ----------------
    grad_hidden   = torch.randn_like(hidden_out)
    grad_residual = torch.randn_likelike(residual_out)

    grad_hidden_ref = grad_hidden.detach().clone().float()
    grad_residual_ref = grad_residual.detach().clone().float()

    print("grad_hidden", grad_hidden)
    print("grad_residual", grad_residual)

    # Triton path
    torch.autograd.backward((hidden_out, residual_out), (grad_hidden, grad_residual))

    # Reference path
    (dx_ref, dr_ref, dw_ref) = torch.autograd.grad(
        outputs=(hidden_ref_out, residual_ref_out),
        inputs=(h_ref, r_ref, w_ref),
        grad_outputs=(grad_hidden_ref, grad_residual_ref),
        retain_graph=False,
        create_graph=False,
    )
    print("dx_ref", dx_ref.half())
    print("hidden.grad", hidden.grad)
    print("dr_ref", dr_ref)
    print("residual.grad", residual.grad)
    print("dw_ref", dw_ref)
    print("weight.grad", weight.grad)
    # ---------------- Gradient check ----------------
    assert torch.allclose(hidden.grad, dx_ref.half(), atol=1e-2, rtol=1e-2), "dHidden mismatch"
    assert torch.allclose(residual.grad, dr_ref.half(), atol=1e-2, rtol=1e-2), "dResidual mismatch"
    assert torch.allclose(weight.grad, dw_ref.half(), atol=1e-2, rtol=1e-2), "dWeight mismatch"

    print("✅ Fused Residual + RMSNorm kernel passes forward & backward tests")


if __name__ == "__main__":
    run_test()
