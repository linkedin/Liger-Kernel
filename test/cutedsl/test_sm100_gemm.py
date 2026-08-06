import importlib.util

import pytest
import torch

cutedsl_available = importlib.util.find_spec("cutlass") is not None and torch.cuda.is_available()
sm100_available = cutedsl_available and torch.cuda.get_device_capability() == (10, 0)

pytestmark = pytest.mark.skipif(not sm100_available, reason="SM100 CuTe DSL GEMM requires an NVIDIA SM100 GPU")

if cutedsl_available:
    import cutlass
    import cutlass.cute as cute

    from liger_kernel.ops.cutedsl.ops._sm100_gemm import run_epilogue_gemm

    @cute.jit
    def _identity_epilogue(accumulator, output):
        output_dtype = output.element_type
        for element in cutlass.range_constexpr(cute.size(accumulator)):
            output[element] = accumulator[element].to(output_dtype)


@pytest.mark.parametrize("output_features", [96, 97])
def test_sm100_gemm_identity_epilogue(output_features):
    torch.manual_seed(0)
    tokens, hidden = 33, 128
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(output_features, hidden, device="cuda", dtype=torch.bfloat16) * hidden**-0.5
    actual = torch.empty(tokens, output_features, device="cuda", dtype=torch.bfloat16)

    run_epilogue_gemm(x, weight, actual, _identity_epilogue)

    expected = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.05, rtol=0.03)


def test_sm100_gemm_rejects_non_cute_epilogue():
    x = torch.empty(1, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(32, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 32, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="@cute.jit"):
        run_epilogue_gemm(x, weight, out, lambda accumulator, output: None)


def test_sm100_gemm_custom_epilogue_guards_noncurrent_device(monkeypatch):
    import liger_kernel.ops.cutedsl.ops._sm100_gemm as gemm

    entered = []
    called = []

    class Guard:
        def __init__(self, guarded_device):
            self.guarded_device = guarded_device

        def __enter__(self):
            entered.append(self.guarded_device)

        def __exit__(self, *_):
            return False

    class FakeTensor:
        device = torch.device("cuda", 1)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", Guard)
    monkeypatch.setattr(gemm, "_run_epilogue_gemm", lambda *args, **kwargs: called.append((args, kwargs)))

    gemm.run_epilogue_gemm(FakeTensor(), None, None, None)

    assert entered == [torch.device("cuda", 1)]
    assert len(called) == 1
