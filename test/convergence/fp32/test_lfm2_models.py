import pytest
import torch

from test.convergence.lfm2_utils import run_lfm2_convergence
from test.utils import require_deterministic


@require_deterministic
@pytest.mark.parametrize("model_kind", ["lfm2", "lfm2_moe", "lfm2_vl"])
def test_lfm2_models_converge_fp32(model_kind):
    expected_losses, expected_parameters = run_lfm2_convergence(model_kind, torch.float32, with_liger=False)
    actual_losses, actual_parameters = run_lfm2_convergence(model_kind, torch.float32, with_liger=True)

    atol = 5e-4 if model_kind == "lfm2_moe" else 2e-5
    rtol = 5e-4 if model_kind == "lfm2_moe" else 2e-4
    torch.testing.assert_close(actual_losses, expected_losses, atol=atol, rtol=rtol)
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in actual_parameters:
        torch.testing.assert_close(
            actual_parameters[name],
            expected_parameters[name],
            atol=atol,
            rtol=rtol,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
