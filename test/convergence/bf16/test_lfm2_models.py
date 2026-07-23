import pytest
import torch

from test.convergence.lfm2_utils import run_lfm2_convergence
from test.utils import require_deterministic
from test.utils import supports_bfloat16


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported")
@require_deterministic
@pytest.mark.parametrize("model_kind", ["lfm2", "lfm2_moe", "lfm2_vl"])
def test_lfm2_models_converge_bf16(model_kind):
    expected_losses, expected_parameters = run_lfm2_convergence(model_kind, torch.bfloat16, with_liger=False)
    actual_losses, actual_parameters = run_lfm2_convergence(model_kind, torch.bfloat16, with_liger=True)

    torch.testing.assert_close(actual_losses, expected_losses, atol=5e-2, rtol=5e-2)
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in actual_parameters:
        torch.testing.assert_close(
            actual_parameters[name],
            expected_parameters[name],
            atol=2e-2,
            rtol=2e-2,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
