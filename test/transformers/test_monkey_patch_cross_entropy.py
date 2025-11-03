"""
Test cross_entropy monkey patches for all supported models.

Note: This test uses subprocess isolation because cross_entropy patches modify
a global function (transformers.loss.loss_utils.nn.functional.cross_entropy).
Once patched by any model, it affects all subsequent tests in the same process,
making it impossible to verify individual model patches independently.

By running each test in a separate Python process, we ensure that:
1. Each model's patch is tested in isolation
2. Failures can be correctly attributed to specific models
3. The test suite can detect when a patch is incorrectly targeting the wrong object

Trade-off: ~20x slower (60s vs 3s) but provides accurate per-model validation.
"""

import importlib
import inspect
import subprocess
import sys

import pytest
import transformers

from packaging import version

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"


def _extract_model_configs():
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    configs = []
    seen_functions = set()

    for model_type, apply_fn in MODEL_TYPE_TO_APPLY_LIGER_FN.items():
        if apply_fn in seen_functions:
            continue
        seen_functions.add(apply_fn)

        fn_name = apply_fn.__name__
        model_name = fn_name.replace("apply_liger_kernel_to_", "")

        sig = inspect.signature(apply_fn)
        if "cross_entropy" not in sig.parameters:
            continue

        transformers_module = f"transformers.models.{model_name}"

        configs.append(
            {
                "name": model_name,
                "module": transformers_module,
                "apply_fn_name": fn_name,
            }
        )

    return configs


MODEL_CONFIGS = _extract_model_configs()


def is_model_available(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def should_skip_model(model_config):
    if transformer_version < version.parse(SUPPORTED_TRANSFORMER_VERSION):
        return True, f"transformers version {transformer_version} < {SUPPORTED_TRANSFORMER_VERSION}"
    if not is_model_available(model_config["module"]):
        return True, f"{model_config['name']} not available"
    return False, None


ISOLATED_TEST_SCRIPT = '''
import sys
import torch.nn.functional

def test_single_model_patch():
    from liger_kernel.transformers import monkey_patch
    
    apply_fn_name = "{apply_fn_name}"
    model_name = "{model_name}"
    
    from transformers.loss import loss_utils
    original_ce = torch.nn.functional.cross_entropy
    
    if loss_utils.nn.functional.cross_entropy != original_ce:
        print(f"FAIL: cross_entropy was already patched before testing {{model_name}}")
        sys.exit(1)
    
    apply_fn = getattr(monkey_patch, apply_fn_name)
    
    try:
        apply_fn(cross_entropy=True, fused_linear_cross_entropy=False)
    except Exception as e:
        print(f"FAIL: Failed to apply patch: {{e}}")
        sys.exit(1)
    
    patched_ce = loss_utils.nn.functional.cross_entropy
    
    if patched_ce == original_ce:
        print(f"FAIL: cross_entropy was not patched")
        sys.exit(1)
    
    if "liger" not in patched_ce.__module__.lower():
        print(f"FAIL: cross_entropy module is {{patched_ce.__module__}}, expected liger")
        sys.exit(1)
    
    print(f"PASS: {{model_name}} patched correctly to {{patched_ce.__module__}}")
    sys.exit(0)

if __name__ == "__main__":
    test_single_model_patch()
'''


@pytest.mark.parametrize("model_config", MODEL_CONFIGS, ids=[m["name"] for m in MODEL_CONFIGS])
def test_cross_entropy_patch(model_config):
    should_skip, reason = should_skip_model(model_config)
    if should_skip:
        pytest.skip(reason)

    script = ISOLATED_TEST_SCRIPT.format(
        apply_fn_name=model_config["apply_fn_name"],
        model_name=model_config["name"],
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout + result.stderr

    if result.returncode != 0:
        pytest.fail(f"{model_config['name']} test failed:\n{output}")

    assert "PASS" in output, f"{model_config['name']}: Unexpected output:\n{output}"
