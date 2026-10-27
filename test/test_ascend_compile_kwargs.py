"""Ascend compile defaults for Triton-Ascend 3.6.0. No device required."""

from __future__ import annotations

import liger_kernel.ops.backends._ascend.compile_kwargs as compile_kwargs


def test_compile_kwargs_empty_without_compiler_option(monkeypatch):
    monkeypatch.setattr(compile_kwargs, "_npu_option_fields", lambda: {})
    kwargs = {"num_warps": 4}
    assert compile_kwargs.apply_ascend_compile_kwargs(kwargs) == {"num_warps": 4}


def test_compile_kwargs_defaults_to_one_slot_and_keeps_override(monkeypatch):
    monkeypatch.setattr(
        compile_kwargs,
        "_npu_option_fields",
        lambda: {"set_workspace_multibuffer": None},
    )
    assert compile_kwargs.apply_ascend_compile_kwargs({"num_warps": 4}) == {
        "num_warps": 4,
        "set_workspace_multibuffer": 1,
    }
    assert compile_kwargs.apply_ascend_compile_kwargs({"set_workspace_multibuffer": 2}) == {
        "set_workspace_multibuffer": 2,
    }


def test_install_wraps_jit_function_run_once():
    from triton.runtime.jit import JITFunction

    compile_kwargs.install_ascend_compile_kwargs()
    wrapped = JITFunction.run
    compile_kwargs.install_ascend_compile_kwargs()
    assert JITFunction.run is wrapped
