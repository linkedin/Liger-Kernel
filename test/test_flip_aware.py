"""CPU tests for the flip-aware MoE convergence gate in test/utils.py."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import test.utils as test_utils

from test.utils import RoutingRecorder
from test.utils import compare_losses_with_flip_gate
from test.utils import first_routing_flip


class _FakeRouter(nn.Module):
    """Stand-in for a HF ``*Router`` module: returns the logits assigned to it."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self):
        return self.logits


def _make_model(num_routers=2, num_tokens=3, num_experts=4, num_experts_per_tok=2):
    model = nn.Module()
    model.config = SimpleNamespace(num_experts_per_tok=num_experts_per_tok)
    model.routers = nn.ModuleList([_FakeRouter(torch.zeros(num_tokens, num_experts)) for _ in range(num_routers)])
    return model


def _record(model, recorder, step_logits):
    for router, logits in zip(model.routers, step_logits):
        router.logits = logits
        router()
    recorder.end_step()


_BASE_LOGITS = torch.tensor(
    [
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 2.0, 1.0],
        [2.0, 1.0, 4.0, 3.0],
    ]
)


def _identical_logits(num_routers=2):
    return [_BASE_LOGITS.clone() for _ in range(num_routers)]


def test_identical_routing_has_no_flip():
    model_a, model_b = _make_model(), _make_model()
    rec_a, rec_b = RoutingRecorder(model_a), RoutingRecorder(model_b)
    for _ in range(2):
        _record(model_a, rec_a, _identical_logits())
        _record(model_b, rec_b, _identical_logits())

    assert rec_a.n_routers == 2
    assert first_routing_flip(rec_a, rec_b) == (None, [0, 0])


def test_flip_at_step_zero_counts_tokens():
    model_a, model_b = _make_model(), _make_model()
    rec_a, rec_b = RoutingRecorder(model_a), RoutingRecorder(model_b)

    logits_a = _identical_logits()
    logits_b = _identical_logits()
    logits_b[0] = logits_b[0].clone()
    logits_b[0][0] = torch.tensor([0.0, 0.0, 5.0, 0.5])  # token 0: top-2 {0,1} -> {2,3}

    _record(model_a, rec_a, logits_a)
    _record(model_b, rec_b, logits_b)

    assert first_routing_flip(rec_a, rec_b) == (0, [1])


def test_flip_at_later_step_sums_across_routers():
    model_a, model_b = _make_model(), _make_model()
    rec_a, rec_b = RoutingRecorder(model_a), RoutingRecorder(model_b)

    _record(model_a, rec_a, _identical_logits())
    _record(model_b, rec_b, _identical_logits())

    step1_a = _identical_logits()
    step1_b = _identical_logits()
    step1_b[0] = step1_b[0].clone()
    step1_b[1] = step1_b[1].clone()
    step1_b[0][0] = torch.tensor([0.0, 0.0, 5.0, 0.5])  # router 0, token 0
    step1_b[1][2] = torch.tensor([0.0, 3.0, 0.0, 5.0])  # router 1, token 2

    _record(model_a, rec_a, step1_a)
    _record(model_b, rec_b, step1_b)

    assert first_routing_flip(rec_a, rec_b) == (1, [0, 2])


def test_selection_order_is_ignored():
    model_a, model_b = _make_model(num_routers=1), _make_model(num_routers=1)
    rec_a, rec_b = RoutingRecorder(model_a), RoutingRecorder(model_b)

    logits_a = [torch.tensor([[4.0, 3.0, 2.0, 1.0], [0.0, 0.0, 5.0, 4.0], [4.0, 0.5, 0.0, 0.0]])]
    logits_b = [torch.tensor([[3.0, 4.0, 2.0, 1.0], [0.0, 0.0, 5.0, 4.0], [4.0, 0.5, 0.0, 0.0]])]

    _record(model_a, rec_a, logits_a)
    _record(model_b, rec_b, logits_b)

    assert first_routing_flip(rec_a, rec_b) == (None, [0])


def test_dense_model_falls_back_silently(capsys):
    model = nn.Module()
    model.config = SimpleNamespace()
    model.linear = nn.Linear(2, 2)

    recorder = RoutingRecorder(model)
    assert recorder.n_routers == 0
    assert recorder.capture_warning is None
    assert first_routing_flip(recorder, RoutingRecorder(model)) == (None, [])
    assert "[routing]" not in capsys.readouterr().out


def test_missing_num_experts_per_tok_disables_capture(capsys):
    model = _make_model()
    model.config = SimpleNamespace()

    recorder = RoutingRecorder(model)
    assert recorder.n_routers == 2
    assert "num_experts_per_tok" in recorder.capture_warning
    assert first_routing_flip(recorder, RoutingRecorder(model)) == (None, [])
    assert "capture unusable" in capsys.readouterr().out


def test_mismatched_step_counts_fall_back(capsys):
    model_a, model_b = _make_model(), _make_model()
    rec_a, rec_b = RoutingRecorder(model_a), RoutingRecorder(model_b)
    _record(model_a, rec_a, _identical_logits())
    _record(model_a, rec_a, _identical_logits())
    _record(model_b, rec_b, _identical_logits())

    assert first_routing_flip(rec_a, rec_b) == (None, [])
    assert "recorded 2 vs 1 steps" in capsys.readouterr().out


def test_different_router_sets_fall_back(capsys):
    model_a, model_b = _make_model(num_routers=2), _make_model(num_routers=1)
    rec_a, rec_b = RoutingRecorder(model_a), RoutingRecorder(model_b)
    _record(model_a, rec_a, _identical_logits(num_routers=2))
    _record(model_b, rec_b, _identical_logits(num_routers=1))

    assert first_routing_flip(rec_a, rec_b) == (None, [])
    assert "different router modules" in capsys.readouterr().out


def test_env_flag_disables_the_gate(monkeypatch):
    model_a, model_b = _make_model(), _make_model()
    rec_a, rec_b = RoutingRecorder(model_a), RoutingRecorder(model_b)
    logits_b = _identical_logits()
    logits_b[0] = logits_b[0].clone()
    logits_b[0][0] = torch.tensor([0.0, 0.0, 5.0, 0.5])
    _record(model_a, rec_a, _identical_logits())
    _record(model_b, rec_b, logits_b)

    monkeypatch.setattr(test_utils, "FLIP_AWARE_ENABLED", False)
    assert first_routing_flip(rec_a, rec_b) == (None, [])


def test_k_is_read_from_text_config():
    model = _make_model(num_routers=1, num_experts_per_tok=1)
    model.config = SimpleNamespace(text_config=SimpleNamespace(num_experts_per_tok=1))
    assert RoutingRecorder(model).k == 1


def test_gate_compares_all_steps_without_flip():
    expected = [1.0, 2.0, 3.0]
    assert compare_losses_with_flip_gate(expected, list(expected), None, 1e-3, 1e-3) is False
    with pytest.raises(AssertionError):
        compare_losses_with_flip_gate(expected, [1.0, 2.0, 3.1], None, 1e-3, 1e-3)


def test_gate_compares_prefix_before_mid_run_flip():
    expected = [1.0, 2.0, 3.0]
    assert compare_losses_with_flip_gate(expected, [1.0, 2.0, 99.0], 2, 1e-3, 1e-3) is False
    with pytest.raises(AssertionError):
        compare_losses_with_flip_gate(expected, [1.0, 2.5, 3.0], 2, 1e-3, 1e-3)


def test_gate_compares_step_zero_on_step_zero_flip():
    assert compare_losses_with_flip_gate([1.0, 2.0], [1.0, 99.0], 0, 1e-3, 1e-3) is True
    with pytest.raises(AssertionError):
        compare_losses_with_flip_gate([1.0, 2.0], [1.5, 2.0], 0, 1e-3, 1e-3)
