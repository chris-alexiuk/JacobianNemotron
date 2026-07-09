from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_jlens.loading import force_torch_mamba


class _FakeGatedNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.group_size = 2
        self.variance_epsilon = 1e-5

    def forward(self, hidden_states, gate=None):
        raise AssertionError("fused norm should have been replaced")


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _FakeGatedNorm()
        self.torch_calls = 0

    def forward(self, hidden_states, **kwargs):
        raise AssertionError("fused mixer should have been replaced")

    def torch_forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        self.torch_calls += 1
        return hidden_states + 1


class _FakeNemotron(nn.Module):
    def __init__(self, *, expected=2, actual=2):
        super().__init__()
        self.config = SimpleNamespace(
            layers_block_type=["mamba"] * expected + ["mlp"]
        )
        self.mixers = nn.ModuleList(
            [NemotronHMamba2Mixer() for _ in range(actual)]
        )


def test_force_torch_mamba_replaces_mixer_and_fused_group_norm():
    model = _FakeNemotron()
    assert force_torch_mamba(model) == 2

    x = torch.randn(2, 3, 4, requires_grad=True)
    gate = torch.randn(2, 3, 4, requires_grad=True)
    mixer = model.mixers[0]
    torch.testing.assert_close(mixer(x), x + 1)
    assert mixer.torch_calls == 1

    actual = mixer.norm(x, gate)
    gated = x.float() * torch.nn.functional.silu(gate.float())
    grouped = gated.reshape(2, 3, 2, 2)
    expected = grouped * torch.rsqrt(
        grouped.square().mean(dim=-1, keepdim=True) + 1e-5
    )
    expected = expected.reshape_as(x) * mixer.norm.weight.float()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(gate.grad).all()


def test_force_torch_mamba_fails_closed_on_remote_layout_drift():
    with pytest.raises(RuntimeError, match="patched 1.*expected 2"):
        force_torch_mamba(_FakeNemotron(expected=2, actual=1))

    model = _FakeNemotron(expected=0, actual=0)
    with pytest.raises(RuntimeError, match="patched 0.*expected 0"):
        force_torch_mamba(model)
