import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from halt.faults import FrozenLayer, kill_units  # noqa: E402
from halt.recorder import Recorder  # noqa: E402
from halt.signals import dead_unit_fraction  # noqa: E402
from halt.torch_adapter import TorchAdapter  # noqa: E402


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(3, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def halt_layers(self):
        return {"fc1": (self.fc1, self.act, -1), "fc2": (self.fc2, self.fc2, -1)}


def make():
    model = Small()
    return model, TorchAdapter(model, model.halt_layers())


def probe():
    return np.random.default_rng(0).standard_normal((16, 3)).astype(np.float32)


def step(model, optimizer):
    optimizer.zero_grad()
    model(torch.as_tensor(probe())).pow(2).mean().backward()
    optimizer.step()


def test_layer_names_keep_order():
    _, adapter = make()
    assert adapter.layer_names() == ["fc1", "fc2"]


def test_parameters_share_memory_with_model():
    model, adapter = make()
    adapter.parameters("fc1")["weights"][0, 0] = 5.0
    assert model.fc1.weight[0, 0].item() == 5.0


def test_gradients_are_zero_before_backward():
    _, adapter = make()
    grads = adapter.gradients("fc1")
    assert set(grads) == {"weights", "bias"}
    assert not grads["weights"].any()


def test_gradients_match_torch_after_backward():
    model, adapter = make()
    model(torch.as_tensor(probe())).sum().backward()
    assert np.allclose(adapter.gradients("fc2")["weights"], model.fc2.weight.grad.numpy())


def test_probe_outputs_after_activation():
    _, adapter = make()
    outs = adapter.probe_outputs(probe())
    assert outs["fc1"].shape == (16, 8)
    assert (outs["fc1"] >= 0).all()
    assert outs["fc2"].shape == (16, 2)


def test_probe_restores_mode_and_removes_hooks():
    model, adapter = make()
    model.train()
    adapter.probe_outputs(probe())
    assert model.training
    assert not model.act._forward_hooks


def test_frozen_layer_survives_optimizer_steps():
    model, adapter = make()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    fc1_before = model.fc1.weight.detach().clone()
    fc2_before = model.fc2.weight.detach().clone()
    frozen = FrozenLayer(adapter, "fc1")
    for _ in range(3):
        step(model, optimizer)
        frozen.restore()
    assert torch.equal(model.fc1.weight, fc1_before)
    assert not torch.equal(model.fc2.weight, fc2_before)


def test_kill_units_on_torch_linear():
    _, adapter = make()
    kill_units(adapter, "fc1", [0, 1, 2, 3], weight_axis=0, bias_value=-1.0)
    outs = adapter.probe_outputs(probe())
    assert dead_unit_fraction(outs["fc1"]) == pytest.approx(0.5)


def test_killed_units_stay_dead_through_training():
    model, adapter = make()
    kill_units(adapter, "fc1", [0, 1], weight_axis=0, bias_value=-1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(5):
        step(model, optimizer)
    outs = adapter.probe_outputs(probe())
    assert (outs["fc1"][:, :2] == 0).all()


def test_recorder_reads_frozen_torch_layer_as_zero_change():
    model, adapter = make()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    recorder = Recorder(adapter, probe())
    frozen = FrozenLayer(adapter, "fc1")
    recorder.start_epoch()
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.as_tensor(probe())).pow(2).mean().backward()
        recorder.after_batch()
        optimizer.step()
        frozen.restore()
    entry = recorder.end_epoch(1, 0.5, 0.5, 1.0)
    assert entry["layers"]["fc1"]["weight_change"] == 0.0
    assert entry["layers"]["fc1"]["grad_norm"] > 0
    assert entry["layers"]["fc2"]["weight_change"] > 0
