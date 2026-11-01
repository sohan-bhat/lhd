import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from experiments.train import PlantedFault  # noqa: E402
from lhd.torch_adapter import TorchAdapter  # noqa: E402


class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(20, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def lhd_layers(self):
        return {"fc1": (self.fc1, self.act, -1), "fc2": (self.fc2, self.fc2, -1)}


def setup(spec, lr=0.05):
    model = Wide()
    adapter = TorchAdapter(model, model.lhd_layers())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    fault = PlantedFault(spec, model, adapter, optimizer, seed=0)
    return model, adapter, optimizer, fault


def batch():
    x = torch.randn(64, 20) - 0.8
    return x, torch.randint(0, 3, (64,))


def train(model, optimizer, fault, steps):
    for _ in range(steps):
        x, y = batch()
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
        fault.after_step()


def test_late_dead_units_stay_dead_despite_adam_momentum():
    torch.manual_seed(1)
    model, adapter, optimizer, fault = setup({"type": "dead", "layer": "fc1", "fraction": 0.5, "onset": 2})
    fault.on_epoch_start(1)
    train(model, optimizer, fault, 50)
    fault.on_epoch_start(2)
    train(model, optimizer, fault, 200)
    probe = torch.randn(512, 20) * 3
    out = adapter.probe_outputs(probe.numpy())["fc1"]
    assert (out[:, fault.units] == 0).all()
    assert len(fault.units) == 8


def test_detach_cuts_gradient_to_earlier_layers():
    model, adapter, optimizer, fault = setup({"type": "detach", "layer": "fc2", "onset": 1})
    fault.on_epoch_start(1)
    x, y = batch()
    nn.functional.cross_entropy(model(x), y).backward()
    assert not np.any(adapter.gradients("fc1")["weights"])
    assert np.any(adapter.gradients("fc2")["weights"])


def test_lr_fault_multiplies_learning_rate_at_onset():
    model, adapter, optimizer, fault = setup({"type": "lr", "factor": 10, "onset": 3}, lr=0.001)
    fault.on_epoch_start(2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    fault.on_epoch_start(3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
