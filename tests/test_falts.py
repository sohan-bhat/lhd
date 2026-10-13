import numpy as np
import pytest
from tiny_net import TinyAdapter, TinyNet

from halt.faults import FrozenLayer, kill_units
from halt.recorder import Recorder
from halt.signals import dead_unit_fraction


def test_frozen_layer_restores_after_update():
    net = TinyNet()
    adapter = TinyAdapter(net)
    original = net.W1.copy()
    frozen = FrozenLayer(adapter, "dense1")
    net.W1 += 1.0
    frozen.restore()
    assert np.array_equal(net.W1, original)


def test_frozen_layer_writes_in_place():
    net = TinyNet()
    adapter = TinyAdapter(net)
    w = net.W1
    frozen = FrozenLayer(adapter, "dense1")
    w += 1
    frozen.restore()
    assert net.W1 is w


def test_frozen_layer_shows_row_two_in_recorder():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    frozen = FrozenLayer(adapter, "dense1")
    recorder.start_epoch()
    net.W1 += 1
    frozen.restore()
    recorder.after_batch()
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert entry["layers"]["dense1"]["weight_change"] == 0.0
    assert entry["layers"]["dense1"]["grad_norm"] > 0


def test_kill_units_zeroes_chosen_unit():
    net = TinyNet()
    adapter = TinyAdapter(net)
    kill_units(adapter, "dense1", [1])
    outs = adapter.probe_outputs(np.vstack([np.eye(3), -np.eye(3)]))
    assert (outs["dense1"][:, 1] == 0).all()


def test_kill_units_dead_fraction():
    net = TinyNet()
    adapter = TinyAdapter(net)
    kill_units(adapter, "dense1", [0, 2])
    outs = adapter.probe_outputs(np.vstack([np.eye(3), -np.eye(3)]))
    assert dead_unit_fraction(outs["dense1"]) == pytest.approx(0.5)


def test_kill_units_leaves_others_alive():
    net = TinyNet()
    adapter = TinyAdapter(net)
    kill_units(adapter, "dense1", [1])
    outs = adapter.probe_outputs(np.vstack([np.eye(3), -np.eye(3)]))
    for col in [0, 2, 3]:
        assert (outs["dense1"][:, col] != 0).any()