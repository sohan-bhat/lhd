import json

import numpy as np
import pytest
from tiny_net import TinyAdapter, TinyNet

from lhd.recorder import Recorder, flatten_parameters


def test_flattened_params_shape():
    arrays = {"weights": np.ones((2, 3)), "bias": np.zeros(3)}
    assert flatten_parameters(arrays).shape == (9,)


def test_flattened_params_order():
    arrays = {"weights": np.array([1.0]), "bias": np.array([2.0])}
    assert (flatten_parameters(arrays) == [2.0, 1.0]).all()


def test_recorder_before_snapshot_inplace():
    net = TinyNet()
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    original_snapshot = np.copy(recorder._before["dense1"])
    net.W1 += 1.0
    assert (recorder._before["dense1"] == original_snapshot).all()


def test_after_batch_records_one_grad_norm_per_call():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    recorder.after_batch()
    assert len(recorder._grad_norms["dense1"]) == 2
    assert recorder._grad_norms["dense1"][0] == pytest.approx(np.sqrt(12))


def test_end_epoch_frozen_layer_reads_zero_change():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert entry["layers"]["dense1"]["weight_change"] == 0.0


def test_end_epoch_averages_grad_norms():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    net.dW1 = np.ones((3, 4)) * 3
    recorder.after_batch()
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert entry["layers"]["dense1"]["grad_norm"] == pytest.approx(2 * np.sqrt(12))


def test_end_epoch_bias_change():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    net.b1 += 1.0
    recorder.after_batch()
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert entry["layers"]["dense1"]["bias_change"] == pytest.approx(2.0)


def test_end_epoch_appends_to_history():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert len(recorder.history) == 1


def test_save_writes_readable_json(tmp_path):
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    path = tmp_path / "run.json"
    recorder.save(str(path), metadata={"seed": 0})
    with open(path) as f:
        data = json.load(f)
    assert data["metadata"] == {"seed": 0}
    assert len(data["history"]) == 1


def test_end_epoch_without_batches_raises():
    net = TinyNet()
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    with pytest.raises(ValueError):
        recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)

def _dying_net():
    """A TinyNet whose first hidden unit is dead on the probe from the start."""
    net = TinyNet()
    net.W1[:, 0] = 0.0
    net.b1[0] = -1.0
    net.dW1 = np.ones((3, 4))
    return net


def test_never_fired_counts_units_dead_before_training():
    net = _dying_net()
    recorder = Recorder(TinyAdapter(net), probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert entry["layers"]["dense1"]["never_fired_fraction"] >= 0.25


def test_never_fired_drops_a_unit_once_it_fires():
    net = _dying_net()
    recorder = Recorder(TinyAdapter(net), probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    first = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    net.b1[0] = 5.0
    recorder.start_epoch()
    recorder.after_batch()
    second = recorder.end_epoch(epoch=2, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    net.b1[0] = -1.0
    recorder.start_epoch()
    recorder.after_batch()
    third = recorder.end_epoch(epoch=3, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    never = [e["layers"]["dense1"]["never_fired_fraction"] for e in (first, second, third)]
    dead = [e["layers"]["dense1"]["dead_unit_fraction"] for e in (first, second, third)]
    assert never[1] == never[0] - 0.25
    assert never[2] == never[1]
    assert dead[2] == dead[0]


def test_never_fired_ignores_units_that_die_during_training():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    recorder = Recorder(TinyAdapter(net), probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    before = recorder._never_fired["dense1"].copy()
    net.W1[:] = 0.0
    net.b1[:] = -1.0
    recorder.after_batch()
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert entry["layers"]["dense1"]["dead_unit_fraction"] == 1.0
    assert entry["layers"]["dense1"]["never_fired_fraction"] == float(np.mean(before))
