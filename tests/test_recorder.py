import numpy as np
import pytest
from tiny_net import TinyAdapter, TinyNet

from halt.recorder import Recorder, flatten_parameters


def test_flattened_params_shape():
    arrays = {"weights": np.ones((2, 3)), "bias": np.zeros(3)}
    assert flatten_parameters(arrays).shape == (9,)


def test_flattened_params_order():
    arrays = {"weights": np.array([1.0]), "bias": np.array([2.0])}
    assert (flatten_parameters(arrays) == [2.0, 1.0]).all()


def test_recorder_before_snapshot_inplace():
    net = TinyNet()
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.array([]))
    recorder.start_epoch()
    original_snapshot = np.copy(recorder._before["dense1"])
    net.W1 += 1.0
    assert (recorder._before["dense1"] == original_snapshot).all()


def test_after_batch_records_one_grad_norm_per_call():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.array([]))
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
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.5, seconds=0.0)
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
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.5, seconds=0.0)
    assert entry["layers"]["dense1"]["grad_norm"] == pytest.approx(2 * np.sqrt(12))


def test_end_epoch_bias_change():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    net.b1 += 1.0
    recorder.after_batch()
    entry = recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.5, seconds=0.0)
    assert entry["layers"]["dense1"]["bias_change"] == pytest.approx(2.0)


def test_end_epoch_appends_to_history():
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.5, seconds=0.0)
    assert len(recorder.history) == 1.0