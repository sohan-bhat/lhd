import numpy as np
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