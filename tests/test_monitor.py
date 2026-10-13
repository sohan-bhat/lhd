import numpy as np
from tiny_net import TinyAdapter, TinyNet

from halt.baseline import SIGNALS
from halt.faults import FrozenLayer
from halt.monitor import Monitor


def wide_band(epochs=3):
    """A huge healthy range, so only clear faults leave it.

    Dead-unit fraction is healthy at exactly 0.
    """
    band = {}
    for layer in ("dense1", "dense2"):
        band[layer] = {s: {"low": np.full(epochs, 1e-6),
                           "high": np.full(epochs, 1e6)} for s in SIGNALS}
        band[layer]["dead_unit_fraction"] = {"low": np.zeros(epochs),
                                             "high": np.zeros(epochs)}
    return band


def make_monitor(net):
    net.dW1 = np.ones_like(net.W1)
    net.dW2 = np.ones_like(net.W2)
    return Monitor(TinyAdapter(net), probe_batch=np.vstack([np.eye(3), -np.eye(3)]),
                   band=wide_band(), margin=0.5, floor=0.05, patience=1)


def test_monitor_silent_on_healthy_epoch():
    net = TinyNet()
    monitor = make_monitor(net)
    monitor.start_epoch()
    net.W1 += 0.1
    net.W2 += 0.1
    monitor.after_batch()
    assert monitor.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0) is None


def test_monitor_reports_frozen_layer():
    net = TinyNet()
    monitor = make_monitor(net)
    frozen = FrozenLayer(monitor.recorder.adapter, "dense1")
    monitor.start_epoch()
    net.W1 += 0.1
    net.W2 += 0.1
    frozen.restore()
    monitor.after_batch()
    report = monitor.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    assert report is not None
    assert report["layer"] == "dense1"
    assert report["cause"] == "frozen_layer"
