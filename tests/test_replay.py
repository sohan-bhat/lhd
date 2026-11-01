import numpy as np
from tiny_net import TinyAdapter, TinyNet

from lhd.baseline import SIGNALS
from lhd.faults import FrozenLayer
from lhd.monitor import Monitor
from lhd.replay import replay


def wide_band(epochs):
    band = {}
    for layer in ("dense1", "dense2"):
        band[layer] = {s: {"low": np.full(epochs, 1e-6), "high": np.full(epochs, 1e6)}
                       for s in SIGNALS}
        band[layer]["dead_unit_fraction"] = {"low": np.zeros(epochs), "high": np.zeros(epochs)}
    return band


def run_live(freeze_at, epochs=6, patience=2):
    """Train TinyNet by hand with a live Monitor, freezing dense1 from freeze_at."""
    net = TinyNet()
    rng = np.random.default_rng(0)
    band = wide_band(epochs)
    monitor = Monitor(TinyAdapter(net), np.vstack([np.eye(3), -np.eye(3)]),
                      band, margin=0.5, floor=0.05, patience=patience)
    frozen = None
    live_report = None
    for epoch in range(1, epochs + 1):
        if epoch == freeze_at:
            frozen = FrozenLayer(monitor.recorder.adapter, "dense1")
        monitor.start_epoch()
        for _ in range(3):
            net.dW1 = rng.standard_normal(net.W1.shape)
            net.dW2 = rng.standard_normal(net.W2.shape)
            monitor.after_batch()
            net.W1 -= 0.01 * net.dW1
            net.W2 -= 0.01 * net.dW2
            if frozen is not None:
                frozen.restore()
        report = monitor.end_epoch(epoch, 0.5, 0.5, 1.0)
        if report is not None and live_report is None:
            live_report = report
    return monitor.recorder.history, band, live_report


def test_replay_matches_live_monitor():
    history, band, live = run_live(freeze_at=3)
    replayed = replay(history, band, margin=0.5, floor=0.05, patience=2)
    assert live is not None
    assert {k: replayed[k] for k in live} == live
    assert replayed["epoch"] == 4
    assert replayed["cause"] == "frozen_layer"


def test_replay_silent_on_healthy_run():
    history, band, live = run_live(freeze_at=None)
    assert live is None
    assert replay(history, band, margin=0.5, floor=0.05, patience=2) is None


def test_replay_reports_positions():
    history, band, _ = run_live(freeze_at=1)
    replayed = replay(history, band, margin=0.5, floor=0.05, patience=2)
    assert replayed["positions"]["weight_change"] == "below"
