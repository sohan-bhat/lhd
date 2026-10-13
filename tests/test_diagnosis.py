import numpy as np

from halt.baseline import SIGNALS
from halt.diagnosis import positions


def make_band(low, high, epochs=5):
    """The same healthy range for every dense1 signal, every epoch."""
    return {"dense1": {s: {"low": np.full(epochs, low),
                           "high": np.full(epochs, high)}
                       for s in SIGNALS}}


def make_entry(epoch, value):
    """One epoch where every dense1 signal equals value."""
    return {"epoch": epoch, "layers": {"dense1": {s: value for s in SIGNALS}}}


def test_positions_all_inside():
    result = positions(make_entry(1, 0.03), make_band(0.01, 0.05), "dense1",
                       margin=0.5, floor=0.0)
    assert result == {s: "inside" for s in SIGNALS}


def test_positions_frozen_reads_below():
    result = positions(make_entry(1, 0.0), make_band(0.01, 0.05), "dense1",
                       margin=0.5, floor=0.0)
    assert result == {s: "below" for s in SIGNALS}
