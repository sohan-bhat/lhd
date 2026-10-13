import numpy as np

from halt.baseline import SIGNALS
from halt.diagnosis import diagnose, positions


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


def make_pos(**changes):
    """All signals inside, except the ones given."""
    pos = {s: "inside" for s in SIGNALS}
    pos.update(changes)
    return pos


def test_diagnose_diverged():
    assert diagnose(make_pos(grad_norm="invalid"))["cause"] == "diverged"


def test_diagnose_dead_units_wins_over_vanishing():
    assert diagnose(make_pos(dead_unit_fraction="above", weight_change="below", grad_norm="below"))["cause"] == "dead_units"


def test_diagnose_frozen_layer():
    assert diagnose(make_pos(weight_change="below"))["cause"] == "frozen_layer"


def test_diagnose_vanishing_gradient():
    assert diagnose(make_pos(weight_change="below", grad_norm="below"))["cause"] == "vanishing_gradient"


def test_diagnose_runaway_weights():
    assert diagnose(make_pos(weight_norm="above"))["cause"] == "runaway_weights"


def test_diagnose_unrecognized():
    assert diagnose(make_pos(grad_norm="above"))["cause"] == "unrecognized"