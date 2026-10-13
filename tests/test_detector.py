import numpy as np

from halt.baseline import SIGNALS
from halt.detector import Detector, band_position, out_of_band


def make_band(low, high, epochs=5):
    """The same healthy range for every dense1 signal, every epoch."""
    return {"dense1": {s: {"low": np.full(epochs, low),
                           "high": np.full(epochs, high)}
                       for s in SIGNALS}}


def make_entry(epoch, value):
    """One epoch where every dense1 signal equals value."""
    return {"epoch": epoch, "layers": {"dense1": {s: value for s in SIGNALS}}}


def make_detector():
    return Detector(make_band(0.01, 0.05), margin=0.5, floor=0.0, patience=3)


def test_out_of_band_frozen_zero_always_flags():
    assert out_of_band(0.0, 0.01, 0.05, 0.5, 0.0) is True


def test_out_of_band_inside_range_is_fine():
    assert out_of_band(0.03, 0.01, 0.05, 0.5, 0.0) is False


def test_out_of_band_floor_allows_stray_dead_unit():
    assert out_of_band(0.01, 0.0, 0.0, 0.5, 0.05) is False


def test_out_of_band_large_dead_fraction_flags():
    assert out_of_band(0.5, 0.0, 0.0, 0.5, 0.05) is True


def test_detector_fires_after_patience_epochs():
    detector = make_detector()
    assert detector.check(make_entry(1, 0.0)) is None
    assert detector.check(make_entry(2, 0.0)) is None
    fired = detector.check(make_entry(3, 0.0))
    assert fired is not None
    assert fired["epoch"] == 3
    assert fired["layer"] == "dense1"


def test_detector_streak_resets_on_healthy_epoch():
    detector = make_detector()
    for epoch, value in enumerate([0.0, 0.0, 0.03, 0.0, 0.0], start=1):
        assert detector.check(make_entry(epoch, value)) is None


def test_detector_never_fires_on_healthy_run():
    detector = make_detector()
    for epoch in range(1, 6):
        detector.check(make_entry(epoch, 0.03))
    assert detector.fired is None


def test_detector_fires_only_once():
    detector = make_detector()
    for epoch in range(1, 4):
        detector.check(make_entry(epoch, 0.0))
    fired = detector.check(make_entry(4, 0.0))
    assert fired is not None
    assert fired["epoch"] == 3


def test_band_position_below():
    assert band_position(0.0, 0.01, 0.05, 0.5, 0.0) == "below"


def test_band_position_inside():
    assert band_position(0.03, 0.01, 0.05, 0.5, 0.0) == "inside"


def test_band_position_above():
    assert band_position(1.0, 0.01, 0.05, 0.5, 0.0) == "above"