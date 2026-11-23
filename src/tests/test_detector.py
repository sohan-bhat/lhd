import numpy as np

from lhd.baseline import NEVER_FIRED, SIGNALS, SIGNALS_V2
from lhd.detector import Detector, band_position, out_of_band, signal_margin


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


def test_band_position_invalid_for_nan():
    assert band_position(float("nan"), 0.01, 0.05, 0.5, 0.0) == "invalid"


def test_band_position_invalid_for_inf():
    assert band_position(float("inf"), 0.01, 0.05, 0.5, 0.0) == "invalid"


def make_band_v2(low, high, epochs=5):
    """make_band with never_fired_fraction added."""
    return {"dense1": {s: {"low": np.full(epochs, low), "high": np.full(epochs, high)}
                       for s in SIGNALS_V2}}


def test_signal_margin_is_zero_only_for_never_fired():
    assert signal_margin(NEVER_FIRED, 0.5) == 0.0
    assert all(signal_margin(s, 0.5) == 0.5 for s in SIGNALS)


def test_detector_ignores_signals_the_band_does_not_have():
    detector = Detector(make_band(0.1, 0.2), margin=0.5, floor=0.05, patience=3)
    for epoch in (1, 2, 3):
        entry = make_entry(epoch, 0.15)
        entry["layers"]["dense1"][NEVER_FIRED] = 0.9
        assert detector.check(entry) is None


def test_never_fired_fires_past_the_floor_with_no_margin():
    detector = Detector(make_band_v2(0.1, 0.2), margin=0.5, floor=0.05, patience=3)
    fired = None
    for epoch in (1, 2, 3):
        entry = make_entry(epoch, 0.15)
        entry["layers"]["dense1"][NEVER_FIRED] = 0.28
        fired = detector.check(entry)
    assert fired == {"epoch": 3, "layer": "dense1", "signals": [NEVER_FIRED]}


def test_never_fired_inside_the_floor_stays_quiet():
    detector = Detector(make_band_v2(0.1, 0.2), margin=0.5, floor=0.05, patience=3)
    for epoch in (1, 2, 3):
        entry = make_entry(epoch, 0.15)
        entry["layers"]["dense1"][NEVER_FIRED] = 0.24
        assert detector.check(entry) is None


def test_other_signals_keep_their_margin_beside_never_fired():
    detector = Detector(make_band_v2(0.1, 0.2), margin=0.5, floor=0.05, patience=3)
    for epoch in (1, 2, 3):
        entry = make_entry(epoch, 0.28)
        entry["layers"]["dense1"][NEVER_FIRED] = 0.15
        assert detector.check(entry) is None
