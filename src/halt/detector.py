"""Decides when a layer's signals have left the healthy range for too long."""
from __future__ import annotations

import math
from typing import Optional

from halt.baseline import SIGNALS


def band_position(value: float, low: float, high: float,
                  margin: float, floor: float) -> str:
    """Where a value sits relative to the widened healthy range.

    Uses the same widening as out_of_band.

    Returns
    -------
    str
        "below", "inside", or "above", or "invalid" if the value is
        nan or infinite (training blew up).
    """
    if not math.isfinite(value):
        return "invalid"
    bottom = low * (1 - margin)
    top = high * (1 + margin) + floor
    if bottom <= value <= top:
        return "inside"
    return "below" if value < bottom else "above"


def out_of_band(value: float, low: float, high: float,
                margin: float, floor: float) -> bool:
    """Whether one signal value falls outside the widened healthy range.

    The range is widened so ordinary noise does not trigger it: the
    bottom becomes low * (1 - margin) and the top becomes
    high * (1 + margin) + floor. Since margin is below 1, a healthy
    bottom above 0 stays above 0, so a value of exactly 0 (a frozen
    layer) is always outside the range.

    Parameters
    ----------
    value : float
        The signal measured this epoch.
    low, high : float
        The healthy range for this epoch.
    margin : float
        How far to widen the range, as a fraction between 0 and 1.
    floor : float
        Extra room added to the top, so a healthy range of exactly 0
        does not flag one stray dead unit.

    Returns
    -------
    bool
        True if the value is outside the widened range.
    """
    return band_position(value, low, high, margin, floor) != "inside"


class Detector:
    """Watches a run epoch by epoch and fires when a layer stays abnormal.

    A layer fires when any one of its signals is out of band for
    `patience` epochs in a row. The detector fires once; later epochs
    do not change the result.

    Attributes
    ----------
    fired : dict or None
        None until the detector fires, then
        {"epoch": int, "layer": str, "signals": list[str]}.
    """

    def __init__(self, band: dict, margin: float, floor: float, patience: int):
        """Store the healthy band and settings, and start every streak at 0."""
        self.band = band
        self.margin = margin
        self.floor = floor
        self.patience = patience
        self._streaks = {}
        self.fired = None

    def check(self, entry: dict) -> dict | None:
        """Update streaks with one epoch's signals and fire if a streak
        reaches patience.

        Parameters
        ----------
        entry : dict
            One epoch, as returned by Recorder.end_epoch.

        Returns
        -------
        dict or None
            self.fired.
        """
        if self.fired is not None:
            return self.fired

        i = entry["epoch"] - 1
        for layer, layer_band in self.band.items():
            for signal in SIGNALS:
                value = entry["layers"][layer][signal]
                low = layer_band[signal]["low"][i]
                high = layer_band[signal]["high"][i]
                key = (layer, signal)
                if out_of_band(value, low, high, self.margin, self.floor):
                    self._streaks[key] = self._streaks.get(key, 0) + 1
                else:
                    self._streaks[key] = 0

            reached = [s for s in SIGNALS if self._streaks[(layer, s)] >= self.patience]
            if reached:
                self.fired = {"epoch": entry["epoch"], "layer": layer, "signals": reached}
                return self.fired
        return self.fired
