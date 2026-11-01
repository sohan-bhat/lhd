"""Runs a saved history through LHD after the fact.

Replay makes the same decisions, in the same order, as a Monitor
watching the run live, so one saved run can be checked against any
healthy band and any setting without training it again.
"""

from __future__ import annotations

from lhd.detector import Detector
from lhd.diagnosis import diagnose, positions


def replay(history: list[dict], band: dict, margin: float, floor: float,
           patience: int) -> dict | None:
    """Feed a run to a fresh Detector one epoch at a time.

    Parameters
    ----------
    history : list[dict]
        Epoch entries, as saved by Recorder.
    band : dict
        Healthy band from healthy_band.
    margin, floor, patience
        Detector settings.

    Returns
    -------
    dict or None
        None if LHD never fires. Otherwise the report Monitor would
        return, {"epoch", "layer", "signals", "cause", "check"}, plus
        "positions", the place of every signal of the flagged layer.
    """
    detector = Detector(band, margin, floor, patience)
    for entry in history:
        fired = detector.check(entry)
        if fired is not None:
            pos = positions(entry, band, fired["layer"], margin, floor)
            return {**fired, **diagnose(pos), "positions": pos}
    return None
