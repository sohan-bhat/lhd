"""HALT's front door: watches a run and says when to stop."""

from typing import Optional

import numpy as np

from halt.adapter import NetworkAdapter
from halt.detector import Detector
from halt.diagnosis import diagnose, positions
from halt.recorder import Recorder


class Monitor:
    """Combines the recorder, detector, and diagnosis.

    The training loop calls start_epoch, after_batch, and end_epoch,
    and stops the run if end_epoch returns a report.
    """

    def __init__(self, adapter: NetworkAdapter, probe_batch: np.ndarray,
                 band: dict, margin: float, floor: float, patience: int):
        """Create a Recorder and a Detector, and keep margin and floor
        for diagnosis."""
        self.recorder = Recorder(adapter, probe_batch)
        self.detector = Detector(band, margin, floor, patience)
        self.band = band
        self.margin = margin
        self.floor = floor

    def start_epoch(self) -> None:
        """Pass through to the recorder."""
        self.recorder.start_epoch()

    def after_batch(self) -> None:
        """Pass through to the recorder."""
        self.recorder.after_batch()

    def end_epoch(self, epoch: int, train_accuracy: float,
                  val_accuracy: float, seconds: float) -> Optional[dict]:
        """Record the epoch, check it, and return a report if HALT fires.

        Returns
        -------
        dict or None
            None while the run looks healthy. Otherwise
            {"epoch", "layer", "signals", "cause", "check"}.
        """
        entry = self.recorder.end_epoch(epoch, train_accuracy, val_accuracy, seconds)
        fired = self.detector.check(entry)
        if fired is None:
            return None
        pos = positions(entry, self.band, fired["layer"], self.margin, self.floor)
        return {**fired, **diagnose(pos)}
