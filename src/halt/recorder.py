"""Records HALT's signals for every watched layer, every epoch."""

import json

import numpy as np

from halt.adapter import NetworkAdapter
from halt.signals import dead_unit_fraction, grad_norm, weight_change, weight_norm


def flatten_parameters(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Join a layer's arrays into one flat vector.

    Arrays are joined in sorted key order, so the same layer always
    produces its numbers in the same order. That matters because
    HALT subtracts the start-of-epoch vector from the end-of-epoch
    vector entry by entry.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        A layer's arrays, such as {"weights": W, "bias": b}.

    Returns
    -------
    np.ndarray
        All entries in one 1D array.
    """
    sorted_keys = sorted(arrays)
    flattened_list = [arrays[k].ravel() for k in sorted_keys]
    return np.concatenate(flattened_list)


class Recorder:
    """Collects HALT's signals for one training run.

    Attributes
    ----------
    adapter : NetworkAdapter
        Answers HALT's questions about the network being watched.
    probe_batch : np.ndarray
        Fixed inputs used every epoch to measure dead units.
    history : list[dict]
        One entry per epoch with every layer's signals.
    """

    def __init__(self, adapter: NetworkAdapter, probe_batch: np.ndarray):
        """Store the adapter and probe batch and set up empty storage."""
        self.adapter = adapter
        self.probe_batch = probe_batch
        self.history = []
        self._before = {}
        self._grad_norms = {}
        self._bias_before = {}

    def start_epoch(self) -> None:
        """Save each layer's starting parameters and clear its gradient list."""
        for name in self.adapter.layer_names():
            params = self.adapter.parameters(name)
            self._before[name] = flatten_parameters(params)
            self._grad_norms[name] = []
            if "bias" in params:
                self._bias_before[name] = params["bias"].copy()

    def after_batch(self) -> None:
        """Record each layer's gradient norm for the batch that just finished.

        Call once per batch, after the backward pass.
        """
        for name in self.adapter.layer_names():
            grads = self.adapter.gradients(name)
            flat_grads = flatten_parameters(grads)
            self._grad_norms[name].append(grad_norm(flat_grads))

    def end_epoch(self, epoch: int, train_accuracy: float, val_accuracy: float, seconds: float) -> dict:
        """Measure every layer's signals and add the epoch to history.

        Per layer: weight_change, weight_norm, grad_norm (the average
        over this epoch's batches), dead_unit_fraction (on the probe
        batch), and bias_change (the size of the bias change alone, not
        divided by anything; logged for analysis only).

        Returns
        -------
        dict
            {"epoch", "train_accuracy", "val_accuracy", "seconds",
                "layers": {layer name: {signal name: value}}}
        """
        probe_outs = self.adapter.probe_outputs(self.probe_batch)
        layer_signals = {}

        for name in self.adapter.layer_names():
            params = self.adapter.parameters(name)
            now = flatten_parameters(params)
            before = self._before[name]
            w_change = weight_change(before, now)
            w_norm = weight_norm(now)
            norms = self._grad_norms[name]
            g_norm = float(np.mean(norms)) if norms else float("nan")
            dead_units = dead_unit_fraction(probe_outs[name], self.adapter.unit_axis(name))
            b_change = None
            if "bias" in params and name in self._bias_before:
                b_change = weight_norm(params["bias"] - self._bias_before[name])
            layer_signals[name] = {
                "weight_change": w_change,
                "weight_norm": w_norm,
                "grad_norm": g_norm,
                "dead_unit_fraction": dead_units,
                "bias_change": b_change
            }
        entry = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "seconds": seconds,
            "layers": layer_signals
        }
        self.history.append(entry)
        return entry

    def save(self, path: str, metadata: dict) -> None:
        """Write the run to a JSON file.

        Parameters
        ----------
        path : str
            Where to write the file.
        metadata : dict
            Facts about the run, such as seed, architecture, and fault label.
        """
        data = {
            "metadata": metadata,
            "history": self.history
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)