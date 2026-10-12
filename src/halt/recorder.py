"""Records HALT's signals for every watched layer, every epoch."""

import numpy as np

from halt.adapter import NetworkAdapter
from halt.signals import grad_norm


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

    def start_epoch(self) -> None:
        """Save each layer's starting parameters and clear its gradient list."""
        for name in self.adapter.layer_names():
            self._before[name] = flatten_parameters(self.adapter.parameters(name))
            self._grad_norms[name] = []

    def after_batch(self) -> None:
        """Record each layer's gradient norm for the batch that just finished.

        Call once per batch, after the backward pass.
        """
        for name in self.adapter.layer_names():
            grads = self.adapter.gradients(name)
            flat_grads = flatten_parameters(grads)
            self._grad_norms[name].append(grad_norm(flat_grads))