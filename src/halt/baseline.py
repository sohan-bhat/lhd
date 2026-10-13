"""Healthy ranges for HALT's signals, built only from healthy runs."""

import json

import numpy as np

SIGNALS = ("weight_change", "weight_norm", "grad_norm", "dead_unit_fraction")


def load_history(path: str) -> list[dict]:
    """Read one saved run and return its history list."""
    with open(path) as f:
        data = json.load(f)
    return data["history"]


def healthy_band(histories: list[list[dict]]) -> dict:
    """Lowest and highest value of each signal, per layer, per epoch.

    Built only from healthy runs. Broken runs must never be passed in.

    Parameters
    ----------
    histories : list[list[dict]]
        One history per healthy run. All runs must have the same
        number of epochs and the same layers.

    Returns
    -------
    dict
        band[layer][signal] = {"low": array, "high": array}, where
        each array has one value per epoch.

    Raises
    ------
    ValueError
        If the runs have different numbers of epochs.
    """
    lengths = {len(history) for history in histories}
    if len(lengths) > 1:
        raise ValueError(f"runs have different numbers of epochs: {sorted(lengths)}")

    layer_names = histories[0][0]["layers"]
    band = {}
    for layer in layer_names:
        band[layer] = {}
        for signal in SIGNALS:
            values = np.array([
                [entry["layers"][layer][signal] for entry in history]
                for history in histories
            ])
            band[layer][signal] = {
                "low": values.min(axis=0),
                "high": values.max(axis=0)
            }
    return band
