"""Healthy ranges for HALT's signals, built only from healthy runs."""

import json

import numpy as np

SIGNALS = ("weight_change", "weight_norm", "grad_norm", "dead_unit_fraction")


def load_history(path: str) -> list[dict]:
    """Read one saved run and return its history list."""
    with open(path) as f:
        data = json.load(f)
    return data["history"]
