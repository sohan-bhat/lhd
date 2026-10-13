"""Turns the pattern across a layer's signals into a likely cause."""

from halt.baseline import SIGNALS
from halt.detector import band_position


def positions(entry: dict, band: dict, layer: str,
              margin: float, floor: float) -> dict[str, str]:
    """Where each of a layer's signals sits this epoch.

    Returns
    -------
    dict[str, str]
        signal name -> "below", "inside", "above", or "invalid".
    """
    i = entry["epoch"] - 1
    result = {}
    for signal in SIGNALS:
        value = entry["layers"][layer][signal]
        low = band[layer][signal]["low"][i]
        high = band[layer][signal]["high"][i]
        result[signal] = band_position(value, low, high, margin, floor)
    return result
