"""Turns the pattern across a layer's signals into a likely cause."""

from lhd.baseline import NEVER_FIRED
from lhd.detector import band_position, signal_margin

CHECKS = {
    "diverged": "A signal became nan or infinite, so training blew up. "
                "Check the learning rate.",
    "dead_units": "Units output zero for every input. Check the learning "
                  "rate, weight initialization, and bias values.",
    "frozen_layer": "Gradients are computed but never applied. Check that "
                    "this layer's parameters are passed to the optimizer.",
    "vanishing_gradient": "The layer receives almost no learning signal. "
                          "Check the layers after it and the initialization.",
    "runaway_weights": "The weights are growing far past the healthy range. "
                       "Check the learning rate.",
    "unrecognized": "The signals left the healthy range in a pattern LHD "
                    "has no rule for. Inspect this layer by hand.",
}


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
    for signal in band[layer]:
        value = entry["layers"][layer][signal]
        low = band[layer][signal]["low"][i]
        high = band[layer][signal]["high"][i]
        result[signal] = band_position(value, low, high, signal_margin(signal, margin), floor)
    return result


def diagnose(pos: dict[str, str]) -> dict:
    """Name the likely cause from the pattern of positions.

    Rules, checked in this order:
    1. any signal invalid -> diverged
    2. dead_unit_fraction or never_fired_fraction above -> dead_units
    3. weight_change below, grad_norm inside or above -> frozen_layer
    4. weight_change below, grad_norm below -> vanishing_gradient
    5. weight_norm above -> runaway_weights
    6. anything else -> unrecognized

    Returns
    -------
    dict
        {"cause": str, "check": str}, where check comes from CHECKS.
    """
    if "invalid" in pos.values():
        cause = "diverged"
    elif pos["dead_unit_fraction"] == "above" or pos.get(NEVER_FIRED) == "above":
        cause = "dead_units"
    elif pos["weight_change"] == "below" and pos["grad_norm"] in ("inside", "above"):
        cause = "frozen_layer"
    elif pos["weight_change"] == "below" and pos["grad_norm"] == "below":
        cause = "vanishing_gradient"
    elif pos["weight_norm"] == "above":
        cause = "runaway_weights"
    else:
        cause = "unrecognized"
    return {"cause": cause, "check": CHECKS[cause]}