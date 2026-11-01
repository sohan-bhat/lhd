"""Per-layer health signals for neural network training.

Each function is pure: arrays in, one number out. Nothing here
touches a network object, so every signal can be tested alone.
"""

import numpy as np

_EPSILON = 1e-12


def _norm(array: np.ndarray) -> float:
    """Square root of the sum of squared entries, for any shape."""
    return float(np.sqrt(np.sum(np.square(array))))


def weight_norm(weights: np.ndarray) -> float:
    """Overall size of a layer's weights (Frobenius norm).
    
    Parameters
    ----------
    weights : np.ndarray
        The layer's weight array, any shape.

    Returns
    -------
    float
        Square root of the sum of squared entries.
    """
    return _norm(weights)


def weight_change(weights_before: np.ndarray, weights_after: np.ndarray) -> float:
    """How much a layer's weights moved, relative to their size.

    A value of 0 means the weights did not change at all (a frozen
    layer). A value of 1 means the change was as large as the
    weights themselves.

    Pass the layer's weights and bias together as one flattened
    array, so the size being divided by is never close to zero.

    Parameters
    ----------
    weights_before : np.ndarray
        The weights at the start of the epoch.
    weights_after : np.ndarray
        The weights at the end of the epoch. Same shape.

    Returns
    -------
    float
        Size of the change divided by the size of weights_before.
    """
    delta = weights_after - weights_before
    delta_norm = _norm(delta)
    before_norm = _norm(weights_before)
    return delta_norm / (before_norm + _EPSILON)


def grad_norm(grad: np.ndarray) -> float:
    """Overall size of a layer's gradient for one batch.

    A value near 0 means the layer is receiving almost no
    learning signal.

    Parameters
    ----------
    grad : np.ndarray
        The gradient of the loss with respect to the layer's
        weights, any shape.

    Returns
    -------
    float
        Square root of the sum of squared entries.
    """
    return _norm(grad)


def dead_unit_fraction(activations: np.ndarray, unit_axis: int = -1) -> float:
    """Fraction of a layer's units that output zero for every input.

    A unit is one neuron in a dense layer or one filter in a conv
    layer. It counts as dead only if its output is exactly 0 for
    every input in the batch (and every position, for a conv layer).

    Parameters
    ----------
    activations : np.ndarray
        The layer's outputs after ReLU, for the fixed probe batch.
    unit_axis : int, default -1
        Which axis of the array indexes the units.

    Returns
    -------
    float
        Number of dead units divided by total units, from 0 to 1.
    """
    flat = activations.swapaxes(unit_axis, -1)
    flat = flat.reshape(-1, activations.shape[unit_axis])
    dead_units = np.all(flat == 0, axis=0)
    return float(np.mean(dead_units))