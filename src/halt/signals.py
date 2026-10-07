"""Per-layer health signals for neural network training.

Each function is pure: arrays in, one number out. Nothing here
touches a network object, so every signal can be tested alone.
"""
import numpy as np


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
    return float(np.sqrt(np.sum(np.square(weights))))
    