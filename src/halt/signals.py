"""Per-layer health signals for neural network training.

Each function is pure: arrays in, one number out. Nothing here
touches a network object, so every signal can be tested alone.
"""
import numpy as np

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
    delta = np.subtract(weights_after, weights_before)
    delta_norm = np.linalg.norm(delta)
    before_norm = np.linalg.norm(weights_before)
    return float(delta_norm / (before_norm + 1e-12))

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