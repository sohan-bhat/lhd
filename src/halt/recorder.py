"""Records HALT's signals for every watched layer, every epoch."""

import numpy as np


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