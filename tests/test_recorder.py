import numpy as np

from halt.recorder import flatten_parameters


def test_flattened_params_shape():
    arrays = {"weights": np.ones((2, 3)), "bias": np.zeros(3)}
    assert flatten_parameters(arrays).shape == (9,)


def test_flattened_params_order():
    arrays = {"weights": np.array([1.0]), "bias": np.array([2.0])}
    assert (flatten_parameters(arrays) == [2.0, 1.0]).all()