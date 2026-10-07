import numpy as np
import pytest

from halt.signals import weight_norm, weight_change


def test_weight_norm_matches_3_4_5_triangle():
    assert weight_norm(np.array([[3.0, 4.0]])) == pytest.approx(5.0)

def test_norm_4d_array_of_ones():
    assert weight_norm(np.ones((2, 2, 2, 2))) == pytest.approx(4.0)

def test_weight_change_identical():
    W = np.array([0.1, 0.5, 0.4])
    result = weight_change(W, W.copy())
    assert result == 0.0

def test_weight_change_doubled():
    W = np.array([0.1, 0.5, 0.4])
    result = weight_change(W, 2 * W)
    assert result == pytest.approx(1.0)

def test_weight_change_all_zero_before():
    W_zero = np.zeros(3)
    W_after = np.array([0.1, 0.5, 0.4])
    result = weight_change(W_zero, W_after)
    assert np.isfinite(result)