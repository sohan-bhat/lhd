import numpy as np
import pytest

from halt.signals import dead_unit_fraction, grad_norm, weight_change, weight_norm


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


def test_grad_norm_matches_3_4_5_triangle():
    assert grad_norm(np.array([[3.0, 4.0]])) == pytest.approx(5.0)


def test_dead_unit_fraction_all_zeros():
    activations = np.zeros((4, 3))
    assert dead_unit_fraction(activations, unit_axis=1) == 1.0


def test_dead_unit_fraction_all_ones():
    activations = np.ones((4, 3))
    assert dead_unit_fraction(activations, unit_axis=-1) == 0.0


def test_dead_unit_fraction_one_zero_column_out_of_three():
    activations = np.array([
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0]
    ])
    assert dead_unit_fraction(activations, unit_axis=-1) == pytest.approx(1/3)


def test_dead_unit_fraction_partial_zero_is_not_dead():
    activations = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 5.0, 4.0]
    ])
    assert dead_unit_fraction(activations, unit_axis=-1) == 0.0


def test_dead_unit_fraction_4d_array():
    activations = np.ones((2, 3, 2, 2))
    activations[:, 1, :, :] = 0.0
    assert dead_unit_fraction(activations, unit_axis=1) == pytest.approx(1/3)