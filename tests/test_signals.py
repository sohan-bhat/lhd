import numpy as np
import pytest

from halt.signals import weight_norm


def test_weight_norm_matches_3_4_5_triangle():
    assert weight_norm(np.array([[3.0, 4.0]])) == pytest.approx(5.0)

def test_norm_4d_array_of_ones():
    assert weight_norm(np.ones((2, 2, 2, 2))) == pytest.approx(4.0)