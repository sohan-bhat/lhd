from halt.detector import out_of_band


def test_out_of_band_frozen_zero_always_flags():
    assert out_of_band(0.0, 0.01, 0.05, 0.5, 0.0) is True


def test_out_of_band_inside_range_is_fine():
    assert out_of_band(0.03, 0.01, 0.05, 0.5, 0.0) is False


def test_out_of_band_floor_allows_stray_dead_unit():
    assert out_of_band(0.01, 0.0, 0.0, 0.5, 0.05) is False


def test_out_of_band_large_dead_fraction_flags():
    assert out_of_band(0.5, 0.0, 0.0, 0.5, 0.05) is True
