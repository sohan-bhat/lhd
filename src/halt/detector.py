"""Decides when a layer's signals have left the healthy range for too long."""


def out_of_band(value: float, low: float, high: float,
                margin: float, floor: float) -> bool:
    """Whether one signal value falls outside the widened healthy range.

    The range is widened so ordinary noise does not trigger it: the
    bottom becomes low * (1 - margin) and the top becomes
    high * (1 + margin) + floor. Since margin is below 1, a healthy
    bottom above 0 stays above 0, so a value of exactly 0 (a frozen
    layer) is always outside the range.

    Parameters
    ----------
    value : float
        The signal measured this epoch.
    low, high : float
        The healthy range for this epoch.
    margin : float
        How far to widen the range, as a fraction between 0 and 1.
    floor : float
        Extra room added to the top, so a healthy range of exactly 0
        does not flag one stray dead unit.

    Returns
    -------
    bool
        True if the value is outside the widened range.
    """
    bottom = low * (1 - margin)
    top = high * (1 + margin) + floor
    return bool(value < bottom or value > top)
