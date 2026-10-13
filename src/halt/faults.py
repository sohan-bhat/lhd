"""Switches that plant known faults in a network, for HALT's experiments.

Every fault works through the adapter, so it works on any network.
Faults change the network's arrays in place.
"""

import numpy as np

from halt.adapter import NetworkAdapter


class FrozenLayer:
    """Keeps one layer's parameters from ever changing.

    Copies the "layer left out of the optimizer" bug: gradients are
    still computed, but the weights never move. Call restore() right
    after every optimizer step.
    """

    def __init__(self, adapter: NetworkAdapter, name: str):
        """Save copies of the layer's starting parameters."""
        self.adapter = adapter
        self.name = name
        params = adapter.parameters(name)
        self._saved = {}
        for key, arr in params.items():
            self._saved[key] = arr.copy()

    def restore(self) -> None:
        """Write the saved values back into the layer's live arrays, in place."""
        live = self.adapter.parameters(self.name)
        for key, arr in live.items():
            arr[...] = self._saved[key]


def kill_units(adapter: NetworkAdapter, name: str, units: list[int],
               weight_axis: int = -1, bias_value: float = -10.0) -> None:
    """Make chosen units output zero for every input.

    Sets each chosen unit's incoming weights to 0 and its bias to a
    large negative number, so ReLU always outputs 0. ReLU's slope is 0
    there, so the unit gets no gradient and stays dead.

    Parameters
    ----------
    adapter : NetworkAdapter
        Gives access to the layer's live arrays.
    name : str
        Which layer.
    units : list[int]
        Which units to kill.
    weight_axis : int, default -1
        Which axis of the weight array indexes the units.
    bias_value : float, default -10.0
        The bias given to killed units.
    """
    params = adapter.parameters(name)
    weights = params["weights"]
    np.moveaxis(weights, weight_axis, -1)[..., units] = 0.0
    params["bias"][..., units] = bias_value