"""The boundary between HALT and any neural network.

HALT never reads a network directly. It asks an adapter, and each
kind of network gets its own adapter that answers in that
network's terms.
"""

from abc import ABC, abstractmethod

import numpy as np


class NetworkAdapter(ABC):
    """Template that every network adapter must fill in.

    HALT only watches layers with trainable parameters. Layers
    without any (pooling, flatten, activations) are skipped.
    """

    @abstractmethod
    def layer_names(self) -> list[str]:
        """Names of the trainable layers HALT should watch, in order."""

    @abstractmethod
    def parameters(self, name: str) -> dict[str, np.ndarray]:
        """The layer's current parameters, keyed by role.

        Use the key "weights" for the main weight array and "bias"
        for the bias. Return the live arrays; HALT makes its own
        copies when it needs them.
        """

    @abstractmethod
    def gradients(self, name: str) -> dict[str, np.ndarray]:
        """The layer's current gradients, with the same keys as parameters."""

    @abstractmethod
    def probe_outputs(self, probe_batch: np.ndarray) -> dict[str, np.ndarray]:
        """Run the probe batch forward and return each watched layer's
        output after its activation, keyed by layer name."""

    @abstractmethod
    def unit_axis(self, name: str) -> int:
        """Which axis of that layer's output indexes its units."""