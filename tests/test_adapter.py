import numpy as np
from halt.adapter import NetworkAdapter


class TinyNet:
    """A two-layer network used only for testing HALT.

    Layout: input (3 numbers) -> dense1 (4 units, ReLU) ->
    dense2 (2 units, no activation).

    Attributes
    ----------
    W1, b1 : np.ndarray
        dense1 weights, shape (3, 4), and bias, shape (4,), zeros.
    W2, b2 : np.ndarray
        dense2 weights, shape (4, 2), and bias, shape (2,), zeros.
    dW1, db1, dW2, db2 : np.ndarray
        Gradients, same shapes as above, starting as zeros.
        Tests fill these in by hand; TinyNet never computes them.
    hidden : np.ndarray or None
        dense1's output after ReLU from the most recent forward pass.
    """

    def __init__(self, seed: int = 0):
        """Create the layers with small random weights from the given seed."""
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((3, 4)) * 0.1
        self.b1 = np.zeros(4)
        self.W2 = rng.standard_normal((4, 2)) * 0.1
        self.b2 = np.zeros(2)
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        self.hidden = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a batch through both layers.

        Parameters
        ----------
        x : np.ndarray
            Batch of inputs, shape (batch_size, 3).

        Returns
        -------
        np.ndarray
            dense2's output, shape (batch_size, 2). Also stores
            dense1's output after ReLU in self.hidden.
        """
        z1 = x @ self.W1 + self.b1
        self.hidden = np.maximum(0, z1)
        out = self.hidden @ self.W2 + self.b2
        return out


class TinyAdapter(NetworkAdapter):
    """Answers HALT's questions about a TinyNet."""
    
    def __init__(self, net):
        self.net = net

    def layer_names(self) -> list[str]:
        """The two layers HALT watches in TinyNet."""
        return ["dense1", "dense2"]

    def parameters(self, name: str) -> dict[str, np.ndarray]:
        """TinyNet's live weights and bias for the named layer."""
        if name == "dense1":
            return {"weights": self.net.W1, "bias": self.net.b1}
        else:
            return {"weights": self.net.W2, "bias": self.net.b2}

    def gradients(self, name: str) -> dict[str, np.ndarray]:
        """TinyNet's live gradients for the named layer."""
        if name == "dense1":
            return {"weights": self.net.dW1, "bias": self.net.db1}
        else:
            return {"weights": self.net.dW2, "bias": self.net.db2}

    def probe_outputs(self, probe_batch: np.ndarray) -> dict[str, np.ndarray]:
        """Each layer's output on the probe batch."""
        out = self.net.forward(probe_batch)
        return {"dense1": self.net.hidden, "dense2": out}

    def unit_axis(self, name: str) -> int:
        """Units are the last axis for both layers."""
        return -1


def test_forward():
    net = TinyNet(0)
    x = np.ones((5, 3))
    out = net.forward(x)

    assert out.shape == (5, 2)
    assert net.hidden is not None
    assert net.hidden.shape == (5, 4)

def test_adapter_layer_names():
    net = TinyNet()
    adapter = TinyAdapter(net)
    assert adapter.layer_names() == ["dense1", "dense2"]

def test_adapter_returns_live_weights():
    net = TinyNet()
    adapter = TinyAdapter(net)
    assert adapter.parameters("dense1")["weights"] is net.W1

def test_adapter_probe_output_shapes():
    net = TinyNet()
    adapter = TinyAdapter(net)
    assert adapter.probe_outputs(np.ones((5, 3)))["dense1"].shape == (5, 4)
    assert adapter.probe_outputs(np.ones((5, 3)))["dense2"].shape == (5, 2)