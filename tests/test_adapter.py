import numpy as np


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

def test_forward():
    net = TinyNet(0)
    x = np.ones((5, 3))
    out = net.forward(x)

    assert out.shape == (5, 2)
    assert net.hidden is not None
    assert net.hidden.shape == (5, 4)