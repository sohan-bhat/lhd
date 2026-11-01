"""Lets LHD watch a PyTorch model.

The model must live on the CPU. LHD reads and writes parameters
through numpy views that share memory with the tensors, so faults
planted through this adapter change the real model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lhd.adapter import NetworkAdapter


def _numpy_view(tensor: torch.Tensor) -> np.ndarray:
    """A numpy array that shares memory with the tensor."""
    return tensor.detach().numpy()


def _grad_array(param: nn.Parameter) -> np.ndarray:
    """The parameter's gradient, or zeros if backward has not reached it."""
    if param.grad is None:
        return np.zeros(tuple(param.shape), dtype=np.float32)
    return _numpy_view(param.grad)


class TorchAdapter(NetworkAdapter):
    """Answers LHD's questions about a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The whole network. Used to run the probe batch.
    layers : dict[str, tuple[nn.Module, nn.Module, int]]
        Watched layer name -> (module that holds the weight and bias,
        module whose output is the layer's output after its
        activation, unit axis of that output). For a layer with no
        activation, the second module is the layer itself.
    """

    def __init__(self, model: nn.Module, layers: dict):
        self.model = model
        self.layers = layers

    def layer_names(self) -> list[str]:
        """Watched layers, in the order they were given."""
        return list(self.layers)

    def _module(self, name: str) -> nn.Module:
        """The module that holds the named layer's parameters."""
        return self.layers[name][0]

    def parameters(self, name: str) -> dict[str, np.ndarray]:
        """Live numpy views of the layer's weight and bias."""
        module = self._module(name)
        result = {"weights": _numpy_view(module.weight)}
        if getattr(module, "bias", None) is not None:
            result["bias"] = _numpy_view(module.bias)
        return result

    def gradients(self, name: str) -> dict[str, np.ndarray]:
        """The layer's gradients, with the same keys as parameters."""
        module = self._module(name)
        result = {"weights": _grad_array(module.weight)}
        if getattr(module, "bias", None) is not None:
            result["bias"] = _grad_array(module.bias)
        return result

    def probe_outputs(self, probe_batch: np.ndarray) -> dict[str, np.ndarray]:
        """Run the probe batch in eval mode and capture each layer's output.

        The model is put back in its previous mode afterwards, and no
        gradients are computed.
        """
        captured = {}
        handles = []
        for name, (_, output_module, _) in self.layers.items():
            handles.append(output_module.register_forward_hook(_capture_into(captured, name)))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                self.model(torch.as_tensor(probe_batch))
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        return captured

    def unit_axis(self, name: str) -> int:
        """Which axis of the layer's output indexes its units."""
        return self.layers[name][2]


def _capture_into(store: dict, name: str):
    """A forward hook that saves the module's output under name."""
    def hook(module, inputs, output):
        store[name] = output.detach().numpy().copy()
    return hook
