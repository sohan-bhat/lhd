import numpy as np
import pytest
from tiny_net import TinyAdapter, TinyNet


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
    outs = adapter.probe_outputs(np.ones((5, 3)))
    assert outs["dense1"].shape == (5, 4)
    assert outs["dense2"].shape == (5, 2)


def test_adapter_unknown_layer_fails_loudly():
    adapter = TinyAdapter(TinyNet())
    with pytest.raises(KeyError):
        adapter.parameters("dense3")