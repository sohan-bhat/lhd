import numpy as np
import pytest
from tiny_net import TinyAdapter, TinyNet

from lhd.baseline import SIGNALS, SIGNALS_V2, healthy_band, load_history
from lhd.recorder import Recorder


def make_history(weight_changes):
    """A fake healthy history where only dense1's weight_change varies."""
    return [
        {"epoch": i + 1,
         "layers": {"dense1": {"weight_change": wc, "weight_norm": 1.0,
                               "grad_norm": 1.0, "dead_unit_fraction": 0.0,
                               "bias_change": 0.0}}}
        for i, wc in enumerate(weight_changes)
    ]


def test_load_history_reads_saved_run(tmp_path):
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    path = tmp_path / "run.json"
    recorder.save(str(path), metadata={"seed": 0})
    assert len(load_history(str(path))) == 1


def test_healthy_band_low_high_per_epoch():
    band = healthy_band([make_history([0.10, 0.05]), make_history([0.20, 0.03])])
    assert np.allclose(band["dense1"]["weight_change"]["low"], [0.10, 0.03])
    assert np.allclose(band["dense1"]["weight_change"]["high"], [0.20, 0.05])


def test_healthy_band_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        healthy_band([make_history([0.1, 0.2]), make_history([0.1, 0.2, 0.3])])


def test_healthy_band_ignores_bias_change():
    band = healthy_band([make_history([0.10, 0.05])])
    assert "bias_change" not in band["dense1"]


def test_healthy_band_has_four_signals_unless_asked():
    histories = [[{"layers": {"dense1": {s: float(i) for s in SIGNALS_V2}}}] for i in range(3)]
    assert tuple(healthy_band(histories)["dense1"]) == SIGNALS
    band = healthy_band(histories, SIGNALS_V2)
    assert tuple(band["dense1"]) == SIGNALS_V2
    assert band["dense1"]["never_fired_fraction"]["high"].tolist() == [2.0]
