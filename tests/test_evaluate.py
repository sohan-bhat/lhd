import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.evaluate import (  # noqa: E402
    curve_alarm,
    curve_band,
    family,
    paired_tests,
    rate,
    score_arch,
    wilson,
)
from experiments.grid import build_grid, cut_affects, dead_layers, watched_layers  # noqa: E402
from halt.baseline import SIGNALS  # noqa: E402


def curve(values, key="val_accuracy"):
    return [{"epoch": i + 1, key: v} for i, v in enumerate(values)]


def test_curve_band_low_high():
    band = curve_band([curve([0.5, 0.6]), curve([0.4, 0.7])], "val_accuracy")
    assert np.allclose(band["low"], [0.4, 0.6])
    assert np.allclose(band["high"], [0.5, 0.7])


def test_curve_alarm_needs_patience_in_a_row():
    band = {"low": np.full(5, 0.5), "high": np.full(5, 1.0)}
    history = curve([0.4, 0.4, 0.6, 0.4, 0.4])
    assert curve_alarm(history, band, "val_accuracy", "below", 2) == 2
    assert curve_alarm(history, band, "val_accuracy", "below", 3) is None


def test_curve_alarm_loss_above_and_nan():
    band = {"low": np.zeros(3), "high": np.ones(3)}
    assert curve_alarm(curve([0.5, 2.0, 2.0], "val_loss"), band, "val_loss", "above", 2) == 3
    assert curve_alarm(curve([float("nan")] * 3, "val_loss"), band, "val_loss", "above", 1) == 1


def test_family_names():
    assert family({"type": "frozen", "layer": "fc1", "onset": 1}) == "frozen"
    assert family({"type": "frozen", "layer": "fc1", "onset": 6}) == "latefrozen"
    assert family({"type": "dead", "layer": "fc1", "fraction": 0.25, "onset": 1}) == "dead25"
    assert family({"type": "dead", "layer": "fc1", "fraction": 0.5, "onset": 6}) == "latedead50"
    assert family({"type": "lr", "factor": 10, "onset": 6}) == "lr10"


def test_wilson_interval_contains_rate():
    low, high = wilson(8, 10)
    assert low < 0.8 < high
    assert wilson(0, 30)[0] == 0.0
    assert wilson(0, 30)[1] == pytest.approx(0.1135, abs=1e-3)


def test_rate_skips_none():
    rows = [{"x": True}, {"x": False}, {"x": None}]
    assert rate(rows, "x")["n"] == 2
    assert rate(rows, "x")["rate"] == 0.5


def test_paired_tests_count_direction():
    rows = [{"role": "fault", "gap": g} for g in (3, 5, 0, -1, 7)]
    result = paired_tests(rows)
    assert (result["halt_earlier"], result["acc_earlier"], result["ties"]) == (3, 1, 1)


def test_grid_ids_unique_and_seeds_split():
    ids = []
    for phase in ("core", "more", "seeds"):
        ids += [s["run_id"] for s in build_grid(phase)]
    assert len(set(ids)) == len(ids)
    core = build_grid("core")
    calibration = {s["seed"] for s in core if s["role"] == "calibration"}
    evaluation = {s["seed"] for s in core if s["role"] != "calibration"}
    assert not calibration & evaluation


def test_cut_affects_upstream_layers():
    assert cut_affects("cnn", "conv3") == ["conv1", "conv2"]
    assert cut_affects("mlp", "fc4") == ["fc1", "fc2", "fc3"]
    assert cut_affects("transformer", "blocks.1") == [
        "tok", "pos", "b0.qkv", "b0.proj", "b0.ff1", "b0.ff2"]
    assert "head" not in cut_affects("transformer", "head")


def test_dead_layers_are_relu_layers():
    assert dead_layers("cnn") == ["conv1", "conv2", "conv3", "fc1"]
    assert dead_layers("transformer") == ["b0.ff1", "b1.ff1"]
    assert watched_layers("mlp") == ["fc1", "fc2", "fc3", "fc4"]


def fake_run(role, seed, fault, weight_change, acc):
    layers = {"fc1": {"weight_change": weight_change, "weight_norm": 1.0,
                      "grad_norm": 1.0, "dead_unit_fraction": 0.0}}
    history = [{"epoch": e, "val_accuracy": acc, "val_loss": 1.0, "layers": layers}
               for e in range(1, 7)]
    meta = {"arch": "mlp", "run_id": f"{role}-{seed}", "role": role, "seed": seed,
            "fault": fault, "halt_seconds": 1.0, "total_seconds": 10.0}
    return {"metadata": meta, "history": history}


def test_score_arch_end_to_end():
    healthy = {"type": "none"}
    runs = [fake_run("calibration", s, healthy, 0.1 + 0.01 * s, 0.9) for s in range(3)]
    runs.append(fake_run("fresh", 1000, healthy, 0.11, 0.9))
    runs.append(fake_run("fault", 1000, {"type": "frozen", "layer": "fc1", "onset": 1}, 0.0, 0.9))
    rows = {r["role"]: r for r in score_arch("mlp", runs=runs)}
    assert rows["fresh"]["fired"] is False
    fault = rows["fault"]
    assert fault["detected"] and fault["layer_ok"] and fault["cause_ok"]
    assert fault["halt_epoch"] == 3
    assert fault["acc_epoch"] is None
    assert fault["gap"] == 7 - 3
    assert fault["compute_saved"] == pytest.approx(0.5)
    assert set(SIGNALS) <= set(runs[0]["history"][0]["layers"]["fc1"])


def test_clustered_tests_average_within_groups():
    from experiments.evaluate import clustered_tests

    rows = [{"role": "fault", "arch": "mlp", "seed": s, "family": "frozen", "fault_layer": layer,
             "gap": g} for s, layer, g in ((1, "fc1", 4), (1, "fc2", 6), (2, "fc1", 2), (2, "fc2", -2))]
    result = clustered_tests(rows, draws=200)
    seed = result["by_seed"]
    assert (seed["groups"], seed["halt_earlier"], seed["acc_earlier"], seed["ties"]) == (2, 1, 0, 1)
    assert result["by_condition"]["groups"] == 2
    assert result["by_condition"]["halt_earlier"] == 2
    low, high = result["mean_gap_ci"]
    assert low <= 2.5 <= high
