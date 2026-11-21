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
from lhd.baseline import SIGNALS  # noqa: E402


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
    assert (result["lhd_earlier"], result["acc_earlier"], result["ties"]) == (3, 1, 1)


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
            "fault": fault, "lhd_seconds": 1.0, "total_seconds": 10.0}
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
    assert fault["lhd_epoch"] == 3
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
    assert (seed["groups"], seed["lhd_earlier"], seed["acc_earlier"], seed["ties"]) == (2, 1, 0, 1)
    assert result["by_condition"]["groups"] == 2
    assert result["by_condition"]["lhd_earlier"] == 2
    low, high = result["mean_gap_ci"]
    assert low <= 2.5 <= high


def layer_history(values, signal):
    return [{"epoch": i + 1, "layers": {"fc1": {"weight_change": 0.1, "dead_unit_fraction": 0.0, signal: v}}}
            for i, v in enumerate(values)]


def test_zero_update_check_fires_on_first_exact_zero():
    from experiments.evaluate import zero_update_epoch

    assert zero_update_epoch(layer_history([0.2, 1e-9, 0.0, 0.0], "weight_change")) == 3
    assert zero_update_epoch(layer_history([0.2, 0.1], "weight_change")) is None


def test_dead_check_uses_threshold_inclusively():
    from experiments.evaluate import dead_check_epoch

    history = layer_history([0.1, 0.5, 0.9], "dead_unit_fraction")
    assert dead_check_epoch(history, 0.5) == 2
    assert dead_check_epoch(history, 0.95) is None
    assert dead_check_epoch(history, 0.5, layers={"fc2"}) is None


def test_harm_bins_cover_every_drop():
    from experiments.evaluate import harm_bin

    assert harm_bin(-0.01) == "under 0.5 pts"
    assert harm_bin(0.005) == "0.5 to 2 pts"
    assert harm_bin(0.02) == "2 pts or more"
    assert harm_bin(None) is None


def test_score_arch_scores_simple_checks():
    healthy = {"type": "none"}
    runs = [fake_run("calibration", s, healthy, 0.1 + 0.01 * s, 0.9) for s in range(3)]
    runs.append(fake_run("fresh", 1000, healthy, 0.11, 0.9))
    runs.append(fake_run("fault", 1000, {"type": "frozen", "layer": "fc1", "onset": 1}, 0.0, 0.9))
    rows = {r["role"]: r for r in score_arch("mlp", runs=runs)}
    assert rows["fresh"]["zero_update_epoch"] is None
    fault = rows["fault"]
    assert fault["zero_update_epoch"] == 1 and fault["simple_detected"]
    assert fault["simple_gap"] == 1 - 3
    assert fault["harm"] == "under 0.5 pts"
    assert fault["zero_update_layer"] == "fc1"
    assert (fault["zero_update_p3_epoch"], fault["zero_update_p3_layer"]) == (3, "fc1")
    assert fault["dead50_check_p3_epoch"] is None and fault["dead50_check_p3_layer"] is None


def two_layer_history(fc1, fc2, signal):
    return [{"epoch": i + 1, "layers": {
        "fc1": {"weight_change": 0.1, "dead_unit_fraction": 0.0, signal: a},
        "fc2": {"weight_change": 0.1, "dead_unit_fraction": 0.0, signal: b}}}
        for i, (a, b) in enumerate(zip(fc1, fc2))]


def test_fixed_checks_need_patience_in_a_row():
    from experiments.evaluate import dead_check, zero_update_check

    two = layer_history([0.1, 0.0, 0.0, 0.1, 0.1], "weight_change")
    assert zero_update_check(two, 1) == (2, "fc1")
    assert zero_update_check(two, 3) == (None, None)
    three = layer_history([0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], "weight_change")
    assert zero_update_check(three, 3) == (7, "fc1")
    dead = layer_history([0.6, 0.6, 0.2, 0.6, 0.6, 0.6], "dead_unit_fraction")
    assert dead_check(dead, 0.5, patience=1) == (1, "fc1")
    assert dead_check(dead, 0.5, patience=3) == (6, "fc1")


def test_fixed_checks_name_the_layer():
    from experiments.evaluate import dead_check, fixed_checks, zero_update_check

    later = two_layer_history([0.1, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.1], "weight_change")
    assert zero_update_check(later, 1) == (1, "fc2")
    assert zero_update_check(later, 3) == (3, "fc2")
    tie = two_layer_history([0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], "weight_change")
    assert zero_update_check(tie, 3) == (4, "fc1")
    dead = two_layer_history([0.9, 0.9, 0.9], [0.6, 0.6, 0.6], "dead_unit_fraction")
    assert dead_check(dead, 0.5, patience=3) == (3, "fc1")
    assert dead_check(dead, 0.5, layers={"fc2"}, patience=3) == (3, "fc2")
    fields = fixed_checks(dead, {"fc2"})
    assert (fields["dead50_check_epoch"], fields["dead50_check_layer"]) == (1, "fc2")
    assert (fields["dead50_check_p3_epoch"], fields["dead50_check_p3_layer"]) == (3, "fc2")
    assert fields["dead70_check_p3_epoch"] is None and fields["zero_update_p3_epoch"] is None
    assert {"deadany_check_layer", "deadany_check_p3_layer", "dead80_check_p3_epoch"} <= set(fields)


def detector_row(role, family, kind, layer, onset=1, **fields):
    from experiments.evaluate import DEAD_CHECKS, FIXED_PATIENCES

    row = {"arch": "mlp", "role": role, "family": family, "fault_type": kind, "fault_layer": layer,
           "onset": onset, "epochs": 10, "lhd_epoch": None, "lhd_layer": None, "detected": False,
           "layer_ok": None, "acc_epoch": None, "acc1_epoch": None, "loss_epoch": None}
    names = ["zero_update", "deadany_check"] + [f"dead{int(t * 100)}_check" for t in DEAD_CHECKS]
    for name in names:
        for suffix in FIXED_PATIENCES:
            row[f"{name}{suffix}_epoch"] = row[f"{name}{suffix}_layer"] = None
    row.update(fields)
    return row


def test_detector_table_counts_right_layer():
    from experiments.analyze import detector_table
    from experiments.evaluate import wilson

    rows = [
        detector_row("fault", "frozen", "frozen", "fc2", lhd_epoch=3, detected=True, layer_ok=True,
                     acc_epoch=6, zero_update_epoch=2, zero_update_layer="fc2",
                     zero_update_p3_epoch=4, zero_update_p3_layer="fc2",
                     dead50_check_epoch=1, dead50_check_layer="fc1"),
        detector_row("fault", "dead25", "dead", "fc1", lhd_epoch=4, detected=True, layer_ok=False,
                     dead25_check_epoch=1, dead25_check_layer="fc3",
                     dead25_check_p3_epoch=3, dead25_check_p3_layer="fc1",
                     dead50_check_epoch=5, dead50_check_layer="fc1"),
        detector_row("fault", "lr10", "lr", None, onset=6, lhd_epoch=7, detected=True,
                     zero_update_epoch=8, zero_update_layer="fc1"),
        detector_row("fresh", "none", "none", None, dead25_check_epoch=2, dead25_check_layer="fc3"),
    ]
    table = {(r["family"], r["detector"]): r for r in detector_table(rows) if r["arch"] == "mlp"}

    def counts(family, detector):
        r = table[(family, detector)]
        return r["caught"], r["caught_right_layer"]

    assert counts("proposal", "LHD") == (2, 1)
    assert counts("proposal", "Zero update check") == (1, 1)
    assert counts("proposal", "Zero update check, 3 epochs") == (1, 1)
    assert counts("proposal", "Dead units check, 25%") == (1, 0)
    assert counts("proposal", "Dead units check, 25%, 3 epochs") == (1, 1)
    assert counts("proposal", "Zero update or dead units 50%") == (2, 1)
    assert counts("proposal", "Zero update or dead units 50%, 3 epochs") == (1, 1)
    assert counts("frozen", "Zero update or dead units 50%") == (1, 0)
    lhd = table[("proposal", "LHD")]
    assert lhd["caught_right_layer_rate"] == 0.5
    assert (lhd["caught_right_layer_low"], lhd["caught_right_layer_high"]) == pytest.approx(wilson(1, 2))
    curve = table[("proposal", "Accuracy curve, 3 epochs below healthy")]
    assert curve["caught"] == 1
    assert all(curve[k] is None for k in ("caught_right_layer", "caught_right_layer_rate",
                                          "caught_right_layer_low", "caught_right_layer_high"))
    assert table[("lr10", "LHD")]["caught_right_layer"] is None
    assert table[("lr10", "Zero update check")]["caught"] == 1
    assert table[("lr10", "Zero update check")]["caught_right_layer"] is None
    assert table[("proposal", "Dead units check, 25%")]["false_alarms"] == 1
    assert table[("proposal", "Dead units check, 25%, 3 epochs")]["false_alarms"] == 0
