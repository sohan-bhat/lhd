import gzip
import json

import pytest

pytest.importorskip("torch")

from experiments import evaluate  # noqa: E402
from experiments.never_fired import all_specs, same_as_saved  # noqa: E402
from experiments.v2 import detail, headline, is_proposal  # noqa: E402
from lhd.baseline import NEVER_FIRED, SIGNALS  # noqa: E402


def history(value=0.5, epochs=2):
    layer = {**{s: value for s in SIGNALS}, "bias_change": 0.1, NEVER_FIRED: 0.0}
    return [{"epoch": i + 1, "train_accuracy": 0.9, "val_accuracy": 0.8, "train_loss": 0.3,
             "val_loss": 0.4, "layers": {"fc1": dict(layer)}} for i in range(epochs)]


def test_all_specs_covers_the_study_once():
    ids = [spec["run_id"] for spec in all_specs()]
    assert len(ids) == len(set(ids)) == 885


def test_same_as_saved_ignores_the_new_signal():
    saved = history()
    for entry in saved:
        del entry["layers"]["fc1"][NEVER_FIRED]
    assert same_as_saved(history(), saved)


def test_same_as_saved_catches_one_changed_number():
    changed = history()
    changed[1]["layers"]["fc1"]["weight_norm"] += 1e-9
    assert not same_as_saved(changed, history())
    shorter = history(epochs=1)
    assert not same_as_saved(shorter, history())


def test_add_never_fired_fills_missing_layers_with_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    (tmp_path / "runs").mkdir()
    with gzip.open(evaluate.never_fired_path("mlp"), "wt") as f:
        json.dump({"a": {"fc1": [0.25, 0.125]}, "b": {}}, f)
    runs = [{"metadata": {"run_id": name}, "history": history()} for name in ("a", "b", "c")]
    for run in runs:
        for entry in run["history"]:
            del entry["layers"]["fc1"][NEVER_FIRED]
    a, b, c = evaluate.add_never_fired("mlp", runs)
    assert [e["layers"]["fc1"][NEVER_FIRED] for e in a["history"]] == [0.25, 0.125]
    assert [e["layers"]["fc1"][NEVER_FIRED] for e in b["history"]] == [0.0, 0.0]
    assert NEVER_FIRED not in c["history"][0]["layers"]["fc1"]


def row(**changes):
    base = {"role": "fault", "fault_type": "dead", "onset": 1, "detected": True, "layer_ok": True,
            "lhd_epoch": 3, "harm": "2 pts or more", "fired": True}
    return {**base, **changes}


def test_is_proposal_leaves_out_late_and_other_faults():
    assert is_proposal(row())
    assert not is_proposal(row(onset=6))
    assert not is_proposal(row(fault_type="lr"))
    assert not is_proposal(row(role="fresh", fault_type="none"))


def test_headline_counts():
    rows = [row(), row(detected=False, layer_ok=None, harm="under 0.5 pts"), row(layer_ok=False),
            row(onset=6), row(role="fresh", fault_type="none", detected=False, fired=True),
            row(role="fresh", fault_type="none", detected=False, fired=False)]
    assert headline(rows) == {"broken": 3, "caught": 2, "right_layer": 1, "median_epoch": 3.0,
                              "harmful": 2, "harmful_caught": 2, "healthy": 2, "healthy_flagged": 1}


def test_detail_counts_what_the_write_up_quotes():
    rows = [
        row(run_id="a", family="dead10", acc_detected=True, acc_epoch=9),
        row(run_id="b", family="dead10", acc_detected=False, acc_epoch=None, layer_ok=False),
        row(run_id="c", family="dead50", detected=False, acc_detected=True, acc_epoch=5),
        row(run_id="d", family="frozen", fault_type="frozen", acc_detected=True, acc_epoch=4),
        row(run_id="e", family="latedead50", onset=6, acc_detected=False, acc_epoch=None),
        row(run_id="f", role="fresh", family="none", fault_type="none", acc_epoch=7),
    ]
    result = detail(rows)
    assert result["accuracy_curve_caught"] == 3
    assert result["accuracy_curve_healthy_flagged"] == 1
    assert result["caught_both_ways"] == 2
    assert result["mean_epochs_before_accuracy_curve"] == pytest.approx((6 + 1) / 2)
    assert result["frozen"] == {"caught": 1, "runs": 1}
    assert result["dead"] == {"caught": 2, "runs": 3}
    assert result["dead_by_share"]["dead10"] == {"caught": 2, "runs": 2}
    assert result["dead_by_share"]["dead50"] == {"caught": 0, "runs": 1}
    assert result["missed"] == ["c"]
    assert result["wrong_layer"] == ["b"]
