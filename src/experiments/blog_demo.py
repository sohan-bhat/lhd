"""Writes blog/demo.json, the five runs shown in the blog post's figure.

Each run is replayed through LHD with all five signals and the study's
settings, so the figure shows what LHD reported and nothing else. The
runs are a frozen layer (the one pick_example chooses), dead units,
a healthy run, a broken run LHD missed, and a healthy run LHD flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.evaluate import (SETTINGS, curve_alarm, family, load_runs,
                                  reference, score_arch, split_runs)
from lhd.baseline import NEVER_FIRED, SIGNALS_V2
from lhd.detector import signal_margin
from lhd.replay import replay

OUT = Path(__file__).resolve().parents[2] / "blog" / "demo.json"
NETWORK = {
    "mlp": "Fully connected network on Fashion-MNIST",
    "transformer": "Character transformer on Tiny Shakespeare",
    "cnn": "CNN on GTSRB",
}
DIGITS = 5
SHARES = ("dead_unit_fraction", NEVER_FIRED)


def _round(values) -> list[float]:
    return [round(float(v), DIGITS) for v in values]


def _widened(band: dict, signal: str) -> tuple[list[float], list[float]]:
    """The healthy range as the detector uses it, widened by margin and floor."""
    margin = signal_margin(signal, SETTINGS["margin"])
    low = band["low"] * (1 - margin)
    high = band["high"] * (1 + margin) + SETTINGS["floor"]
    if signal in SHARES:
        high = high.clip(max=1.0)
    return _round(low), _round(high)


def pick_example(rows: list[dict], groups: dict, ref: dict) -> dict | None:
    """A CNN run with a frozen layer that LHD detected in the right layer with
    the right cause, whose accuracy at the alarm epoch was still inside the
    calibration range. Among those, the one the accuracy curve noticed
    latest (or never), then the one that lost the most accuracy."""
    by_id = {run["metadata"]["run_id"]: run for run in groups["fault"]}
    picks = []
    for row in rows:
        if not (row["arch"] == "cnn" and row["family"] == "frozen" and row["detected"]
                and row["layer_ok"] and row["cause_ok"] and row["run_id"] in by_id):
            continue
        run = by_id[row["run_id"]]
        report = replay(run["history"], ref["band"], **SETTINGS)
        if report is None:
            continue
        i = report["epoch"] - 1
        value = run["history"][i]["val_accuracy"]
        if not ref["accuracy"]["low"][i] <= value <= ref["accuracy"]["high"][i]:
            continue
        picks.append((-(row["acc_censored"] - row["lhd_censored"]), -(row["acc_drop_vs_twin"] or 0),
                      row["run_id"], run, report))
    if not picks:
        return None
    _, _, run_id, run, report = min(picks, key=lambda p: p[:3])
    return {"run": run, "report": report}


def _example(key: str, run: dict, ref: dict, layer: str, signal: str) -> dict:
    """One run's curves, its healthy ranges, and what LHD reported."""
    meta, history = run["metadata"], run["history"]
    report = replay(history, ref["band"], **SETTINGS)
    low, high = _widened(ref["band"][layer][signal], signal)
    return {
        "key": key,
        "run_id": meta["run_id"],
        "network": NETWORK[meta["arch"]],
        "fault": None if meta["fault"]["type"] == "none" else
                 {"kind": family(meta["fault"]), "layer": meta["fault"]["layer"]},
        "epochs": len(history),
        "layer": layer,
        "signal": signal,
        "values": _round(e["layers"][layer][signal] for e in history),
        "low": low,
        "high": high,
        "accuracy": _round(e["val_accuracy"] for e in history),
        "accuracy_low": _round(ref["accuracy"]["low"]),
        "accuracy_high": _round(ref["accuracy"]["high"]),
        "lhd": None if report is None else
               {"epoch": report["epoch"], "layer": report["layer"], "cause": report["cause"],
                "positions": report["positions"]},
        "accuracy_alarm": curve_alarm(history, ref["accuracy"], "val_accuracy", "below",
                                      SETTINGS["patience"]),
    }


def build() -> list[dict]:
    """The five examples, in the order the figure shows them."""
    refs, groups, rows = {}, {}, {}
    for arch in ("cnn", "mlp"):
        runs = load_runs(arch)
        groups[arch] = split_runs(runs)
        refs[arch] = reference(groups[arch], SIGNALS_V2)
        rows[arch] = score_arch(arch, runs=runs, signals=SIGNALS_V2)

    def run_of(arch: str, run_id: str) -> dict:
        both = groups[arch]["fresh"] + groups[arch]["fault"]
        return next(r for r in both if r["metadata"]["run_id"] == run_id)

    frozen = pick_example(rows["cnn"], groups["cnn"], refs["cnn"])["run"]
    layer = frozen["metadata"]["fault"]["layer"]
    dead = run_of("mlp", "mlp-fault-dead25-fc1-s1000")
    missed = run_of("cnn", "cnn-fault-dead10-fc1-s1000")
    # The healthy run is the frozen run's twin, or the next seed LHD stayed silent on,
    # and the flagged run is the first healthy CNN run the fifth signal flagged.
    fresh = sorted((r for r in rows["cnn"] if r["role"] == "fresh"), key=lambda r: r["seed"])
    seed = frozen["metadata"]["seed"]
    quiet = next(r for r in fresh if r["seed"] >= seed and not r["fired"])
    loud = next(r for r in fresh if r["fired"] and NEVER_FIRED in r["lhd_signals"].split("+"))
    return [
        _example("frozen", frozen, refs["cnn"], layer, "weight_change"),
        _example("dead", dead, refs["mlp"], "fc1", "dead_unit_fraction"),
        _example("healthy", run_of("cnn", quiet["run_id"]), refs["cnn"], layer, "weight_change"),
        _example("missed", missed, refs["cnn"], "fc1", NEVER_FIRED),
        _example("flagged", run_of("cnn", loud["run_id"]), refs["cnn"], loud["lhd_layer"], NEVER_FIRED),
    ]


if __name__ == "__main__":
    examples = build()
    OUT.write_text(json.dumps(examples, indent=1) + "\n")
    for example in examples:
        lhd = example["lhd"]
        print(example["key"], example["run_id"],
              f"LHD {lhd['layer']} {lhd['cause']} at epoch {lhd['epoch']}" if lhd else "LHD silent",
              f"accuracy curve {example['accuracy_alarm']}")
