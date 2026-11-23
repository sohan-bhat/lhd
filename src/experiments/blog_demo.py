"""Writes blog/demo.json, the four runs shown in the blog post's figure.

Each run is replayed through LHD with the study's settings, so the
figure shows what LHD reported and nothing else. The runs are a
frozen layer (the one in the paper's Figure 1), dead units, a healthy
run, and a broken run LHD missed.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.evaluate import (SETTINGS, curve_alarm, family, load_runs,
                                  reference, score_arch, split_runs)
from experiments.paper_figures import pick_example
from lhd.replay import replay

OUT = Path(__file__).resolve().parents[2] / "blog" / "demo.json"
NETWORK = {
    "mlp": "Fully connected network on Fashion-MNIST",
    "transformer": "Character transformer on Tiny Shakespeare",
    "cnn": "CNN on GTSRB",
}
DIGITS = 5


def _round(values) -> list[float]:
    return [round(float(v), DIGITS) for v in values]


def _widened(band: dict, cap: float | None = None) -> tuple[list[float], list[float]]:
    """The healthy range as the detector uses it, widened by margin and floor."""
    low = band["low"] * (1 - SETTINGS["margin"])
    high = band["high"] * (1 + SETTINGS["margin"]) + SETTINGS["floor"]
    if cap is not None:
        high = high.clip(max=cap)
    return _round(low), _round(high)


def _example(key: str, run: dict, ref: dict, layer: str, signal: str) -> dict:
    """One run's curves, its healthy ranges, and what LHD reported."""
    meta, history = run["metadata"], run["history"]
    report = replay(history, ref["band"], **SETTINGS)
    low, high = _widened(ref["band"][layer][signal], cap=1.0 if signal == "dead_unit_fraction" else None)
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
    """The four examples, in the order the figure shows them."""
    refs, groups, rows = {}, {}, {}
    for arch in ("cnn", "mlp"):
        groups[arch] = split_runs(load_runs(arch))
        refs[arch] = reference(groups[arch])
        rows[arch] = score_arch(arch)

    def run_of(arch: str, run_id: str) -> dict:
        both = groups[arch]["fresh"] + groups[arch]["fault"]
        return next(r for r in both if r["metadata"]["run_id"] == run_id)

    frozen = pick_example(rows["cnn"], groups["cnn"], refs["cnn"])["run"]
    layer = frozen["metadata"]["fault"]["layer"]
    dead = run_of("mlp", "mlp-fault-dead25-fc1-s1000")
    healthy = run_of("cnn", f"cnn-fresh-none-s{frozen['metadata']['seed']}")
    missed = run_of("cnn", "cnn-fault-dead10-fc1-s1000")
    return [
        _example("frozen", frozen, refs["cnn"], layer, "weight_change"),
        _example("dead", dead, refs["mlp"], "fc1", "dead_unit_fraction"),
        _example("healthy", healthy, refs["cnn"], layer, "weight_change"),
        _example("missed", missed, refs["cnn"], "fc1", "dead_unit_fraction"),
    ]


if __name__ == "__main__":
    examples = build()
    OUT.write_text(json.dumps(examples, indent=1) + "\n")
    for example in examples:
        lhd = example["lhd"]
        print(example["key"], example["run_id"],
              f"LHD {lhd['layer']} {lhd['cause']} at epoch {lhd['epoch']}" if lhd else "LHD silent",
              f"accuracy curve {example['accuracy_alarm']}")
