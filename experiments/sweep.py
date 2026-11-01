"""Robustness checks that replay saved runs under other settings.

None of these choose LHD's settings. The settings were fixed before
any broken run was trained; these sweeps only show how much the
results would move if they had been different.
"""

from __future__ import annotations

import copy

import numpy as np

from experiments.evaluate import SETTINGS, reference, score_run, split_runs
from lhd.baseline import SIGNALS

PROPOSAL_FAMILIES = ("frozen", "dead10", "dead25", "dead50")


def _proposal_faults(groups: dict) -> list[dict]:
    """Broken runs from the proposal: frozen layers and dead units from epoch 1."""
    return [r for r in groups["fault"]
            if r["metadata"]["fault"]["type"] in ("frozen", "dead")
            and r["metadata"]["fault"].get("onset", 1) == 1]


def outcome(groups: dict, ref: dict, settings: dict, faults: list[dict] | None = None) -> dict:
    """Headline numbers for one band and one setting."""
    faults = faults if faults is not None else _proposal_faults(groups)
    fault_rows = [score_run(r, ref, settings, {}) for r in faults]
    fresh_rows = [score_run(r, ref, settings, {}) for r in groups["fresh"]]
    detected = [r for r in fault_rows if r["detected"]]
    return {
        "faults": len(fault_rows),
        "fresh": len(fresh_rows),
        "detection": len(detected) / len(fault_rows) if fault_rows else float("nan"),
        "false_alarm": float(np.mean([r["fired"] for r in fresh_rows])) if fresh_rows else float("nan"),
        "layer_ok": float(np.mean([r["layer_ok"] for r in detected])) if detected else float("nan"),
        "cause_ok": float(np.mean([r["cause_ok"] for r in detected])) if detected else float("nan"),
        "both_ok": (sum(bool(r["layer_ok"] and r["cause_ok"]) for r in fault_rows) / len(fault_rows)
                    if fault_rows else float("nan")),
        "mean_epoch": float(np.mean([r["lhd_epoch"] for r in detected])) if detected else float("nan"),
        "compute_saved": float(np.mean([r["compute_saved"] for r in fault_rows])) if fault_rows else float("nan"),
    }


def settings_sweep(arch: str, runs: list[dict], margins, floors, patiences) -> list[dict]:
    """Outcome for every combination of margin, floor, and patience."""
    groups = split_runs(runs)
    ref = reference(groups)
    rows = []
    for margin in margins:
        for floor in floors:
            for patience in patiences:
                settings = {"margin": margin, "floor": floor, "patience": patience}
                rows.append({"arch": arch, **settings, **outcome(groups, ref, settings)})
    return rows


def calibration_sweep(arch: str, runs: list[dict], sizes, draws: int = 20, seed: int = 0) -> list[dict]:
    """Outcome when the band is built from only k of the calibration runs.

    Each size is repeated with `draws` random subsets.
    """
    groups = split_runs(runs)
    rng = np.random.default_rng(seed)
    pool = groups["calibration"]
    rows = []
    for k in sizes:
        repeats = 1 if k >= len(pool) else draws
        for draw in range(repeats):
            chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
            ref = reference({**groups, "calibration": chosen})
            rows.append({"arch": arch, "k": k, "draw": draw, **outcome(groups, ref, SETTINGS)})
    return rows


def _without(band: dict, signals) -> dict:
    """A copy of the band where the given signals can never leave it."""
    band = copy.deepcopy(band)
    for layer in band.values():
        for signal in signals:
            size = len(layer[signal]["low"])
            layer[signal] = {"low": np.full(size, -np.inf), "high": np.full(size, np.inf)}
    return band


def signal_ablation(arch: str, runs: list[dict]) -> list[dict]:
    """Outcome with all four signals, with each one removed, and with each one alone."""
    groups = split_runs(runs)
    ref = reference(groups)
    variants = [("all four", ())]
    variants += [(f"without {s}", (s,)) for s in SIGNALS]
    variants += [(f"only {s}", tuple(x for x in SIGNALS if x != s)) for s in SIGNALS]
    rows = []
    for name, removed in variants:
        trimmed = {**ref, "band": _without(ref["band"], removed)}
        rows.append({"arch": arch, "variant": name, **outcome(groups, trimmed, SETTINGS)})
    return rows
