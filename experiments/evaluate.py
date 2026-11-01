"""Scores LHD and the accuracy curve on every saved run.

The healthy band comes from calibration runs only. Broken runs and
fresh healthy runs are scored against it and never change it.
The accuracy curve gets the same healthy runs and the same patience:
it raises suspicion once validation accuracy sits below the lowest
healthy value for that epoch, `patience` epochs in a row.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.grid import cut_affects, dead_layers
from lhd.baseline import healthy_band
from lhd.replay import replay

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETTINGS = {"margin": 0.5, "floor": 0.05, "patience": 3}
EXPECTED_CAUSE = {
    "frozen": "frozen_layer",
    "dead": "dead_units",
    "detach": "vanishing_gradient",
}


def load_runs(arch: str) -> list[dict]:
    """Every saved run for an architecture, sorted by run id.

    Reads the per-run files when they exist, and the compressed
    bundle in git otherwise.
    """
    paths = sorted((RESULTS / "runs" / arch).glob("*.json"))
    if not paths and (RESULTS / "runs" / f"{arch}.jsonl.gz").exists():
        from experiments.bundle import read_bundle

        return sorted(read_bundle(arch), key=lambda run: run["metadata"]["run_id"])
    runs = []
    for path in paths:
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def split_runs(runs: list[dict]) -> dict[str, list[dict]]:
    """Group runs by role: calibration, fresh, fault."""
    groups = {"calibration": [], "fresh": [], "fault": []}
    for run in runs:
        groups[run["metadata"]["role"]].append(run)
    return groups


def family(fault: dict) -> str:
    """Short name for a fault's kind, such as frozen, dead25, latefrozen, lr10."""
    kind = fault["type"]
    late = "late" if fault.get("onset", 1) > 1 and kind in ("frozen", "dead") else ""
    if kind == "dead":
        return f"{late}dead{int(round(fault['fraction'] * 100))}"
    if kind == "lr":
        return f"lr{fault['factor']}"
    return f"{late}{kind}"


def curve_band(histories: list[list[dict]], key: str) -> dict:
    """Lowest and highest value of a curve (val_accuracy or val_loss) per epoch."""
    values = np.array([[entry[key] for entry in history] for history in histories])
    return {"low": values.min(axis=0), "high": values.max(axis=0)}


def curve_alarm(history: list[dict], band: dict, key: str, direction: str,
                patience: int) -> int | None:
    """First epoch at which a curve has been worse than every healthy run
    for `patience` epochs in a row, or None.

    direction is "below" for accuracy and "above" for loss. A value
    that is nan or infinite always counts as worse.
    """
    streak = 0
    for i, entry in enumerate(history):
        value = entry[key]
        if not math.isfinite(value):
            worse = True
        elif direction == "below":
            worse = value < band["low"][i]
        else:
            worse = value > band["high"][i]
        streak = streak + 1 if worse else 0
        if streak >= patience:
            return entry["epoch"]
    return None


DEAD_CHECKS = (0.25, 0.5, 0.7, 0.8)
ANY_DEAD = 1e-9
HARM_BINS = ((None, 0.005, "under 0.5 pts"), (0.005, 0.02, "0.5 to 2 pts"), (0.02, None, "2 pts or more"))
FIXED_PATIENCES = {"": 1, "_p3": 3}


def first_streak(history: list[dict], hit, patience: int = 1, layers=None) -> tuple[int | None, str | None]:
    """First epoch and layer at which some layer has met `hit` for
    `patience` epochs in a row, or (None, None).

    `hit` takes one layer's values for one epoch. Layers are tried in
    the order of entry["layers"], which is data-flow order, so the
    earliest layer wins when several qualify in the same epoch. A
    layer missing from an epoch starts its count again.
    """
    streaks: dict[str, int] = {}
    for entry in history:
        streaks = {name: streaks.get(name, 0) + 1 if hit(values) else 0
                   for name, values in entry["layers"].items()
                   if layers is None or name in layers}
        for name, streak in streaks.items():
            if streak >= patience:
                return entry["epoch"], name
    return None, None


def zero_update_check(history: list[dict], patience: int = 1) -> tuple[int | None, str | None]:
    """First epoch and layer at which some watched layer's weights did not
    move at all, `patience` epochs in a row.

    A check that needs no healthy runs, the kind of assert an engineer
    might add by hand. It is a baseline for LHD, not part of it.
    """
    return first_streak(history, lambda values: values["weight_change"] == 0.0, patience)


def zero_update_epoch(history: list[dict], patience: int = 1) -> int | None:
    """Epoch of `zero_update_check`, or None."""
    return zero_update_check(history, patience)[0]


def dead_check(history: list[dict], threshold: float, layers=None,
               patience: int = 1) -> tuple[int | None, str | None]:
    """First epoch and layer at which some layer has had at least
    `threshold` of its units dead, `patience` epochs in a row.

    A fixed rule that needs no healthy runs, used as a baseline. On ReLU
    layers, threshold 0.5 matches the default of SageMaker Debugger's
    DeadRelu rule, 0.7 the Dead Node check of DeepDiagnosis, and 0.8 the
    default of the wandb-callbacks DeadReluCallback. ANY_DEAD matches
    that callback's verbose mode, which warns on a single dead unit.
    """
    return first_streak(history, lambda values: values["dead_unit_fraction"] >= threshold,
                        patience, layers)


def dead_check_epoch(history: list[dict], threshold: float, layers=None, patience: int = 1) -> int | None:
    """Epoch of `dead_check`, or None."""
    return dead_check(history, threshold, layers, patience)[0]


def fixed_checks(history: list[dict], relu: set[str]) -> dict:
    """Epoch and layer of every fixed check, at patience 1 and 3.

    Names are `<check>_epoch` and `<check>_layer` at patience 1 and
    `<check>_p3_epoch` and `<check>_p3_layer` at patience 3. Dead
    unit checks look only at ReLU layers.
    """
    fields = {}
    for suffix, patience in FIXED_PATIENCES.items():
        found = {"zero_update": zero_update_check(history, patience)}
        for t in DEAD_CHECKS:
            found[f"dead{int(t * 100)}_check"] = dead_check(history, t, relu, patience)
        found["deadany_check"] = dead_check(history, ANY_DEAD, relu, patience)
        for name, (epoch, layer) in found.items():
            fields[f"{name}{suffix}_epoch"] = epoch
            fields[f"{name}{suffix}_layer"] = layer
    return fields


@lru_cache(maxsize=None)
def right_layers(arch: str, kind: str, layer: str | None) -> tuple[str, ...] | None:
    """Layers a detector should name for a planted fault, or None when
    the fault has no layer.

    The planted layer for frozen layers and dead units, and the layers
    upstream of the cut for a cut gradient, which are the ones that
    stop learning.
    """
    if kind in ("frozen", "dead"):
        return (layer,)
    if kind == "detach":
        return tuple(cut_affects(arch, layer))
    return None


_RELU = {}


def _relu_layers(arch: str) -> set[str]:
    if arch not in _RELU:
        _RELU[arch] = set(dead_layers(arch))
    return _RELU[arch]


def harm_bin(drop: float | None) -> str | None:
    """Which harm group a broken run falls in, from its accuracy drop against its twin."""
    if drop is None:
        return None
    for low, high, label in HARM_BINS:
        if (low is None or drop >= low) and (high is None or drop < high):
            return label
    return None


def _first(*epochs):
    epochs = [e for e in epochs if e is not None]
    return min(epochs) if epochs else None


def reference(groups: dict[str, list[dict]]) -> dict:
    """Everything built from calibration runs: LHD's band and the curve bands."""
    histories = [run["history"] for run in groups["calibration"]]
    finals = [h[-1]["val_accuracy"] for h in histories]
    return {
        "band": healthy_band(histories),
        "accuracy": curve_band(histories, "val_accuracy"),
        "loss": curve_band(histories, "val_loss"),
        "final_mean": float(np.mean(finals)),
        "final_low": float(np.min(finals)),
    }


def score_run(run: dict, ref: dict, settings: dict, twins: dict[int, float]) -> dict:
    """One row of results for one run."""
    meta, history = run["metadata"], run["history"]
    fault = meta["fault"]
    epochs = len(history)
    onset = fault.get("onset", 1)
    report = replay(history, ref["band"], **settings)
    patience = settings["patience"]
    relu = _relu_layers(meta["arch"])
    lhd_epoch = report["epoch"] if report else None
    final = history[-1]["val_accuracy"]
    row = {
        "arch": meta["arch"],
        "run_id": meta["run_id"],
        "role": meta["role"],
        "phase": meta.get("phase", "core"),
        "seed": meta["seed"],
        "family": family(fault) if fault["type"] != "none" else "none",
        "fault_type": fault["type"],
        "fault_layer": fault.get("layer"),
        "onset": onset,
        "epochs": epochs,
        "lhd_epoch": lhd_epoch,
        "lhd_layer": report["layer"] if report else None,
        "lhd_cause": report["cause"] if report else None,
        "lhd_signals": "+".join(report["signals"]) if report else None,
        "acc_epoch": curve_alarm(history, ref["accuracy"], "val_accuracy", "below", patience),
        "acc1_epoch": curve_alarm(history, ref["accuracy"], "val_accuracy", "below", 1),
        "loss_epoch": curve_alarm(history, ref["loss"], "val_loss", "above", patience),
        **fixed_checks(history, relu),
        "final_acc": final,
        "best_acc": max(e["val_accuracy"] for e in history),
        "healthy_final_mean": ref["final_mean"],
        "final_in_healthy_range": final >= ref["final_low"],
        "twin_final_acc": twins.get(meta["seed"]),
        "lhd_seconds": meta.get("lhd_seconds"),
        "total_seconds": meta.get("total_seconds"),
    }
    _judge(row, fault, meta["arch"])
    return row


def _judge(row: dict, fault: dict, arch: str) -> None:
    """Add detection, layer, cause, gap, and compute-saved fields to a row."""
    kind, onset, epochs = fault["type"], row["onset"], row["epochs"]
    lhd = row["lhd_epoch"]
    row["fired"] = lhd is not None
    row["fired_before_onset"] = lhd is not None and kind != "none" and lhd < onset
    row["detected"] = lhd is not None and lhd >= onset and kind != "none"
    right = right_layers(arch, kind, fault.get("layer"))
    row["layer_ok"] = (row["lhd_layer"] in right) if (row["detected"] and right) else None
    expected = EXPECTED_CAUSE.get(kind)
    row["cause_ok"] = (row["lhd_cause"] == expected) if (row["detected"] and expected) else None
    if kind == "none":
        row["delay"] = row["acc_delay"] = row["gap"] = row["compute_saved"] = None
        return
    censor = epochs + 1
    acc = row["acc_epoch"] if row["acc_epoch"] is not None and row["acc_epoch"] >= onset else None
    row["acc_detected"] = acc is not None
    row["delay"] = (lhd - onset + 1) if row["detected"] else None
    row["acc_delay"] = (acc - onset + 1) if acc is not None else None
    lhd_c = lhd if row["detected"] else censor
    acc_c = acc if acc is not None else censor
    row["lhd_censored"] = lhd_c
    row["acc_censored"] = acc_c
    row["gap"] = acc_c - lhd_c
    row["compute_saved"] = (epochs - lhd) / epochs if row["detected"] else 0.0
    twin = row["twin_final_acc"]
    row["acc_drop_vs_twin"] = (twin - row["final_acc"]) if twin is not None else None
    row["harm"] = harm_bin(row["acc_drop_vs_twin"])
    zero = row["zero_update_epoch"]
    simple = _first(zero, row["dead50_check_epoch"])
    row["zero_update_detected"] = zero is not None and zero >= onset
    row["simple_epoch"] = simple
    row["simple_detected"] = simple is not None and simple >= onset
    simple_c = simple if row["simple_detected"] else censor
    row["simple_gap"] = simple_c - lhd_c


def score_arch(arch: str, settings: dict = SETTINGS, runs: list[dict] | None = None) -> list[dict]:
    """Rows for every fresh and fault run of an architecture."""
    groups = split_runs(runs if runs is not None else load_runs(arch))
    ref = reference(groups)
    twins = {r["metadata"]["seed"]: r["history"][-1]["val_accuracy"] for r in groups["fresh"]}
    return [score_run(run, ref, settings, twins) for run in groups["fresh"] + groups["fault"]]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(rows: list[dict], key: str) -> dict:
    """Share of rows where key is True, ignoring rows where it is None."""
    values = [r[key] for r in rows if r.get(key) is not None]
    hits = sum(bool(v) for v in values)
    low, high = wilson(hits, len(values))
    return {"k": hits, "n": len(values), "rate": hits / len(values) if values else float("nan"),
            "low": low, "high": high}


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarize_faults(rows: list[dict]) -> dict:
    """Detection, diagnosis, timing, and compute numbers for a set of broken runs."""
    faults = [r for r in rows if r["role"] == "fault"]
    if not faults:
        return {}
    return {
        "runs": len(faults),
        "detected": rate(faults, "detected"),
        "fired_before_onset": rate(faults, "fired_before_onset"),
        "layer_ok": rate(faults, "layer_ok"),
        "cause_ok": rate(faults, "cause_ok"),
        "acc_detected": rate(faults, "acc_detected"),
        "lhd_first": rate([{**r, "x": r["lhd_censored"] < r["acc_censored"]} for r in faults], "x"),
        "lhd_median_epoch": _median([r["lhd_epoch"] for r in faults if r["detected"]]),
        "acc_median_epoch": _median([r["acc_epoch"] for r in faults if r["acc_detected"]]),
        "lhd_median_delay": _median([r["delay"] for r in faults]),
        "acc_median_delay": _median([r["acc_delay"] for r in faults]),
        "mean_gap": float(np.mean([r["gap"] for r in faults])),
        "mean_compute_saved": float(np.mean([r["compute_saved"] for r in faults])),
        "silent_share": rate(faults, "final_in_healthy_range"),
        "zero_update_detected": rate(faults, "zero_update_detected"),
        "simple_detected": rate(faults, "simple_detected"),
        "simple_median_epoch": _median([r["simple_epoch"] for r in faults if r["simple_detected"]]),
        "lhd_before_simple": rate([{**r, "x": r["simple_gap"] > 0} for r in faults], "x"),
        "simple_before_lhd": rate([{**r, "x": r["simple_gap"] < 0} for r in faults], "x"),
        "mean_acc_drop_vs_twin": _mean([r["acc_drop_vs_twin"] for r in faults]),
    }


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize_healthy(rows: list[dict]) -> dict:
    """False alarm numbers on fresh healthy runs."""
    fresh = [r for r in rows if r["role"] == "fresh"]
    return {
        "runs": len(fresh),
        "lhd_false_alarm": rate(fresh, "fired"),
        "acc_false_alarm": rate([{**r, "x": r["acc_epoch"] is not None} for r in fresh], "x"),
        "acc1_false_alarm": rate([{**r, "x": r["acc1_epoch"] is not None} for r in fresh], "x"),
        "loss_false_alarm": rate([{**r, "x": r["loss_epoch"] is not None} for r in fresh], "x"),
        "zero_update_false_alarm": rate([{**r, "x": r["zero_update_epoch"] is not None} for r in fresh], "x"),
        "simple_false_alarm": rate([{**r, "x": _first(r["zero_update_epoch"], r["dead50_check_epoch"])
                                     is not None} for r in fresh], "x"),
        "overhead": _mean([r["lhd_seconds"] / r["total_seconds"] for r in fresh
                           if r["lhd_seconds"] is not None]),
    }


def paired_tests(rows: list[dict]) -> dict:
    """Is LHD earlier than the accuracy curve on the same broken runs?

    Uses the one-sided Wilcoxon signed-rank test and the sign test
    on (accuracy epoch - LHD epoch), with "never" counted as one
    epoch past the end of the run.
    """
    from scipy.stats import binomtest, wilcoxon

    faults = [r for r in rows if r["role"] == "fault"]
    gaps = np.array([r["gap"] for r in faults], dtype=float)
    result = {"n": int(len(gaps)), "mean_gap": float(gaps.mean()) if len(gaps) else None}
    nonzero = gaps[gaps != 0]
    result["lhd_earlier"] = int((gaps > 0).sum())
    result["acc_earlier"] = int((gaps < 0).sum())
    result["ties"] = int((gaps == 0).sum())
    if len(nonzero):
        result["wilcoxon_p"] = float(wilcoxon(nonzero, alternative="greater").pvalue)
        result["sign_p"] = float(binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5,
                                           alternative="greater").pvalue)
    return result


def clustered_tests(rows: list[dict], draws: int = 10000, seed: int = 0) -> dict:
    """Paired tests that respect how broken runs are grouped.

    Runs that share a seed share starting weights and data order, and
    runs that share a fault condition (fault and layer) share the same
    break, so single runs are not independent. Gaps are averaged
    within each group before the one-sided Wilcoxon and sign tests,
    and the 95% interval for the mean gap comes from a bootstrap that
    resamples whole seeds within each architecture.
    """
    from scipy.stats import binomtest, wilcoxon

    faults = [r for r in rows if r["role"] == "fault"]
    by_seed, by_condition = defaultdict(list), defaultdict(list)
    for r in faults:
        by_seed[(r["arch"], r["seed"])].append(r["gap"])
        by_condition[(r["arch"], r["family"], r["fault_layer"])].append(r["gap"])
    result = {}
    for name, groups in (("by_seed", by_seed), ("by_condition", by_condition)):
        means = np.array([np.mean(v) for v in groups.values()], dtype=float)
        entry = {"groups": int(len(means)), "lhd_earlier": int((means > 0).sum()),
                 "acc_earlier": int((means < 0).sum()), "ties": int((means == 0).sum())}
        nonzero = means[means != 0]
        if len(nonzero):
            entry["wilcoxon_p"] = float(wilcoxon(nonzero, alternative="greater").pvalue)
            entry["sign_p"] = float(binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5,
                                              alternative="greater").pvalue)
        result[name] = entry
    rng = np.random.default_rng(seed)
    per_arch = defaultdict(list)
    for (arch, _), gaps in by_seed.items():
        per_arch[arch].append(np.array(gaps, dtype=float))
    if per_arch:
        means = []
        for _ in range(draws):
            picked = []
            for groups in per_arch.values():
                picked.extend(groups[i] for i in rng.integers(0, len(groups), len(groups)))
            means.append(np.concatenate(picked).mean())
        result["mean_gap_ci"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    return result
