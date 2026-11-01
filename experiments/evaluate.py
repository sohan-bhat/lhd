"""Scores HALT and the accuracy curve on every saved run.

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
from pathlib import Path

import numpy as np

from experiments.grid import cut_affects, dead_layers
from halt.baseline import healthy_band
from halt.replay import replay

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


def zero_update_epoch(history: list[dict]) -> int | None:
    """First epoch in which some watched layer's weights did not move at all.

    A check that needs no healthy runs, the kind of assert an engineer
    might add by hand. It is a baseline for HALT, not part of it.
    """
    for entry in history:
        if any(values["weight_change"] == 0.0 for values in entry["layers"].values()):
            return entry["epoch"]
    return None


def dead_check_epoch(history: list[dict], threshold: float, layers=None) -> int | None:
    """First epoch in which some layer has at least `threshold` of its units dead.

    A fixed rule that needs no healthy runs, used as a baseline. On ReLU
    layers, threshold 0.5 matches the default of SageMaker Debugger's
    DeadRelu rule, 0.7 the Dead Node check of DeepDiagnosis, and 0.8 the
    default of the wandb-callbacks DeadReluCallback. ANY_DEAD matches
    that callback's verbose mode, which warns on a single dead unit.
    """
    for entry in history:
        if any(values["dead_unit_fraction"] >= threshold for name, values in entry["layers"].items()
               if layers is None or name in layers):
            return entry["epoch"]
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
    """Everything built from calibration runs: HALT's band and the curve bands."""
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
    halt_epoch = report["epoch"] if report else None
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
        "halt_epoch": halt_epoch,
        "halt_layer": report["layer"] if report else None,
        "halt_cause": report["cause"] if report else None,
        "halt_signals": "+".join(report["signals"]) if report else None,
        "acc_epoch": curve_alarm(history, ref["accuracy"], "val_accuracy", "below", patience),
        "acc1_epoch": curve_alarm(history, ref["accuracy"], "val_accuracy", "below", 1),
        "loss_epoch": curve_alarm(history, ref["loss"], "val_loss", "above", patience),
        "zero_update_epoch": zero_update_epoch(history),
        **{f"dead{int(t * 100)}_check_epoch": dead_check_epoch(history, t, relu) for t in DEAD_CHECKS},
        "deadany_check_epoch": dead_check_epoch(history, ANY_DEAD, relu),
        "final_acc": final,
        "best_acc": max(e["val_accuracy"] for e in history),
        "healthy_final_mean": ref["final_mean"],
        "final_in_healthy_range": final >= ref["final_low"],
        "twin_final_acc": twins.get(meta["seed"]),
        "halt_seconds": meta.get("halt_seconds"),
        "total_seconds": meta.get("total_seconds"),
    }
    _judge(row, fault, meta["arch"])
    return row


def _judge(row: dict, fault: dict, arch: str) -> None:
    """Add detection, layer, cause, gap, and compute-saved fields to a row."""
    kind, onset, epochs = fault["type"], row["onset"], row["epochs"]
    halt = row["halt_epoch"]
    row["fired"] = halt is not None
    row["fired_before_onset"] = halt is not None and kind != "none" and halt < onset
    row["detected"] = halt is not None and halt >= onset and kind != "none"
    if kind in ("frozen", "dead"):
        right_layers = [fault["layer"]]
    elif kind == "detach":
        right_layers = cut_affects(arch, fault["layer"])
    else:
        right_layers = None
    row["layer_ok"] = (row["halt_layer"] in right_layers) if (row["detected"] and right_layers) else None
    expected = EXPECTED_CAUSE.get(kind)
    row["cause_ok"] = (row["halt_cause"] == expected) if (row["detected"] and expected) else None
    if kind == "none":
        row["delay"] = row["acc_delay"] = row["gap"] = row["compute_saved"] = None
        return
    censor = epochs + 1
    acc = row["acc_epoch"] if row["acc_epoch"] is not None and row["acc_epoch"] >= onset else None
    row["acc_detected"] = acc is not None
    row["delay"] = (halt - onset + 1) if row["detected"] else None
    row["acc_delay"] = (acc - onset + 1) if acc is not None else None
    halt_c = halt if row["detected"] else censor
    acc_c = acc if acc is not None else censor
    row["halt_censored"] = halt_c
    row["acc_censored"] = acc_c
    row["gap"] = acc_c - halt_c
    row["compute_saved"] = (epochs - halt) / epochs if row["detected"] else 0.0
    twin = row["twin_final_acc"]
    row["acc_drop_vs_twin"] = (twin - row["final_acc"]) if twin is not None else None
    row["harm"] = harm_bin(row["acc_drop_vs_twin"])
    zero = row["zero_update_epoch"]
    simple = _first(zero, row["dead50_check_epoch"])
    row["zero_update_detected"] = zero is not None and zero >= onset
    row["simple_epoch"] = simple
    row["simple_detected"] = simple is not None and simple >= onset
    simple_c = simple if row["simple_detected"] else censor
    row["simple_gap"] = simple_c - halt_c


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
        "halt_first": rate([{**r, "x": r["halt_censored"] < r["acc_censored"]} for r in faults], "x"),
        "halt_median_epoch": _median([r["halt_epoch"] for r in faults if r["detected"]]),
        "acc_median_epoch": _median([r["acc_epoch"] for r in faults if r["acc_detected"]]),
        "halt_median_delay": _median([r["delay"] for r in faults]),
        "acc_median_delay": _median([r["acc_delay"] for r in faults]),
        "mean_gap": float(np.mean([r["gap"] for r in faults])),
        "mean_compute_saved": float(np.mean([r["compute_saved"] for r in faults])),
        "silent_share": rate(faults, "final_in_healthy_range"),
        "zero_update_detected": rate(faults, "zero_update_detected"),
        "simple_detected": rate(faults, "simple_detected"),
        "simple_median_epoch": _median([r["simple_epoch"] for r in faults if r["simple_detected"]]),
        "halt_before_simple": rate([{**r, "x": r["simple_gap"] > 0} for r in faults], "x"),
        "simple_before_halt": rate([{**r, "x": r["simple_gap"] < 0} for r in faults], "x"),
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
        "halt_false_alarm": rate(fresh, "fired"),
        "acc_false_alarm": rate([{**r, "x": r["acc_epoch"] is not None} for r in fresh], "x"),
        "acc1_false_alarm": rate([{**r, "x": r["acc1_epoch"] is not None} for r in fresh], "x"),
        "loss_false_alarm": rate([{**r, "x": r["loss_epoch"] is not None} for r in fresh], "x"),
        "zero_update_false_alarm": rate([{**r, "x": r["zero_update_epoch"] is not None} for r in fresh], "x"),
        "simple_false_alarm": rate([{**r, "x": _first(r["zero_update_epoch"], r["dead50_check_epoch"])
                                     is not None} for r in fresh], "x"),
        "overhead": _mean([r["halt_seconds"] / r["total_seconds"] for r in fresh
                           if r["halt_seconds"] is not None]),
    }


def paired_tests(rows: list[dict]) -> dict:
    """Is HALT earlier than the accuracy curve on the same broken runs?

    Uses the one-sided Wilcoxon signed-rank test and the sign test
    on (accuracy epoch - HALT epoch), with "never" counted as one
    epoch past the end of the run.
    """
    from scipy.stats import binomtest, wilcoxon

    faults = [r for r in rows if r["role"] == "fault"]
    gaps = np.array([r["gap"] for r in faults], dtype=float)
    result = {"n": int(len(gaps)), "mean_gap": float(gaps.mean()) if len(gaps) else None}
    nonzero = gaps[gaps != 0]
    result["halt_earlier"] = int((gaps > 0).sum())
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
        entry = {"groups": int(len(means)), "halt_earlier": int((means > 0).sum()),
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
