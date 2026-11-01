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
from pathlib import Path

import numpy as np

from experiments.grid import cut_affects
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
    """Every saved run for an architecture, sorted by run id."""
    runs = []
    for path in sorted((RESULTS / "runs" / arch).glob("*.json")):
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
