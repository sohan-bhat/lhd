"""Scores every saved run and writes the study's tables.

Usage:
    python -m experiments.analyze            all architectures with runs
    python -m experiments.analyze --quick    skip the slower sweeps

Writes results/summary.json and CSV tables in results/tables/.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

from experiments.evaluate import (
    HARM_BINS,
    RESULTS,
    SETTINGS,
    clustered_tests,
    load_runs,
    paired_tests,
    rate,
    right_layers,
    score_arch,
    summarize_faults,
    summarize_healthy,
)
from experiments.grid import ARCH_ORDER
from experiments.sweep import calibration_sweep, settings_sweep, signal_ablation

TABLES = RESULTS / "tables"
PROPOSAL = ("frozen", "dead10", "dead25", "dead50")
DETECTORS = (
    ("LHD", "lhd_epoch"),
    ("Accuracy curve, 3 epochs below healthy", "acc_epoch"),
    ("Accuracy curve, 1 epoch below healthy", "acc1_epoch"),
    ("Loss curve, 3 epochs above healthy", "loss_epoch"),
    ("Zero update check", "zero_update_epoch"),
    ("Zero update check, 3 epochs", "zero_update_p3_epoch"),
    ("Dead units check, 25%", "dead25_check_epoch"),
    ("Dead units check, 25%, 3 epochs", "dead25_check_p3_epoch"),
    ("Dead units check, 50% (SageMaker DeadRelu default)", "dead50_check_epoch"),
    ("Dead units check, 50% (SageMaker DeadRelu default), 3 epochs", "dead50_check_p3_epoch"),
    ("Dead units check, 70% (DeepDiagnosis Dead Node)", "dead70_check_epoch"),
    ("Dead units check, 70% (DeepDiagnosis Dead Node), 3 epochs", "dead70_check_p3_epoch"),
    ("Dead units check, 80% (wandb-callbacks default)", "dead80_check_epoch"),
    ("Dead units check, 80% (wandb-callbacks default), 3 epochs", "dead80_check_p3_epoch"),
    ("Any dead unit (wandb-callbacks verbose)", "deadany_check_epoch"),
    ("Any dead unit (wandb-callbacks verbose), 3 epochs", "deadany_check_p3_epoch"),
    ("Zero update or dead units 50%", "simple_epoch"),
    ("Zero update or dead units 50%, 3 epochs", "simple_p3_epoch"),
)
NO_LAYER = ("acc_epoch", "acc1_epoch", "loss_epoch")
COMBINED = {"simple_epoch": "", "simple_p3_epoch": "_p3"}
MARGINS = (0.0, 0.1, 0.25, 0.5, 0.75)
FLOORS = (0.0, 0.02, 0.05, 0.1)
PATIENCES = (1, 2, 3, 4, 5)
CALIBRATION_SIZES = (2, 3, 5, 10, 15, 20)


def write_csv(path, rows: list[dict]) -> None:
    """Write rows to CSV with the union of their keys as columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def per_layer(rows: list[dict]) -> list[dict]:
    """Detection and diagnosis counts for each fault family and layer."""
    groups = defaultdict(list)
    for r in rows:
        if r["role"] == "fault" and r["fault_layer"]:
            groups[(r["arch"], r["family"], r["fault_layer"])].append(r)
    table = []
    for (arch, fam, layer), group in sorted(groups.items()):
        table.append({
            "arch": arch, "family": fam, "layer": layer, "runs": len(group),
            "detected": sum(r["detected"] for r in group),
            "layer_ok": sum(bool(r["layer_ok"]) for r in group),
            "cause_ok": sum(bool(r["cause_ok"]) for r in group),
            "acc_detected": sum(bool(r["acc_detected"]) for r in group),
            "median_lhd_epoch": _median_int([r["lhd_epoch"] for r in group if r["detected"]]),
            "median_acc_epoch": _median_int([r["acc_epoch"] for r in group if r["acc_detected"]]),
            "mean_acc_drop_vs_twin": _mean([r["acc_drop_vs_twin"] for r in group]),
            "silent": sum(bool(r["final_in_healthy_range"]) for r in group),
        })
    return table


def _median_int(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def confusion(rows: list[dict]) -> dict:
    """How often each planted fault type was diagnosed as each cause."""
    counts = defaultdict(Counter)
    for r in rows:
        if r["role"] == "fault":
            counts[r["family"]][r["lhd_cause"] if r["detected"] else "not detected"] += 1
    return {fam: dict(c) for fam, c in counts.items()}


def wrong_layers(rows: list[dict]) -> list[dict]:
    """Detected runs that named the wrong layer, to read by hand."""
    keep = ("arch", "run_id", "family", "fault_layer", "lhd_layer", "lhd_cause",
            "lhd_signals", "lhd_epoch")
    return [{k: r[k] for k in keep} for r in rows
            if r["role"] == "fault" and r["detected"] and r["layer_ok"] is False]


def false_alarms(rows: list[dict]) -> list[dict]:
    """Fresh healthy runs where LHD fired, to read by hand."""
    keep = ("arch", "run_id", "lhd_epoch", "lhd_layer", "lhd_cause", "lhd_signals")
    return [{k: r[k] for k in keep} for r in rows if r["role"] == "fresh" and r["fired"]]


def _alarm(row: dict, key: str) -> tuple:
    """Epoch and named layer of a detector's first alarm on a run.

    The layer is None for detectors that name no layer. The combined
    rule names the layer of whichever of its two checks fired first,
    and the zero update check's layer on a tie.
    """
    if key in COMBINED:
        suffix = COMBINED[key]
        zero = (row[f"zero_update{suffix}_epoch"], row[f"zero_update{suffix}_layer"])
        dead = (row[f"dead50_check{suffix}_epoch"], row[f"dead50_check{suffix}_layer"])
        if dead[0] is not None and (zero[0] is None or dead[0] < zero[0]):
            return dead
        return zero
    if key in NO_LAYER:
        return row[key], None
    return row[key], row[key[: -len("_epoch")] + "_layer"]


def _detector_epoch(row: dict, key: str):
    return _alarm(row, key)[0]


def _caught(row: dict, key: str) -> bool:
    epoch = _detector_epoch(row, key)
    return epoch is not None and epoch >= row["onset"]


def _caught_right_layer(row: dict, key: str) -> bool | None:
    """Whether a detector caught a broken run and named the planted layer.

    None when the detector names no layer or the fault has no layer.
    LHD uses its own layer_ok. Every detector is held to the same
    layers as LHD: the planted layer for frozen layers and dead units,
    and the layers upstream of the cut for a cut gradient.
    """
    right = right_layers(row["arch"], row["fault_type"], row["fault_layer"])
    if key in NO_LAYER or not right:
        return None
    if key == "lhd_epoch":
        return bool(row["detected"] and row["layer_ok"])
    return _caught(row, key) and _alarm(row, key)[1] in right


def detector_table(rows: list[dict]) -> list[dict]:
    """Every detector on the same runs: broken runs caught, when, and healthy runs flagged.

    One row per architecture (and pooled) and fault family (and the
    proposal's four families together) and detector. The
    caught_right_layer columns count broken runs caught with the
    planted layer named, and stay empty for detectors that name no
    layer and for faults that have no layer.
    """
    table = []
    archs = sorted({r["arch"] for r in rows}) + ["pooled"]
    for arch in archs:
        mine = [r for r in rows if arch == "pooled" or r["arch"] == arch]
        fresh = [r for r in mine if r["role"] == "fresh"]
        faults = [r for r in mine if r["role"] == "fault"]
        families = ["proposal"] + sorted({r["family"] for r in faults})
        for fam in families:
            group = [r for r in faults if (r["family"] in PROPOSAL if fam == "proposal" else r["family"] == fam)]
            if not group:
                continue
            for name, key in DETECTORS:
                caught = rate([{"x": _caught(r, key)} for r in group], "x")
                right = rate([{"x": _caught_right_layer(r, key)} for r in group], "x")
                named = right["n"] > 0
                epochs = [_detector_epoch(r, key) for r in group if _caught(r, key)]
                lhd_c = [r["lhd_epoch"] if r["detected"] else r["epochs"] + 1 for r in group]
                other_c = [_detector_epoch(r, key) if _caught(r, key) else r["epochs"] + 1 for r in group]
                alarms = rate([{"x": _detector_epoch(r, key) is not None} for r in fresh], "x")
                table.append({
                    "arch": arch, "family": fam, "detector": name, "broken_runs": len(group),
                    "caught": caught["k"], "caught_rate": caught["rate"],
                    "caught_low": caught["low"], "caught_high": caught["high"],
                    "caught_right_layer": right["k"] if named else None,
                    "caught_right_layer_rate": right["rate"] if named else None,
                    "caught_right_layer_low": right["low"] if named else None,
                    "caught_right_layer_high": right["high"] if named else None,
                    "median_epoch": _median_float(epochs),
                    "lhd_earlier": sum(h < o for h, o in zip(lhd_c, other_c)),
                    "this_earlier": sum(o < h for h, o in zip(lhd_c, other_c)),
                    "ties": sum(o == h for h, o in zip(lhd_c, other_c)),
                    "healthy_runs": len(fresh), "false_alarms": alarms["k"],
                    "false_alarm_rate": alarms["rate"],
                    "false_alarm_low": alarms["low"], "false_alarm_high": alarms["high"],
                })
    return table


def harm_table(rows: list[dict]) -> list[dict]:
    """Detection split by how much final accuracy the fault cost against its healthy twin."""
    table = []
    labels = [label for _, _, label in HARM_BINS]
    for scope in ("proposal", "all"):
        for arch in sorted({r["arch"] for r in rows}) + ["pooled"]:
            for label in labels:
                group = [r for r in rows if r["role"] == "fault" and r.get("harm") == label
                         and (arch == "pooled" or r["arch"] == arch)
                         and (scope == "all" or r["family"] in PROPOSAL)]
                if not group:
                    continue
                entry = {"scope": scope, "arch": arch, "harm": label, "broken_runs": len(group),
                         "mean_drop_pts": 100 * _mean([r["acc_drop_vs_twin"] for r in group])}
                for name, key in (("lhd", "lhd_epoch"), ("accuracy", "acc_epoch"),
                                  ("loss", "loss_epoch"), ("simple", "simple_epoch")):
                    caught = rate([{"x": _caught(r, key)} for r in group], "x")
                    entry[f"{name}_caught"] = caught["k"]
                    entry[f"{name}_rate"] = caught["rate"]
                    entry[f"{name}_low"] = caught["low"]
                    entry[f"{name}_high"] = caught["high"]
                    entry[f"{name}_median_epoch"] = _median_float(
                        [_detector_epoch(r, key) for r in group if _caught(r, key)])
                table.append(entry)
    return table


def _median_float(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def summarize_arch(rows: list[dict]) -> dict:
    """Every headline number for one architecture."""
    faults = [r for r in rows if r["role"] == "fault"]
    proposal = [r for r in faults if r["family"] in PROPOSAL]
    families = sorted({r["family"] for r in faults})
    return {
        "healthy": summarize_healthy(rows),
        "proposal": summarize_faults(proposal),
        "proposal_tests": paired_tests(proposal),
        "proposal_clustered": clustered_tests(proposal),
        "families": {f: summarize_faults([r for r in faults if r["family"] == f]) for f in families},
        "family_tests": {f: paired_tests([r for r in faults if r["family"] == f]) for f in families},
        "confusion": confusion(faults),
        "first_signal": dict(Counter(r["lhd_signals"] for r in faults if r["detected"])),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    archs = [a for a in ARCH_ORDER if (RESULTS / "runs" / a).exists()
             or (RESULTS / "runs" / f"{a}.jsonl.gz").exists()]
    all_rows, summary = [], {"settings": SETTINGS, "archs": {}}
    sweeps, calibration, ablation = [], [], []
    for arch in archs:
        runs = load_runs(arch)
        if not any(r["metadata"]["role"] == "calibration" for r in runs):
            continue
        counts = Counter(r["metadata"]["role"] for r in runs)
        rows = score_arch(arch, runs=runs)
        all_rows.extend(rows)
        summary["archs"][arch] = {"runs": dict(counts), **summarize_arch(rows)}
        print(f"{arch}: {dict(counts)}", flush=True)
        if not args.quick:
            sweeps.extend(settings_sweep(arch, runs, MARGINS, FLOORS, PATIENCES))
            calibration.extend(calibration_sweep(arch, runs, CALIBRATION_SIZES))
            ablation.extend(signal_ablation(arch, runs))

    proposal = [r for r in all_rows if r["family"] in PROPOSAL]
    summary["pooled"] = {
        "healthy": summarize_healthy(all_rows),
        "proposal": summarize_faults(proposal),
        "proposal_tests": paired_tests(proposal),
        "proposal_clustered": clustered_tests(proposal),
        "all_faults": summarize_faults([r for r in all_rows if r["role"] == "fault"]),
        "all_faults_clustered": clustered_tests([r for r in all_rows if r["role"] == "fault"]),
        "acc_never_on_detected": rate(
            [{**r, "x": not r["acc_detected"]} for r in proposal if r["detected"]], "x"),
    }
    write_csv(TABLES / "runs.csv", all_rows)
    write_csv(TABLES / "per_layer.csv", per_layer(all_rows))
    write_csv(TABLES / "wrong_layers.csv", wrong_layers(all_rows))
    write_csv(TABLES / "false_alarms.csv", false_alarms(all_rows))
    write_csv(TABLES / "detectors.csv", detector_table(all_rows))
    write_csv(TABLES / "harm.csv", harm_table(all_rows))
    if not args.quick:
        write_csv(TABLES / "sweep_settings.csv", sweeps)
        write_csv(TABLES / "sweep_calibration.csv", calibration)
        write_csv(TABLES / "ablation.csv", ablation)
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {len(all_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
