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
    RESULTS,
    SETTINGS,
    clustered_tests,
    load_runs,
    paired_tests,
    rate,
    score_arch,
    summarize_faults,
    summarize_healthy,
)
from experiments.grid import ARCH_ORDER
from experiments.sweep import calibration_sweep, settings_sweep, signal_ablation

TABLES = RESULTS / "tables"
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
            "median_halt_epoch": _median_int([r["halt_epoch"] for r in group if r["detected"]]),
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
            counts[r["family"]][r["halt_cause"] if r["detected"] else "not detected"] += 1
    return {fam: dict(c) for fam, c in counts.items()}


def wrong_layers(rows: list[dict]) -> list[dict]:
    """Detected runs that named the wrong layer, to read by hand."""
    keep = ("arch", "run_id", "family", "fault_layer", "halt_layer", "halt_cause",
            "halt_signals", "halt_epoch")
    return [{k: r[k] for k in keep} for r in rows
            if r["role"] == "fault" and r["detected"] and r["layer_ok"] is False]


def false_alarms(rows: list[dict]) -> list[dict]:
    """Fresh healthy runs where HALT fired, to read by hand."""
    keep = ("arch", "run_id", "halt_epoch", "halt_layer", "halt_cause", "halt_signals")
    return [{k: r[k] for k in keep} for r in rows if r["role"] == "fresh" and r["fired"]]


def summarize_arch(rows: list[dict]) -> dict:
    """Every headline number for one architecture."""
    faults = [r for r in rows if r["role"] == "fault"]
    proposal = [r for r in faults if r["family"] in ("frozen", "dead10", "dead25", "dead50")]
    families = sorted({r["family"] for r in faults})
    return {
        "healthy": summarize_healthy(rows),
        "proposal": summarize_faults(proposal),
        "proposal_tests": paired_tests(proposal),
        "proposal_clustered": clustered_tests(proposal),
        "families": {f: summarize_faults([r for r in faults if r["family"] == f]) for f in families},
        "family_tests": {f: paired_tests([r for r in faults if r["family"] == f]) for f in families},
        "confusion": confusion(faults),
        "first_signal": dict(Counter(r["halt_signals"] for r in faults if r["detected"])),
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

    proposal = [r for r in all_rows if r["family"] in ("frozen", "dead10", "dead25", "dead50")]
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
    if not args.quick:
        write_csv(TABLES / "sweep_settings.csv", sweeps)
        write_csv(TABLES / "sweep_calibration.csv", calibration)
        write_csv(TABLES / "ablation.csv", ablation)
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {len(all_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
