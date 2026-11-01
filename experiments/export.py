"""Exports every number in the study as flat tables and one workbook.

Usage:
    python -m experiments.export

Run after experiments.analyze. Writes these files.
    results/tables/networks.csv     the three networks and their training setup
    results/tables/layers.csv       every watched layer and its size
    results/tables/run_counts.csv   runs by network, phase, and role
    results/tables/compute.csv      training time by network
    results/tables/epochs.csv       every run, every epoch, accuracy and loss
    results/tables/signals.csv.gz   every run, epoch, and layer, all signals
    results/LHD_data.xlsx          all of the above plus the analysis tables
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict

from experiments.evaluate import RESULTS, family, load_runs
from experiments.grid import ARCH_ORDER, dead_layers
from experiments.train import ARCHS, build_model, get_task
from lhd.baseline import SIGNALS

TABLES = RESULTS / "tables"
SHEETS = (
    ("Networks", "networks.csv", "The three networks, datasets, and training settings"),
    ("Layers", "layers.csv", "Every watched layer with its parameter count"),
    ("Run counts", "run_counts.csv", "Runs by network, phase, and role"),
    ("Compute", "compute.csv", "Training and monitoring time by network"),
    ("Headline", "headline.csv", "Main results per network and pooled, with 95% intervals"),
    ("Detectors", "detectors.csv", "LHD against every other detector on the same runs"),
    ("Harm", "harm.csv", "Detection split by how much accuracy each fault cost"),
    ("Per layer", "per_layer.csv", "Detection and diagnosis for every fault and layer"),
    ("Ablation", "ablation.csv", "LHD with signals removed or used alone"),
    ("Settings sweep", "sweep_settings.csv", "Every margin, floor, and patience combination"),
    ("Calibration size", "sweep_calibration.csv", "Bands built from fewer calibration runs"),
    ("Wrong layer", "wrong_layers.csv", "Caught runs where LHD named the wrong layer"),
    ("False alarms", "false_alarms.csv", "Fresh healthy runs LHD flagged"),
    ("Runs", "runs.csv", "One row per scored run"),
    ("Epochs", "epochs.csv", "Accuracy and loss for every run and epoch"),
)


def write_csv(path, rows: list[dict]) -> None:
    """Write rows with the union of their keys as columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def network_rows() -> tuple[list[dict], list[dict]]:
    """One row per network and one row per watched layer."""
    networks, layers = [], []
    for arch in ARCH_ORDER:
        cfg = ARCHS[arch]
        task = get_task(arch)
        model = build_model(arch, task, 0)
        text = hasattr(task, "vocab")
        relu = set(dead_layers(arch))
        networks.append({
            "arch": arch,
            "dataset": {"gtsrb": "GTSRB traffic signs, 32x32 color",
                        "fashion": "Fashion-MNIST, 28x28 grayscale",
                        "shakespeare": "Tiny Shakespeare, characters"}[cfg["dataset"]],
            "task": "next character prediction" if text else "image classification",
            "classes_or_vocab": task.vocab if text else task.classes,
            "train_examples": len(task.train) if text else len(task.train_x),
            "val_examples": len(task.val_x),
            "probe_examples": len(task.probe),
            "parameters": sum(p.numel() for p in model.parameters()),
            "watched_layers": len(model.lhd_layers()),
            "relu_layers": len(relu),
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "steps_per_epoch": cfg.get("steps") or -(-len(task.train_x) // cfg["batch_size"]),
            "context": cfg.get("context"),
            "optimizer": "Adam",
            "learning_rate": cfg["lr"],
            "schedule": "cosine annealing to 0 over the run",
        })
        for name, (module, _, _) in model.lhd_layers().items():
            weight = module.weight
            layers.append({
                "arch": arch, "layer": name, "module": type(module).__name__,
                "weight_shape": "x".join(str(d) for d in weight.shape),
                "parameters": sum(p.numel() for p in module.parameters()),
                "relu_after": name in relu,
            })
    return networks, layers


def run_rows(runs_by_arch: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run counts, compute time, per-epoch rows, and per-layer signal rows."""
    counts, compute, epochs, signals = [], [], [], []
    for arch, runs in runs_by_arch.items():
        tally = Counter((r["metadata"].get("phase", "core"), r["metadata"]["role"]) for r in runs)
        counts += [{"arch": arch, "phase": phase, "role": role, "runs": n}
                   for (phase, role), n in sorted(tally.items())]
        total = sum(r["metadata"]["total_seconds"] for r in runs)
        lhd = sum(r["metadata"]["lhd_seconds"] for r in runs)
        compute.append({"arch": arch, "runs": len(runs), "cpu_threads_per_run": 1,
                        "training_hours": total / 3600, "lhd_hours": lhd / 3600,
                        "mean_minutes_per_run": total / 60 / len(runs)})
        for run in runs:
            meta = run["metadata"]
            fault = meta["fault"]
            base = {"run_id": meta["run_id"], "arch": arch, "phase": meta.get("phase", "core"),
                    "role": meta["role"], "seed": meta["seed"],
                    "family": family(fault) if fault["type"] != "none" else "none",
                    "fault_layer": fault.get("layer")}
            for entry in run["history"]:
                epochs.append({**base, "epoch": entry["epoch"],
                               "train_accuracy": entry["train_accuracy"],
                               "val_accuracy": entry["val_accuracy"],
                               "train_loss": entry["train_loss"], "val_loss": entry["val_loss"],
                               "epoch_seconds": entry["seconds"],
                               "lhd_seconds": entry["lhd_seconds"]})
                for layer, values in entry["layers"].items():
                    signals.append({"run_id": meta["run_id"], "arch": arch, "role": meta["role"],
                                    "family": base["family"], "epoch": entry["epoch"], "layer": layer,
                                    **{s: values[s] for s in (*SIGNALS, "bias_change") if s in values}})
    compute.append({"arch": "all", "runs": sum(c["runs"] for c in compute), "cpu_threads_per_run": 1,
                    "training_hours": sum(c["training_hours"] for c in compute),
                    "lhd_hours": sum(c["lhd_hours"] for c in compute),
                    "mean_minutes_per_run": None})
    return counts, compute, epochs, signals


def headline_rows(summary: dict) -> list[dict]:
    """Main numbers per network and pooled, one metric per row."""
    sections = {**{a: summary["archs"][a] for a in ARCH_ORDER if a in summary["archs"]},
                "pooled": summary["pooled"]}
    rows = defaultdict(dict)
    for name, section in sections.items():
        for part in ("proposal", "healthy"):
            for metric, value in section.get(part, {}).items():
                key = f"{part} {metric}"
                if isinstance(value, dict):
                    rows[key][name] = f"{100 * value['rate']:.1f}% ({value['k']}/{value['n']})"
                    rows[key + " 95% interval"][name] = f"{100 * value['low']:.1f} to {100 * value['high']:.1f}%"
                elif value is not None:
                    rows[key][name] = round(value, 4) if isinstance(value, float) else value
        clustered = section.get("proposal_clustered", {})
        for group in ("by_seed", "by_condition"):
            entry = clustered.get(group)
            if entry:
                rows[f"clustered {group} lhd earlier"][name] = f"{entry['lhd_earlier']} of {entry['groups']}"
                rows[f"clustered {group} wilcoxon p"][name] = entry.get("wilcoxon_p")
        if clustered.get("mean_gap_ci"):
            low, high = clustered["mean_gap_ci"]
            rows["mean epochs earlier, seed bootstrap 95% interval"][name] = f"{low:.2f} to {high:.2f}"
    return [{"metric": metric, **values} for metric, values in rows.items()]


def workbook() -> None:
    """Every table in one Excel file, one sheet each, with an index sheet first."""
    from openpyxl import Workbook

    book = Workbook(write_only=True)
    index = book.create_sheet("About")
    index.append(["LHD study data. Every sheet is also a CSV in results/tables."])
    index.append([])
    index.append(["Sheet", "What it holds", "CSV file"])
    present = [(sheet, name, about) for sheet, name, about in SHEETS if (TABLES / name).exists()]
    for sheet, name, about in present:
        index.append([sheet, about, name])
    for sheet, name, _ in present:
        ws = book.create_sheet(sheet)
        with open(TABLES / name) as f:
            for i, row in enumerate(csv.reader(f)):
                ws.append(row if i == 0 else [_cell(v) for v in row])
    book.save(RESULTS / "LHD_data.xlsx")


def _cell(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value and "e" not in value else number


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    networks, layers = network_rows()
    write_csv(TABLES / "networks.csv", networks)
    write_csv(TABLES / "layers.csv", layers)
    runs_by_arch = {a: load_runs(a) for a in ARCH_ORDER}
    runs_by_arch = {a: r for a, r in runs_by_arch.items() if r}
    counts, compute, epochs, signals = run_rows(runs_by_arch)
    write_csv(TABLES / "run_counts.csv", counts)
    write_csv(TABLES / "compute.csv", compute)
    write_csv(TABLES / "epochs.csv", epochs)
    write_csv(TABLES / "signals.csv.gz", signals)
    with open(RESULTS / "summary.json") as f:
        write_csv(TABLES / "headline.csv", headline_rows(json.load(f)))
    workbook()
    print(f"exported {sum(len(r) for r in runs_by_arch.values())} runs, {len(epochs)} epochs, "
          f"{len(signals)} layer rows", flush=True)


if __name__ == "__main__":
    main()
