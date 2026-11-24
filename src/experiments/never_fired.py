"""Measures the fifth signal, never_fired_fraction, for every run in the study.

The saved runs hold four signals per layer. This trains each run again
from its spec with the recorder that also counts units that have never
fired, checks that everything saved before comes out exactly the same,
and keeps the new signal.

A run with no dead unit before its first batch is not trained again.
A unit that has fired once never counts, so the signal is 0 for every
epoch of such a run. One seed in ten of those runs is trained anyway,
as a check that it really is 0.

Usage:
    python -m experiments.never_fired --workers 8    measure what is missing
    python -m experiments.never_fired --pack         write results/runs/<arch>.never_fired.json.gz
                                                     and results/never_fired_counts.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.evaluate import RESULTS, never_fired_path
from experiments.grid import ARCH_ORDER, build_grid
from lhd.baseline import NEVER_FIRED, SIGNALS

OUT = RESULTS / "never_fired"
COUNTS = RESULTS / "never_fired_counts.json"
PHASES = ("core", "seeds", "more")
CHECK_EVERY = 10
CURVES = ("train_accuracy", "val_accuracy", "train_loss", "val_loss")

_SAVED = {}


def all_specs(archs=ARCH_ORDER) -> list[dict]:
    """Every run in the study, one architecture at a time."""
    return [spec for arch in archs for phase in PHASES for spec in build_grid(phase, (arch,))]


def out_path(spec: dict) -> Path:
    """Where one run's measurement is kept until it is packed."""
    return OUT / spec["arch"] / f"{spec['run_id']}.json"


def born_dead(spec: dict) -> dict[str, int]:
    """How many units of each layer are dead before the first batch,
    with the run's fault planted if it starts at epoch 1."""
    import torch

    from experiments.train import ARCHS, PlantedFault, build_model, get_task
    from lhd.signals import dead_units
    from lhd.torch_adapter import TorchAdapter

    task = get_task(spec["arch"])
    model = build_model(spec["arch"], task, spec["seed"])
    adapter = TorchAdapter(model, model.lhd_layers())
    optimizer = torch.optim.Adam(model.parameters(), lr=ARCHS[spec["arch"]]["lr"])
    PlantedFault(spec["fault"], model, adapter, optimizer, spec["seed"]).on_epoch_start(1)
    outs = adapter.probe_outputs(task.probe)
    return {name: int(dead_units(outs[name], adapter.unit_axis(name)).sum())
            for name in adapter.layer_names()}


def saved_run(spec: dict) -> dict:
    """The run as the study saved it."""
    from experiments.bundle import read_bundle

    arch = spec["arch"]
    if arch not in _SAVED:
        _SAVED[arch] = {run["metadata"]["run_id"]: run for run in read_bundle(arch)}
    return _SAVED[arch][spec["run_id"]]


def same_as_saved(history: list[dict], saved: list[dict]) -> bool:
    """Whether a retrained history repeats the saved one exactly:
    both accuracies, both losses, and every layer's saved signals."""
    if len(history) != len(saved):
        return False
    for new, old in zip(history, saved):
        if any(new[key] != old[key] for key in CURVES):
            return False
        for layer, values in old["layers"].items():
            if any(new["layers"][layer][s] != values[s] for s in (*SIGNALS, "bias_change")):
                return False
    return True


def measure(spec: dict) -> dict:
    """Measure one run and write the result. Runs in a worker process."""
    import torch

    from experiments.train import load_run, train_run

    # One thread from the very first operation, as in the study. With
    # more, sums are added in another order and the run stops matching.
    torch.set_num_threads(1)
    start = time.perf_counter()
    dead = born_dead(spec)
    result = {"run_id": spec["run_id"], "born_dead": dead, "trained": False, "never_fired": {}}
    if any(dead.values()) or spec["seed"] % CHECK_EVERY == 0:
        path = out_path(spec).with_suffix(".run")
        train_run(spec, path)
        history = load_run(path)["history"]
        path.unlink()
        result["trained"] = True
        result["same_as_saved"] = same_as_saved(history, saved_run(spec)["history"])
        result["never_fired"] = {layer: [entry["layers"][layer][NEVER_FIRED] for entry in history]
                                 for layer in history[0]["layers"]}
    result["seconds"] = time.perf_counter() - start
    target = out_path(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(target)
    return result


def pack(arch: str) -> dict:
    """Write an architecture's measurements into one file for git.

    Layers whose value is 0 for every epoch are left out. Raises if a
    run is missing, if a retrained run does not repeat the saved one,
    or if a run that was not expected to have the signal has it.
    """
    stored, counts = {}, {"runs": 0, "trained": 0, "trained_as_a_check": 0}
    for spec in all_specs((arch,)):
        path = out_path(spec)
        if not path.exists():
            raise FileNotFoundError(f"{spec['run_id']} has not been measured")
        result = json.loads(path.read_text())
        counts["runs"] += 1
        if result["trained"]:
            counts["trained"] += 1
            if not result["same_as_saved"]:
                raise ValueError(f"{spec['run_id']} did not repeat the saved run")
            if not any(result["born_dead"].values()):
                counts["trained_as_a_check"] += 1
                if any(any(values) for values in result["never_fired"].values()):
                    raise ValueError(f"{spec['run_id']} had no dead unit at the start but a signal above 0")
        stored[spec["run_id"]] = {layer: values for layer, values in result["never_fired"].items()
                                  if any(values)}
    with open(never_fired_path(arch), "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(json.dumps(stored, sort_keys=True, separators=(",", ":")).encode())
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", choices=ARCH_ORDER, action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args(argv)
    archs = tuple(args.arch or ARCH_ORDER)

    if args.pack:
        counts = {arch: pack(arch) for arch in archs}
        for arch, count in counts.items():
            print(arch, count)
        if archs == tuple(ARCH_ORDER):
            COUNTS.write_text(json.dumps(counts, indent=1) + "\n")
        return

    specs = all_specs(archs)
    todo = [spec for spec in specs if not out_path(spec).exists()]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(specs)} runs, {len(todo)} to measure", flush=True)

    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(measure, spec) for spec in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as error:
                print(f"[{done}/{len(todo)}] FAILED {error!r}", flush=True)
                continue
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (len(todo) - done)
            note = ("same as saved" if result["same_as_saved"] else "DIFFERENT FROM SAVED") \
                if result["trained"] else "no dead unit at the start"
            print(f"[{done}/{len(todo)}] {result['run_id']} {note} {result['seconds']:.0f}s  "
                  f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
