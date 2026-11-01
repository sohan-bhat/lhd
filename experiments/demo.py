"""Trains one network live with HALT watching, and stops it when HALT fires.

Usage:
    python -m experiments.demo --arch cnn --fault frozen --layer conv2
    python -m experiments.demo --arch mlp --fault dead --layer fc2 --fraction 0.25
    python -m experiments.demo --arch transformer --fault none

The healthy band comes from the saved calibration runs, so run the
core phase for that architecture first.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from experiments.evaluate import SETTINGS, load_runs, split_runs
from experiments.train import ARCHS, PlantedFault, build_model, get_task
from halt.baseline import healthy_band
from halt.monitor import Monitor
from halt.torch_adapter import TorchAdapter


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", choices=list(ARCHS), default="cnn")
    parser.add_argument("--fault", choices=("none", "frozen", "dead", "detach"), default="frozen")
    parser.add_argument("--layer", default="conv2")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    calibration = split_runs(load_runs(args.arch))["calibration"]
    band = healthy_band([run["history"] for run in calibration])
    cfg = ARCHS[args.arch]
    task = get_task(args.arch)
    model = build_model(args.arch, task, args.seed)
    adapter = TorchAdapter(model, model.halt_layers())
    monitor = Monitor(adapter, task.probe, band, **SETTINGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    spec = {"type": args.fault, "layer": args.layer, "fraction": args.fraction}
    fault = PlantedFault(spec, model, adapter, optimizer, args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    print(f"Training {args.arch} with fault {args.fault}"
          f"{'' if args.fault == 'none' else ' in ' + args.layer}, "
          f"band from {len(calibration)} healthy runs")
    for epoch in range(1, cfg["epochs"] + 1):
        fault.on_epoch_start(epoch)
        start = time.perf_counter()
        monitor.start_epoch()
        correct = total = 0
        for x, y in task.batches(generator):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            task.loss(logits, y).backward()
            monitor.after_batch()
            optimizer.step()
            fault.after_step()
            hit, count = task.hits(logits.detach(), y)
            correct += hit
            total += count
        scheduler.step()
        val_accuracy, _ = task.evaluate(model)
        report = monitor.end_epoch(epoch, correct / total, val_accuracy, time.perf_counter() - start)
        print(f"epoch {epoch:2d}  val accuracy {100 * val_accuracy:5.1f}%")
        if report is not None:
            saved = 100 * (cfg["epochs"] - epoch) / cfg["epochs"]
            print(f"\nHALT stopped the run at epoch {epoch} of {cfg['epochs']} "
                  f"({saved:.0f}% of the compute saved)")
            entry = monitor.recorder.history[-1]["layers"][report["layer"]]
            print(json.dumps({**report, "values": {k: round(v, 5) for k, v in entry.items()
                                                   if v is not None and np.isfinite(v)}}, indent=2))
            return
    print("\nHALT stayed quiet for the whole run")


if __name__ == "__main__":
    main()
