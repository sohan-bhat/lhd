"""Trains one network, with or without a planted fault, and saves
HALT's signals for every epoch.

Runs always finish every epoch, even when broken, so the saved
history can be replayed later through any detector setting and
compared with the accuracy curve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.data import load_task
from experiments.models import CNN, MLP, TinyTransformer
from halt.faults import FrozenLayer, kill_units
from halt.recorder import Recorder
from halt.torch_adapter import TorchAdapter

ARCHS = {
    "cnn": {"dataset": "gtsrb", "epochs": 15, "batch_size": 128, "lr": 1e-3,
            "val_size": 4000},
    "mlp": {"dataset": "fashion", "epochs": 20, "batch_size": 128, "lr": 1e-3},
    "transformer": {"dataset": "shakespeare", "epochs": 15, "batch_size": 32,
                    "lr": 1e-3, "context": 64, "steps": 100},
}

DEAD_BIAS = -1.0

_TASKS = {}


def get_task(arch: str):
    """The architecture's task, loaded once per process."""
    if arch not in _TASKS:
        cfg = ARCHS[arch]
        _TASKS[arch] = load_task(cfg["dataset"], cfg["batch_size"],
                                 cfg.get("context", 64), cfg.get("steps", 150),
                                 cfg.get("val_size"))
    return _TASKS[arch]


def build_model(arch: str, task, seed: int) -> torch.nn.Module:
    """A freshly initialized network, with weights set by the seed."""
    torch.manual_seed(seed)
    if arch == "cnn":
        return CNN(task.classes)
    if arch == "mlp":
        return MLP(task.train_x.shape[1], task.classes)
    if arch == "transformer":
        return TinyTransformer(task.vocab, task.context)
    raise KeyError(arch)


class PlantedFault:
    """Applies one fault from a spec at the right moments of training.

    Spec types: none; frozen (layer); dead (layer, fraction);
    detach (module path whose input loses its gradient); lr (factor).
    Every type takes an onset epoch, 1 by default.
    """

    def __init__(self, spec: dict, model, adapter: TorchAdapter, optimizer, seed: int):
        self.spec = spec
        self.kind = spec["type"]
        self.onset = spec.get("onset", 1)
        self.model = model
        self.adapter = adapter
        self.optimizer = optimizer
        self.rng = np.random.default_rng(seed + 7919)
        self._frozen = None
        self.units = None

    def on_epoch_start(self, epoch: int) -> None:
        """Switch the fault on when its onset epoch begins."""
        if epoch != self.onset:
            return
        if self.kind == "frozen":
            self._frozen = FrozenLayer(self.adapter, self.spec["layer"])
        elif self.kind == "dead":
            self._kill()
        elif self.kind == "detach":
            module = self.model.get_submodule(self.spec["layer"])
            module.register_forward_pre_hook(_detach_inputs)
        elif self.kind == "lr":
            for group in self.optimizer.param_groups:
                group["lr"] *= self.spec["factor"]

    def _kill(self) -> None:
        """Kill a random fraction of the layer's units, chosen by the seed.

        Killed units get zero incoming weights and a bias of -1, so
        they output 0 for every input. A bias of -1 rather than -10
        keeps the layer's weight norm close to normal, so the fault
        shows up only through the signals meant to catch it.
        """
        name = self.spec["layer"]
        count = self.adapter.parameters(name)["weights"].shape[0]
        n = max(1, int(round(self.spec["fraction"] * count)))
        self.units = sorted(int(u) for u in self.rng.choice(count, size=n, replace=False))
        kill_units(self.adapter, name, self.units, weight_axis=0, bias_value=DEAD_BIAS)
        self._clear_momentum(name)

    def _clear_momentum(self, name: str) -> None:
        """Zero Adam's running averages for the killed units.

        A fault planted mid-run would otherwise keep moving the killed
        weights on leftover momentum and could bring units back to life.
        """
        module = self.adapter.layers[name][0]
        for param in (module.weight, module.bias):
            state = self.optimizer.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][self.units] = 0.0

    def after_step(self) -> None:
        """Undo any update to a frozen layer."""
        if self._frozen is not None:
            self._frozen.restore()


def _detach_inputs(module, args):
    """Forward pre-hook that cuts the gradient path through a module's inputs."""
    return tuple(a.detach() for a in args)


def train_run(spec: dict, out_path: str | Path, threads: int = 1) -> dict:
    """Train one run described by spec and save it as JSON.

    Parameters
    ----------
    spec : dict
        {"run_id", "arch", "seed", "role", "fault": {...}}.
    out_path : str or Path
        Where the run's JSON is written.
    threads : int
        Torch CPU threads for this run.

    Returns
    -------
    dict
        The metadata saved with the run.
    """
    torch.set_num_threads(threads)
    arch, seed = spec["arch"], spec["seed"]
    cfg = ARCHS[arch]
    task = get_task(arch)
    model = build_model(arch, task, seed)
    adapter = TorchAdapter(model, model.halt_layers())
    recorder = Recorder(adapter, task.probe)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    fault = PlantedFault(spec["fault"], model, adapter, optimizer, seed)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    generator = torch.Generator().manual_seed(seed)

    run_start = time.perf_counter()
    halt_total = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        fault.on_epoch_start(epoch)
        epoch_start = time.perf_counter()
        tick = time.perf_counter()
        recorder.start_epoch()
        halt_seconds = time.perf_counter() - tick
        correct = total = 0
        loss_sum = 0.0
        model.train()
        for x, y in task.batches(generator):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = task.loss(logits, y)
            loss.backward()
            tick = time.perf_counter()
            recorder.after_batch()
            halt_seconds += time.perf_counter() - tick
            optimizer.step()
            fault.after_step()
            hit, count = task.hits(logits.detach(), y)
            correct += hit
            total += count
            loss_sum += float(loss) * count
        scheduler.step()
        val_accuracy, val_loss = task.evaluate(model)
        seconds = time.perf_counter() - epoch_start
        tick = time.perf_counter()
        entry = recorder.end_epoch(epoch, correct / total, val_accuracy, seconds)
        halt_seconds += time.perf_counter() - tick
        entry["train_loss"] = loss_sum / total
        entry["val_loss"] = val_loss
        entry["halt_seconds"] = halt_seconds
        halt_total += halt_seconds

    metadata = {
        **spec,
        "config": cfg,
        "killed_units": fault.units,
        "layers": adapter.layer_names(),
        "total_seconds": time.perf_counter() - run_start,
        "halt_seconds": halt_total,
        "torch": torch.__version__,
        "threads": threads,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    recorder.save(str(tmp), metadata)
    tmp.replace(out_path)
    return metadata


def load_run(path: str | Path) -> dict:
    """Read a saved run: {"metadata": ..., "history": [...]}."""
    with open(path) as f:
        return json.load(f)
