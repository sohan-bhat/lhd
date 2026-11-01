"""Every training run in the study, as a list of specs.

Seeds are split so calibration and evaluation never overlap:
calibration runs use seeds 0-19 and set the healthy band. Fresh
healthy runs and broken runs use seeds from 1000 up, so each broken
run has a healthy twin with the same starting weights and data order.

Phase "core" is the proposal: frozen layers and dead units planted
at the start. Phase "more" adds fresh healthy runs and faults beyond
the proposal: faults that start mid-run, cut gradients, or raise the
learning rate. Phase "seeds" repeats the core faults on five more seeds.
"""

from __future__ import annotations

from experiments.models import CNN, MLP, TinyTransformer, relu_layers

CALIBRATION_SEEDS = list(range(0, 20))
FRESH_SEEDS = {"core": list(range(1000, 1030)), "more": list(range(1030, 1050))}
FAULT_SEEDS = {"core": list(range(1000, 1005)), "more": [], "seeds": list(range(1005, 1010))}
DEAD_FRACTIONS = (0.1, 0.25, 0.5)
LATE_ONSET = 6
EXTRA_SEEDS = list(range(1000, 1005))
DETACH_CUTS = {
    "cnn": ["conv3", "fc1"],
    "mlp": ["fc2", "fc4"],
    "transformer": ["blocks.1", "head"],
}
ARCH_ORDER = ("mlp", "transformer", "cnn")


def watched_layers(arch: str) -> list[str]:
    """Watched layer names for an architecture, in data-flow order."""
    return list(_shape_model(arch).halt_layers())


def dead_layers(arch: str) -> list[str]:
    """Watched layers that end in a ReLU."""
    return relu_layers(_shape_model(arch))


def _shape_model(arch: str):
    """A throwaway model with the right layout, only used for layer names."""
    if arch == "cnn":
        return CNN(43)
    if arch == "mlp":
        return MLP(784, 10)
    return TinyTransformer(65, 64)


def cut_affects(arch: str, cut: str) -> list[str]:
    """Watched layers upstream of a gradient cut, which stop learning."""
    layers = watched_layers(arch)
    if arch == "transformer":
        if cut == "head":
            return layers[:-1]
        block = int(cut.split(".")[1])
        return [name for name in layers
                if name in ("tok", "pos") or (name.startswith("b") and int(name[1]) < block)]
    return layers[:layers.index(cut)]


def _spec(arch: str, role: str, seed: int, fault: dict, tag: str, phase: str) -> dict:
    return {"run_id": f"{arch}-{role}-{tag}-s{seed}", "arch": arch, "seed": seed,
            "role": role, "phase": phase, "fault": fault}


def build_grid(phase: str, archs=ARCH_ORDER) -> list[dict]:
    """All run specs for a phase, one architecture at a time,
    calibration runs first within each."""
    specs = []
    healthy = {"type": "none"}
    for arch in archs:
        if phase == "core":
            for seed in CALIBRATION_SEEDS:
                specs.append(_spec(arch, "calibration", seed, healthy, "none", phase))
        for seed in FRESH_SEEDS.get(phase, []):
            specs.append(_spec(arch, "fresh", seed, healthy, "none", phase))
        for seed in FAULT_SEEDS[phase]:
            for layer in watched_layers(arch):
                fault = {"type": "frozen", "layer": layer, "onset": 1}
                specs.append(_spec(arch, "fault", seed, fault, f"frozen-{layer}", phase))
            for layer in dead_layers(arch):
                for fraction in DEAD_FRACTIONS:
                    fault = {"type": "dead", "layer": layer, "fraction": fraction, "onset": 1}
                    tag = f"dead{int(fraction * 100)}-{layer}"
                    specs.append(_spec(arch, "fault", seed, fault, tag, phase))
        if phase == "more":
            specs.extend(_extended(arch, phase))
    return specs


def _extended(arch: str, phase: str) -> list[dict]:
    """Faults beyond the proposal: late onset, gradient cuts, and a learning rate jump."""
    specs = []
    for seed in EXTRA_SEEDS:
        for layer in watched_layers(arch):
            fault = {"type": "frozen", "layer": layer, "onset": LATE_ONSET}
            specs.append(_spec(arch, "fault", seed, fault, f"latefrozen-{layer}", phase))
        for layer in dead_layers(arch):
            fault = {"type": "dead", "layer": layer, "fraction": 0.5, "onset": LATE_ONSET}
            specs.append(_spec(arch, "fault", seed, fault, f"latedead50-{layer}", phase))
        for cut in DETACH_CUTS[arch]:
            fault = {"type": "detach", "layer": cut, "onset": 1}
            specs.append(_spec(arch, "fault", seed, fault, f"detach-{cut}", phase))
        for factor in (10, 100):
            fault = {"type": "lr", "factor": factor, "onset": LATE_ONSET}
            specs.append(_spec(arch, "fault", seed, fault, f"lr{factor}", phase))
    return specs
