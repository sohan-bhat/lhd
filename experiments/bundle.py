"""Packs saved runs into one compressed file per architecture, and back.

The runner writes one JSON file per run in results/runs/<arch>/.
Those stay local. Git keeps results/runs/<arch>.jsonl.gz, one run
per line, which is a fraction of the size.

Usage:
    python -m experiments.bundle            pack every architecture
    python -m experiments.bundle --unpack   restore the per-run files
"""

from __future__ import annotations

import argparse
import gzip
import json

from experiments.evaluate import RESULTS
from experiments.grid import ARCH_ORDER

RUNS = RESULTS / "runs"


def bundle_path(arch: str):
    return RUNS / f"{arch}.jsonl.gz"


def pack(arch: str) -> int:
    """Write every run file of an architecture into its bundle."""
    paths = sorted((RUNS / arch).glob("*.json"))
    lines = []
    for path in paths:
        with open(path) as run_file:
            lines.append(json.dumps(json.load(run_file), separators=(",", ":")) + "\n")
    with open(bundle_path(arch), "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write("".join(lines).encode())
    return len(paths)


def read_bundle(arch: str) -> list[dict]:
    """Every run stored in an architecture's bundle."""
    with gzip.open(bundle_path(arch), "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def unpack(arch: str) -> int:
    """Write each run in the bundle back to its own file, skipping existing ones."""
    folder = RUNS / arch
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for run in read_bundle(arch):
        path = folder / f"{run['metadata']['run_id']}.json"
        if not path.exists():
            with open(path, "w") as f:
                json.dump(run, f, indent=2)
            written += 1
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unpack", action="store_true")
    args = parser.parse_args(argv)
    for arch in ARCH_ORDER:
        if args.unpack and bundle_path(arch).exists():
            print(f"{arch}: restored {unpack(arch)} run files")
        elif not args.unpack and (RUNS / arch).exists():
            print(f"{arch}: packed {pack(arch)} runs")


if __name__ == "__main__":
    main()
