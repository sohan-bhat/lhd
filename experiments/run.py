"""Runs the study's training runs in parallel, skipping finished ones.

Usage:
    python -m experiments.run --phase core --workers 5
    python -m experiments.run --phase more --arch mlp --workers 5
    python -m experiments.run --phase seeds --workers 5
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.grid import ARCH_ORDER, build_grid

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run_path(spec: dict) -> Path:
    """Where a run's JSON lives."""
    return RESULTS / "runs" / spec["arch"] / f"{spec['run_id']}.json"


def _work(spec: dict) -> tuple[str, float, float]:
    """Train one run in a worker process."""
    from experiments.train import train_run

    start = time.perf_counter()
    meta = train_run(spec, run_path(spec))
    return spec["run_id"], time.perf_counter() - start, meta["total_seconds"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("core", "more", "seeds"), required=True)
    parser.add_argument("--arch", choices=ARCH_ORDER, action="append")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    specs = build_grid(args.phase, tuple(args.arch or ARCH_ORDER))
    todo = [s for s in specs if not run_path(s).exists()]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(specs)} runs in phase {args.phase}, {len(todo)} to train", flush=True)

    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(_work, spec) for spec in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                run_id, seconds, _ = future.result()
            except Exception as error:
                print(f"[{done}/{len(todo)}] FAILED {error!r}", flush=True)
                continue
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (len(todo) - done)
            print(f"[{done}/{len(todo)}] {run_id} {seconds:.0f}s  "
                  f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
