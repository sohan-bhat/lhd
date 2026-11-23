"""Scores LHD with its fifth signal beside LHD as the study ran it.

Version 2 is the study's detector with never_fired_fraction added,
and nothing else changed. The same 20 calibration runs per network
set every range. The four original signals keep margin 0.5, and
floor and patience stay 0.05 and 3. The new signal's range is widened
by the floor alone (see lhd.detector.signal_margin).

The new signal and its rule were chosen on a pilot: the calibration
runs of the fully connected network and the CNN, and 27 runs on seeds
2000 to 2002, which the study never scored. Both were committed
before the signal was measured on any scored run, and the scored runs
were then scored once.

Usage:
    python -m experiments.v2     print the comparison and write results/v2.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from experiments.evaluate import HARM_BINS, RESULTS, SETTINGS, load_runs, score_arch
from experiments.grid import ARCH_ORDER
from lhd.baseline import SIGNALS, SIGNALS_V2

VERSIONS = {"four signals": SIGNALS, "five signals": SIGNALS_V2}
HARMFUL = HARM_BINS[-1][2]


def is_proposal(row: dict) -> bool:
    """A broken run from the proposal: a frozen layer or dead units from epoch 1."""
    return row["role"] == "fault" and row["fault_type"] in ("frozen", "dead") and row["onset"] == 1


def headline(rows: list[dict]) -> dict:
    """The numbers the write-up leads with, for one set of scored rows."""
    broken = [r for r in rows if is_proposal(r)]
    caught = [r for r in broken if r["detected"]]
    harmful = [r for r in broken if r.get("harm") == HARMFUL]
    healthy = [r for r in rows if r["role"] == "fresh"]
    return {
        "broken": len(broken),
        "caught": len(caught),
        "right_layer": sum(bool(r["layer_ok"]) for r in caught),
        "median_epoch": float(np.median([r["lhd_epoch"] for r in caught])) if caught else None,
        "harmful": len(harmful),
        "harmful_caught": sum(bool(r["detected"]) for r in harmful),
        "healthy": len(healthy),
        "healthy_flagged": sum(bool(r["fired"]) for r in healthy),
    }


def by_condition(rows: list[dict]) -> dict:
    """Caught and total for every fault and layer, as "arch family layer"."""
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["role"] != "fault":
            continue
        cell = groups[f"{r['arch']} {r['family']} {r['fault_layer'] or ''}".strip()]
        cell[0] += bool(r["detected"])
        cell[1] += 1
    return {name: {"caught": caught, "runs": runs} for name, (caught, runs) in sorted(groups.items())}


def compare() -> dict:
    """Both versions scored on every saved run."""
    runs = {arch: load_runs(arch) for arch in ARCH_ORDER}
    result = {"settings": SETTINGS, "versions": {}}
    for name, signals in VERSIONS.items():
        rows = [row for arch in ARCH_ORDER for row in score_arch(arch, runs=runs[arch], signals=signals)]
        result["versions"][name] = {
            "signals": list(signals),
            "all": headline(rows),
            **{arch: headline([r for r in rows if r["arch"] == arch]) for arch in ARCH_ORDER},
            "by_condition": by_condition(rows),
            "flagged_healthy_runs": sorted(r["run_id"] for r in rows if r["role"] == "fresh" and r["fired"]),
        }
    return result


def main() -> None:
    result = compare()
    (RESULTS / "v2.json").write_text(json.dumps(result, indent=1) + "\n")
    for name, version in result["versions"].items():
        print(name)
        for scope in ("all", *ARCH_ORDER):
            h = version[scope]
            print(f"  {scope:12} caught {h['caught']}/{h['broken']} ({h['caught'] / h['broken']:.1%})"
                  f"  right layer {h['right_layer']}/{h['caught']}"
                  f"  lost 2 pts or more {h['harmful_caught']}/{h['harmful']}"
                  f"  healthy flagged {h['healthy_flagged']}/{h['healthy']}")
    four = result["versions"]["four signals"]["by_condition"]
    five = result["versions"]["five signals"]["by_condition"]
    print("conditions that changed")
    for name, cell in five.items():
        if cell["caught"] != four[name]["caught"]:
            print(f"  {name:36} {four[name]['caught']} -> {cell['caught']} of {cell['runs']}")


if __name__ == "__main__":
    main()
