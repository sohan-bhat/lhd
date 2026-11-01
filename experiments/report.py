"""Writes results/REPORT.md from results/summary.json and the tables.

Usage:
    python -m experiments.report

Every number in the report is read from the analysis output, so the
report can be rebuilt whenever more runs finish.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from experiments.evaluate import RESULTS
from experiments.figures import ARCH_NAMES, FAMILY_NAMES
from experiments.grid import ARCH_ORDER
from experiments.train import ARCHS

TABLES = RESULTS / "tables"


def pct(stat: dict | float | None, digits: int = 0) -> str:
    """A rate as a percent, with k/n and a 95% interval when given a rate dict."""
    if stat is None:
        return "n/a"
    if isinstance(stat, dict):
        if not stat["n"]:
            return "n/a"
        return (f"{100 * stat['rate']:.{digits}f}% ({stat['k']}/{stat['n']}, "
                f"95% CI {100 * stat['low']:.0f} to {100 * stat['high']:.0f}%)")
    return f"{100 * stat:.{digits}f}%"


def short(stat: dict | None) -> str:
    """A rate dict as percent and k/n."""
    if not stat or not stat["n"]:
        return "n/a"
    return f"{100 * stat['rate']:.0f}% ({stat['k']}/{stat['n']})"


def num(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pvalue(p) -> str:
    if p is None:
        return "n/a"
    return "< 1e-10" if p < 1e-10 else f"{p:.2g}"


def read(name: str) -> list[dict]:
    path = TABLES / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def headline_table(summary: dict) -> list[str]:
    archs = [a for a in ARCH_ORDER if a in summary["archs"]]
    lines = ["| | " + " | ".join(ARCH_NAMES[a] for a in archs) + " |",
             "|---|" + "---|" * len(archs)]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(summary["archs"][a]) for a in archs) + " |")

    row("Training runs", lambda s: str(sum(s["runs"].values())))
    row("Epochs per run", lambda s: str(ARCHS[_arch_of(summary, s)]["epochs"]))
    row("Broken runs (frozen and dead units)", lambda s: str(s["proposal"].get("runs", 0)))
    row("Caught by HALT", lambda s: short(s["proposal"].get("detected")))
    row("Right layer, when caught", lambda s: short(s["proposal"].get("layer_ok")))
    row("Right cause, when caught", lambda s: short(s["proposal"].get("cause_ok")))
    row("Median epoch HALT stops", lambda s: num(s["proposal"].get("halt_median_epoch"), 0))
    row("Caught by the accuracy curve", lambda s: short(s["proposal"].get("acc_detected")))
    row("Median epoch accuracy notices", lambda s: num(s["proposal"].get("acc_median_epoch"), 0))
    row("HALT first", lambda s: short(s["proposal"].get("halt_first")))
    row("Final accuracy still inside healthy range", lambda s: short(s["proposal"].get("silent_share")))
    row("Compute saved per broken run", lambda s: pct(s["proposal"].get("mean_compute_saved")))
    row("Fresh healthy runs", lambda s: str(s["healthy"]["runs"]))
    row("HALT false alarms", lambda s: short(s["healthy"]["halt_false_alarm"]))
    row("Accuracy curve false alarms", lambda s: short(s["healthy"]["acc_false_alarm"]))
    row("HALT time overhead", lambda s: pct(s["healthy"]["overhead"], 1))
    row("Fault conditions where HALT was earlier on average",
        lambda s: _groups(s.get("proposal_clustered", {}).get("by_condition")))
    row("Seeds where HALT was earlier on average",
        lambda s: _groups(s.get("proposal_clustered", {}).get("by_seed")))
    row("Mean epochs earlier, 95% seed bootstrap",
        lambda s: _gap(s["proposal"].get("mean_gap"), s.get("proposal_clustered", {}).get("mean_gap_ci")))
    return lines


def _groups(entry: dict | None) -> str:
    """Groups where HALT was earlier, with the one-sided Wilcoxon p on group means."""
    if not entry:
        return "n/a"
    return (f"{entry['halt_earlier']} of {entry['groups']} "
            f"(p {pvalue(entry.get('wilcoxon_p'))})")


def _gap(mean, ci) -> str:
    if mean is None or not ci:
        return "n/a"
    return f"{mean:.1f} ({ci[0]:.1f} to {ci[1]:.1f})"


def _arch_of(summary: dict, section: dict) -> str:
    return next(a for a, s in summary["archs"].items() if s is section)


def family_table(summary: dict) -> list[str]:
    lines = ["| Architecture | Fault | Runs | Caught | Right layer | Right cause | "
             "HALT median epoch | Accuracy caught | Accuracy median epoch | Mean accuracy lost |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arch in [a for a in ARCH_ORDER if a in summary["archs"]]:
        for fam, s in summary["archs"][arch]["families"].items():
            if not s:
                continue
            drop = s.get("mean_acc_drop_vs_twin")
            lines.append(
                f"| {ARCH_NAMES[arch]} | {FAMILY_NAMES.get(fam, fam)} | {s['runs']} | "
                f"{short(s['detected'])} | {short(s['layer_ok'])} | {short(s['cause_ok'])} | "
                f"{num(s['halt_median_epoch'], 0)} | {short(s['acc_detected'])} | "
                f"{num(s['acc_median_epoch'], 0)} | "
                f"{'n/a' if drop is None else f'{100 * drop:.1f} pts'} |")
    return lines


def ablation_table() -> list[str]:
    rows = read("ablation.csv")
    if not rows:
        return []
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    by = {(r["arch"], r["variant"]): r for r in rows}
    lines = ["| Signals used | " + " | ".join(
        f"{ARCH_NAMES[a]}: caught / right layer and cause / false alarms" for a in archs) + " |",
        "|---|" + "---|" * len(archs)]
    for v in variants:
        cells = []
        for a in archs:
            r = by[(a, v)]
            cells.append(f"{100 * float(r['detection']):.0f}% / {100 * float(r['both_ok']):.0f}% / "
                         f"{100 * float(r['false_alarm']):.0f}%")
        lines.append(f"| {v.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return lines


def calibration_table() -> list[str]:
    rows = read("sweep_calibration.csv")
    if not rows:
        return []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arch"], int(r["k"]))].append(r)
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in groups)]
    sizes = sorted({k[1] for k in groups})
    lines = ["| Calibration runs | " + " | ".join(
        f"{ARCH_NAMES[a]}: false alarms / caught" for a in archs) + " |",
        "|---|" + "---|" * len(archs)]
    for k in sizes:
        cells = []
        for a in archs:
            g = groups[(a, k)]
            fa = sum(float(r["false_alarm"]) for r in g) / len(g)
            det = sum(float(r["detection"]) for r in g) / len(g)
            cells.append(f"{100 * fa:.0f}% / {100 * det:.0f}%")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return lines


def misses() -> list[str]:
    lines = []
    wrong = read("wrong_layers.csv")
    alarms = read("false_alarms.csv")
    causes = [r for r in read("runs.csv") if r["role"] == "fault" and r["detected"] == "True"
              and r["layer_ok"] == "True" and r["cause_ok"] == "False"]
    missed = defaultdict(list)
    for r in read("runs.csv"):
        if r["role"] == "fault" and r["detected"] == "False":
            missed[(r["arch"], r["family"], r["fault_layer"] or "all")].append(r["seed"])
    if missed:
        lines += ["", f"Broken runs HALT never caught ({sum(len(v) for v in missed.values())}), "
                  "grouped by fault and layer:", "",
                  "| Architecture | Fault | Layer | Seeds missed |", "|---|---|---|---|"]
        lines += [f"| {ARCH_NAMES[a]} | {FAMILY_NAMES.get(f, f)} | {layer} | {', '.join(seeds)} |"
                  for (a, f, layer), seeds in sorted(missed.items())]
    if not (wrong or alarms or causes or missed):
        return ["", "None. Every caught run named the right layer and cause, "
                "and no fresh healthy run was flagged."]
    if wrong:
        lines += ["", f"Runs where HALT named the wrong layer ({len(wrong)}):", "",
                  "| Run | Planted in | HALT named | Cause named | Signals |", "|---|---|---|---|---|"]
        lines += [f"| {r['run_id']} | {r['fault_layer']} | {r['halt_layer']} | {r['halt_cause']} | "
                  f"{r['halt_signals']} |" for r in wrong]
    if causes:
        lines += ["", f"Runs where HALT found the right layer but named the wrong cause ({len(causes)}):", "",
                  "| Run | Planted | HALT named | Signals | Epoch |", "|---|---|---|---|---|"]
        lines += [f"| {r['run_id']} | {r['fault_type']} | {r['halt_cause']} | {r['halt_signals']} | "
                  f"{r['halt_epoch']} |" for r in causes]
    if alarms:
        lines += ["", f"False alarms on fresh healthy runs ({len(alarms)}):", "",
                  "| Run | Epoch | Layer | Cause named | Signals |", "|---|---|---|---|---|"]
        lines += [f"| {r['run_id']} | {r['halt_epoch']} | {r['halt_layer']} | {r['halt_cause']} | "
                  f"{r['halt_signals']} |" for r in alarms]
    return lines


def main() -> None:
    with open(RESULTS / "summary.json") as f:
        summary = json.load(f)
    settings = summary["settings"]
    pooled = summary["pooled"]
    tests = pooled["proposal_tests"]
    clustered = pooled.get("proposal_clustered", {})
    total_runs = sum(sum(s["runs"].values()) for s in summary["archs"].values())
    figures = sorted(p.name for p in (RESULTS / "figures").glob("*.png"))
    out = [
        "# HALT results",
        "",
        "Generated by `python -m experiments.report` from "
        f"{total_runs} training runs. Settings were fixed before any broken run was trained: "
        f"margin {settings['margin']}, floor {settings['floor']}, patience {settings['patience']}. "
        "The healthy band uses calibration seeds 0 to 19 only. Fresh healthy runs and broken runs "
        "use seeds from 1000 up and never touch the band.",
        "",
        "## Pooled over all three architectures",
        "",
        f"- Broken runs with a frozen layer or dead units: {pooled['proposal'].get('runs', 0)}",
        f"- Caught by HALT: {pct(pooled['proposal'].get('detected'), 1)}",
        f"- Right layer when caught: {pct(pooled['proposal'].get('layer_ok'), 1)}",
        f"- Right cause when caught: {pct(pooled['proposal'].get('cause_ok'), 1)}",
        f"- Caught by the accuracy curve at any point: {pct(pooled['proposal'].get('acc_detected'), 1)}",
        f"- HALT flagged the run before the accuracy curve did: {pct(pooled['proposal'].get('halt_first'), 1)}",
        f"- Of the runs HALT caught, the accuracy curve never noticed: {pct(pooled.get('acc_never_on_detected'), 1)}",
        f"- Mean compute saved per broken run: {pct(pooled['proposal'].get('mean_compute_saved'), 1)}",
        f"- HALT false alarms on fresh healthy runs: {pct(pooled['healthy']['halt_false_alarm'], 1)}",
        f"- Accuracy curve false alarms on the same runs: {pct(pooled['healthy']['acc_false_alarm'], 1)}",
        f"- Mean epochs HALT was earlier than the accuracy curve, with \"never\" counted as one "
        f"epoch past the end: {num(pooled['proposal'].get('mean_gap'))}, 95% interval "
        f"{_gap(pooled['proposal'].get('mean_gap'), clustered.get('mean_gap_ci')).split(' ', 1)[-1]} "
        f"from a bootstrap that resamples whole seeds.",
        f"- Fault conditions (fault and layer) where HALT was earlier on average: "
        f"{_groups(clustered.get('by_condition'))}, one-sided Wilcoxon on condition means.",
        f"- Seeds where HALT was earlier on average: {_groups(clustered.get('by_seed'))}.",
        f"- Run by run, for reference only since runs that share a seed are not independent: "
        f"HALT earlier on {tests.get('halt_earlier')}, accuracy earlier on {tests.get('acc_earlier')}, "
        f"ties {tests.get('ties')}.",
        "",
        "## By architecture",
        "",
        *headline_table(summary),
        "",
        "## By fault",
        "",
        *family_table(summary),
        "",
        "## Each signal's part",
        "",
        "Each cell shows broken runs caught, then runs where HALT named the right layer and cause, "
        "then fresh healthy runs flagged.",
        "",
        *ablation_table(),
        "",
        "## How many healthy runs the band needs",
        "",
        "Bands built from random subsets of the 20 calibration runs, 20 draws per size.",
        "",
        *calibration_table(),
        "",
        "## Misses to read by hand",
        *misses(),
        "",
        "## Figures",
        "",
        *[f"![{name}](figures/{name})" for name in figures],
        "",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
