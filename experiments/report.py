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
    row("Caught by LHD", lambda s: short(s["proposal"].get("detected")))
    row("Right layer, when caught", lambda s: short(s["proposal"].get("layer_ok")))
    row("Right cause, when caught", lambda s: short(s["proposal"].get("cause_ok")))
    row("Median epoch LHD stops", lambda s: num(s["proposal"].get("lhd_median_epoch"), 0))
    row("Caught by the accuracy curve", lambda s: short(s["proposal"].get("acc_detected")))
    row("Median epoch accuracy notices", lambda s: num(s["proposal"].get("acc_median_epoch"), 0))
    row("LHD first", lambda s: short(s["proposal"].get("lhd_first")))
    row("Final accuracy still inside healthy range", lambda s: short(s["proposal"].get("silent_share")))
    row("Compute saved per broken run", lambda s: pct(s["proposal"].get("mean_compute_saved")))
    row("Fresh healthy runs", lambda s: str(s["healthy"]["runs"]))
    row("LHD false alarms", lambda s: short(s["healthy"]["lhd_false_alarm"]))
    row("Accuracy curve false alarms", lambda s: short(s["healthy"]["acc_false_alarm"]))
    row("LHD time overhead", lambda s: pct(s["healthy"]["overhead"], 1))
    row("Fault conditions where LHD was earlier on average",
        lambda s: _groups(s.get("proposal_clustered", {}).get("by_condition")))
    row("Seeds where LHD was earlier on average",
        lambda s: _groups(s.get("proposal_clustered", {}).get("by_seed")))
    row("Mean epochs earlier, 95% seed bootstrap",
        lambda s: _gap(s["proposal"].get("mean_gap"), s.get("proposal_clustered", {}).get("mean_gap_ci")))
    return lines


def _groups(entry: dict | None) -> str:
    """Groups where LHD was earlier, with the one-sided Wilcoxon p on group means."""
    if not entry:
        return "n/a"
    return (f"{entry['lhd_earlier']} of {entry['groups']} "
            f"(p {pvalue(entry.get('wilcoxon_p'))})")


def _gap(mean, ci) -> str:
    if mean is None or not ci:
        return "n/a"
    return f"{mean:.1f} ({ci[0]:.1f} to {ci[1]:.1f})"


def _arch_of(summary: dict, section: dict) -> str:
    return next(a for a, s in summary["archs"].items() if s is section)


def family_table(summary: dict) -> list[str]:
    lines = ["| Architecture | Fault | Runs | Caught | Right layer | Right cause | "
             "LHD median epoch | Accuracy caught | Accuracy median epoch | Mean accuracy lost |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arch in [a for a in ARCH_ORDER if a in summary["archs"]]:
        for fam, s in summary["archs"][arch]["families"].items():
            if not s:
                continue
            drop = s.get("mean_acc_drop_vs_twin")
            lines.append(
                f"| {ARCH_NAMES[arch]} | {FAMILY_NAMES.get(fam, fam)} | {s['runs']} | "
                f"{short(s['detected'])} | {short(s['layer_ok'])} | {short(s['cause_ok'])} | "
                f"{num(s['lhd_median_epoch'], 0)} | {short(s['acc_detected'])} | "
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


DEAD_FAMILIES = ("dead10", "dead25", "dead50")


def _sum_rows(rows: list[dict], families) -> dict:
    """Caught counts summed over several fault families of one detector."""
    group = [r for r in rows if r["family"] in families]
    if not group:
        return {}
    caught = sum(int(r["caught"]) for r in group)
    n = sum(int(r["broken_runs"]) for r in group)
    return {"k": caught, "n": n, "rate": caught / n}


def _kn(stat: dict) -> str:
    return f"{100 * stat['rate']:.0f}% ({stat['k']}/{stat['n']})" if stat and stat["n"] else "n/a"


def detector_section() -> list[str]:
    """LHD against the accuracy and loss curves and against simple fixed checks, pooled."""
    rows = [r for r in read("detectors.csv") if r["arch"] == "pooled"]
    if not rows:
        return []
    names = list(dict.fromkeys(r["detector"] for r in rows))
    lines = ["| Detector | Frozen layers caught | Dead units caught | Median alarm epoch | "
             "Earlier than LHD / later / tie | Healthy runs flagged |",
             "|---|---|---|---|---|---|"]
    for name in names:
        mine = [r for r in rows if r["detector"] == name]
        proposal = next((r for r in mine if r["family"] == "proposal"), None)
        frozen = _sum_rows(mine, ("frozen",))
        dead = _sum_rows(mine, DEAD_FAMILIES)
        if proposal is None:
            continue
        timing = ("n/a" if name == "LHD" else
                  f"{proposal['this_earlier']} / {proposal['lhd_earlier']} / {proposal['ties']}")
        alarms = {"k": int(proposal["false_alarms"]), "n": int(proposal["healthy_runs"]),
                  "rate": float(proposal["false_alarm_rate"])}
        lines.append(f"| {name} | {_kn(frozen)} | {_kn(dead)} | {num(_float(proposal['median_epoch']), 1)} | "
                     f"{timing} | {_kn(alarms)} |")
    return lines


def _float(value):
    return None if value in ("", None) else float(value)


def harm_section() -> list[str]:
    """Detection split by how much final accuracy the fault cost."""
    rows = [r for r in read("harm.csv") if r["scope"] == "proposal"]
    if not rows:
        return []
    lines = ["| Network | Accuracy cost against twin | Broken runs | Mean cost | LHD | Accuracy curve | "
             "Loss curve | Simple fixed checks |", "|---|---|---|---|---|---|---|---|"]
    order = [a for a in ARCH_ORDER] + ["pooled"]
    for arch in order:
        for r in [r for r in rows if r["arch"] == arch]:
            cells = []
            for name in ("lhd", "accuracy", "loss", "simple"):
                cells.append(_kn({"k": int(r[f"{name}_caught"]), "n": int(r["broken_runs"]),
                                  "rate": float(r[f"{name}_rate"])}))
            label = "All three" if arch == "pooled" else ARCH_NAMES[arch]
            lines.append(f"| {label} | {r['harm']} | {r['broken_runs']} | "
                         f"{float(r['mean_drop_pts']):.2f} pts | " + " | ".join(cells) + " |")
    return lines


def honest_section(summary: dict) -> list[str]:
    """What the results support and what they do not, with the numbers behind each point."""
    pooled = summary["pooled"]
    prop = pooled["proposal"]
    healthy = pooled["healthy"]
    rows = [r for r in read("detectors.csv") if r["arch"] == "pooled"]
    by = defaultdict(list)
    for r in rows:
        by[r["detector"]].append(r)
    if not rows or not prop:
        return []
    lhd_frozen = _sum_rows(by["LHD"], ("frozen",))
    zero_frozen = _sum_rows(by["Zero update check"], ("frozen",))
    zero_prop = next(r for r in by["Zero update check"] if r["family"] == "proposal")
    lhd_dead = _sum_rows(by["LHD"], DEAD_FAMILIES)
    dead_rules = []
    for t in (25, 50, 70, 80):
        mine = next((v for k, v in by.items() if k.startswith(f"Dead units check, {t}%")), [])
        if mine:
            caught = _sum_rows(mine, DEAD_FAMILIES)
            alarms = next(r for r in mine if r["family"] == "proposal")
            dead_rules.append(f"at {t}% caught {_kn(caught)} and flagged {alarms['false_alarms']} of "
                              f"{alarms['healthy_runs']} healthy runs")
    anyrule = by.get("Any dead unit (wandb-callbacks verbose)", [])
    if anyrule:
        alarms = next(r for r in anyrule if r["family"] == "proposal")
        dead_rules.append(f"and a rule that fires on any dead unit caught {_kn(_sum_rows(anyrule, DEAD_FAMILIES))} "
                          f"but flagged {alarms['false_alarms']} of {alarms['healthy_runs']} healthy runs")
    harm = {r["harm"]: r for r in read("harm.csv") if r["scope"] == "proposal" and r["arch"] == "pooled"}
    clustered = pooled.get("proposal_clustered", {})
    lines = [
        "These points are written to be read by a skeptic. Each one gives the numbers behind it.",
        "",
        f"- LHD caught {_kn(prop['detected'])} of the proposal's broken runs and flagged "
        f"{_kn(healthy['lhd_false_alarm'])} of fresh healthy runs, with the same rules on all three networks.",
        f"- LHD was earlier than the accuracy curve on average for {clustered.get('by_seed', {}).get('lhd_earlier')} "
        f"of {clustered.get('by_seed', {}).get('groups')} seed groups (one per network and seed) and {clustered.get('by_condition', {}).get('lhd_earlier')} "
        f"of {clustered.get('by_condition', {}).get('groups')} fault conditions. Its alarm epoch is mostly 3 "
        "because patience 3 makes epoch 3 the earliest possible alarm for a fault planted at epoch 1.",
        f"- For frozen layers LHD adds nothing over a one line check. The zero update check caught "
        f"{_kn(zero_frozen)} at epoch 1 with {zero_prop['false_alarms']} false alarms in "
        f"{zero_prop['healthy_runs']} healthy runs, while LHD caught {_kn(lhd_frozen)} at epoch 3.",
        f"- For dead units LHD does better than the fixed thresholds tested, including the defaults of "
        "SageMaker Debugger (50%), DeepDiagnosis (70%), and wandb-callbacks (80%). LHD caught "
        f"{_kn(lhd_dead)} with {healthy['lhd_false_alarm']['k']} false alarms. A fixed dead unit rule "
        + ", ".join(dead_rules) + ". Healthy networks differ in how many units die on their own, so one "
        "threshold cannot fit all of them, and LHD's per layer healthy range can.",
        "- LHD misses small dead unit faults in layers where healthy runs already lose many units, such as "
        "10% in the fully connected network's middle layers and every fraction in the CNN's fc1.",
        "- The planted faults were designed to show up in the signals LHD measures. A frozen layer has a "
        "weight change of exactly 0 and a killed unit outputs exactly 0. LHD's high rate of naming the "
        "right cause is largely built into that design, since its cause rules map those same patterns.",
    ]
    if harm:
        low = harm.get("under 0.5 pts")
        high = harm.get("2 pts or more")
        total = sum(int(r["broken_runs"]) for r in harm.values())
        if low:
            lines.append(f"- Most planted faults did little harm. {low['broken_runs']} of {total} broken runs "
                         "lost under half a point of final accuracy against their healthy twin. The accuracy "
                         "curve missing those runs is partly the curve being right.")
        if high:
            lines.append(f"- On broken runs that lost 2 points or more, LHD caught {high['lhd_caught']} of "
                         f"{high['broken_runs']} and the accuracy curve caught {high['accuracy_caught']}.")
    lines += [
        "- Not tested here are real bugs found in the wild, large networks, long training runs, other "
        "optimizers, and GPU training. The networks are small, and the transformer only reaches about "
        f"{_transformer_accuracy()} next character accuracy.",
        "- The accuracy cost uses one healthy twin per broken run, so it carries noise. Some broken runs "
        "ended above their twin, which shows up as a negative mean cost.",
    ]
    return lines


def _transformer_accuracy() -> str:
    values = [float(r["final_acc"]) for r in read("runs.csv")
              if r["arch"] == "transformer" and r["role"] == "fresh"]
    return f"{100 * sum(values) / len(values):.0f}%" if values else "n/a"


CAPTIONS = (
    ("healthy_band_{arch}", "How the healthy range is built for one layer of the {name}. Gray lines are "
     "the 20 calibration runs that set the range. Blue lines are fresh healthy runs that never touched it. "
     "The shaded band is the range after widening by the margin and floor."),
    ("accuracy_curves", "Validation accuracy over training. The shaded band is the range of the 20 "
     "calibration runs and the dark line is the mean of the fresh healthy runs. Colored lines are the mean "
     "of broken runs for each planted fault."),
    ("anatomy_{arch}_frozen", "One broken {name} run with a frozen layer. Each panel is one signal of the "
     "planted layer against its healthy range, and the last panel is validation accuracy. The vertical rule "
     "marks LHD's alarm."),
    ("anatomy_{arch}_dead25", "One broken {name} run with 25% of one layer's units killed, drawn the same way."),
    ("cumulative_detection", "Share of the proposal's broken runs flagged by each epoch, for LHD, the "
     "accuracy curve, the loss curve, and the simple fixed checks."),
    ("detector_comparison", "Every detector on the same runs, pooled over the three networks. Whiskers are "
     "95% Wilson intervals."),
    ("dose_response", "Share of dead unit faults caught as the killed fraction grows, for LHD and for the "
     "accuracy curve, by network. Whiskers are 95% Wilson intervals."),
    ("harm", "Share of broken runs caught, split by how much final accuracy the fault cost against the "
     "run's healthy twin. Whiskers are 95% Wilson intervals."),
    ("false_alarms", "Share of fresh healthy runs each detector flagged, with 95% Wilson intervals."),
    ("diagnosis_matrix", "Planted fault against the cause LHD named, pooled over the three networks."),
    ("per_layer", "Share of broken runs where LHD named both the planted layer and the right cause, by "
     "fault and layer."),
    ("ablation", "LHD with each signal removed, or with one signal alone, on the same runs."),
    ("tradeoff", "Broken runs caught and healthy runs flagged as the margin changes, at floor 0.05. Solid "
     "lines use patience 3 and dashed lines patience 1. The rule marks the margin fixed before testing."),
    ("sensitivity", "Broken runs caught and healthy runs flagged for every margin and patience, at floor 0.05."),
    ("calibration_size", "False alarms and broken runs caught when the healthy range is built from fewer "
     "calibration runs, 20 random draws per size."),
    ("silent_damage", "Accuracy each fault cost against when each detector noticed it."),
)


def figure_list() -> list[str]:
    """Every figure that exists, numbered in reading order, with its caption."""
    folder = RESULTS / "figures"
    lines, number = [], 0
    for pattern, caption in CAPTIONS:
        archs = [a for a in ARCH_ORDER] if "{arch}" in pattern else [None]
        for arch in archs:
            name = pattern.format(arch=arch) if arch else pattern
            if not (folder / f"{name}.png").exists():
                continue
            number += 1
            text = caption.format(name=ARCH_NAMES[arch]) if arch else caption
            lines += [f"**Figure {number}.** {text} File `figures/{name}.png`, vector copy "
                      f"`figures/pdf/{name}.pdf`.", "", f"![Figure {number}](figures/{name}.png)", ""]
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
        lines += ["", f"Broken runs LHD never caught ({sum(len(v) for v in missed.values())}), "
                  "grouped by fault and layer:", "",
                  "| Architecture | Fault | Layer | Seeds missed |", "|---|---|---|---|"]
        lines += [f"| {ARCH_NAMES[a]} | {FAMILY_NAMES.get(f, f)} | {layer} | {', '.join(seeds)} |"
                  for (a, f, layer), seeds in sorted(missed.items())]
    if not (wrong or alarms or causes or missed):
        return ["", "None. Every caught run named the right layer and cause, "
                "and no fresh healthy run was flagged."]
    if wrong:
        lines += ["", f"Runs where LHD named the wrong layer ({len(wrong)}):", "",
                  "| Run | Planted in | LHD named | Cause named | Signals |", "|---|---|---|---|---|"]
        lines += [f"| {r['run_id']} | {r['fault_layer']} | {r['lhd_layer']} | {r['lhd_cause']} | "
                  f"{r['lhd_signals']} |" for r in wrong]
    if causes:
        lines += ["", f"Runs where LHD found the right layer but named the wrong cause ({len(causes)}):", "",
                  "| Run | Planted | LHD named | Signals | Epoch |", "|---|---|---|---|---|"]
        lines += [f"| {r['run_id']} | {r['fault_type']} | {r['lhd_cause']} | {r['lhd_signals']} | "
                  f"{r['lhd_epoch']} |" for r in causes]
    if alarms:
        lines += ["", f"False alarms on fresh healthy runs ({len(alarms)}):", "",
                  "| Run | Epoch | Layer | Cause named | Signals |", "|---|---|---|---|---|"]
        lines += [f"| {r['run_id']} | {r['lhd_epoch']} | {r['lhd_layer']} | {r['lhd_cause']} | "
                  f"{r['lhd_signals']} |" for r in alarms]
    return lines


def main() -> None:
    with open(RESULTS / "summary.json") as f:
        summary = json.load(f)
    settings = summary["settings"]
    pooled = summary["pooled"]
    tests = pooled["proposal_tests"]
    clustered = pooled.get("proposal_clustered", {})
    total_runs = sum(sum(s["runs"].values()) for s in summary["archs"].values())
    out = [
        "# LHD results",
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
        f"- Caught by LHD: {pct(pooled['proposal'].get('detected'), 1)}",
        f"- Right layer when caught: {pct(pooled['proposal'].get('layer_ok'), 1)}",
        f"- Right cause when caught: {pct(pooled['proposal'].get('cause_ok'), 1)}",
        f"- Caught by the accuracy curve at any point: {pct(pooled['proposal'].get('acc_detected'), 1)}",
        f"- LHD flagged the run before the accuracy curve did: {pct(pooled['proposal'].get('lhd_first'), 1)}",
        f"- Of the runs LHD caught, the accuracy curve never noticed: {pct(pooled.get('acc_never_on_detected'), 1)}",
        f"- Mean compute saved per broken run: {pct(pooled['proposal'].get('mean_compute_saved'), 1)}",
        f"- LHD false alarms on fresh healthy runs: {pct(pooled['healthy']['lhd_false_alarm'], 1)}",
        f"- Accuracy curve false alarms on the same runs: {pct(pooled['healthy']['acc_false_alarm'], 1)}",
        f"- Mean epochs LHD was earlier than the accuracy curve, with \"never\" counted as one "
        f"epoch past the end: {num(pooled['proposal'].get('mean_gap'))}, 95% interval "
        f"{_gap(pooled['proposal'].get('mean_gap'), clustered.get('mean_gap_ci')).split(' ', 1)[-1]} "
        f"from a bootstrap that resamples whole seeds.",
        f"- Fault conditions (fault and layer) where LHD was earlier on average: "
        f"{_groups(clustered.get('by_condition'))}, one-sided Wilcoxon on condition means.",
        f"- Seeds where LHD was earlier on average: {_groups(clustered.get('by_seed'))}.",
        f"- Run by run, for reference only since runs that share a seed are not independent: "
        f"LHD earlier on {tests.get('lhd_earlier')}, accuracy earlier on {tests.get('acc_earlier')}, "
        f"ties {tests.get('ties')}.",
        "",
        "## What the results show and what they do not",
        "",
        *honest_section(summary),
        "",
        "## LHD against other detectors",
        "",
        "Pooled over all three networks, on the proposal's broken runs. The simple fixed checks need no "
        "healthy runs and were added after the core results to test whether LHD beats rules anyone could "
        "write by hand. Earlier, later, and tie compare each detector with LHD run by run.",
        "",
        *detector_section(),
        "",
        "## Does LHD catch faults that cost accuracy",
        "",
        "Broken runs split by how much final accuracy they lost against their healthy twin.",
        "",
        *harm_section(),
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
        "Each cell shows broken runs caught, then runs where LHD named the right layer and cause, "
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
        *figure_list(),
        "",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
