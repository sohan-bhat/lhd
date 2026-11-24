# LHD

Layer Health Diagnostics: early detection of silent
training failures in neural networks.

**Status:** in development. Signal functions and tests are being built.
Experiments begin after science fair review approval.

## Setup
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    pytest


## Layout

    src/lhd/           the monitor itself, numpy only
      signals.py         the per-layer signals, four from the study and a fifth added later
      recorder.py        logs every layer's signals each epoch
      baseline.py        healthy band from healthy runs only
      detector.py        fires after `patience` epochs out of band
      diagnosis.py       names the likely cause from the signal pattern
      monitor.py         ties the three together for a live run
      replay.py          runs a saved history through LHD after the fact
      faults.py          plants a frozen layer or dead units
      torch_adapter.py   lets LHD watch any PyTorch model
    src/experiments/   the study from the proposal
      models.py          CNN, fully connected network, small transformer
      data.py            GTSRB, Fashion-MNIST, Tiny Shakespeare
      train.py           one run with one planted fault
      grid.py            every run in the study and its seed
      run.py             trains the grid in parallel, resumable
      evaluate.py        scores LHD against the accuracy curve
      sweep.py           settings, calibration size, and signal ablations
      analyze.py         writes results/summary.json and results/tables
      figures.py         writes results/figures
      report.py          writes results/REPORT.md
      demo.py            trains live and stops when LHD fires
      never_fired.py     measures the fifth signal on every saved run
      v2.py              scores four signals beside five, into results/v2.json
      blog_demo.py       writes blog/demo.json, the five runs in the blog post's figure
    src/tests/         the tests for the library and the scoring
    results/           every saved run, the tables, figures, and report
    blog/              the blog post

## Where everything is

Every LHD file lives in this folder.

    blog/lhd.md                      the blog post, the short version of the project
    results/REPORT.md                the full report from every run
    results/METHODS.md               the methods record
    results/LHD_data.xlsx            every table in one workbook
    archive/                         the Google Docs proposal draft and the AI use
                                     disclosure, kept here and not on GitHub

## Running the experiments

The experiments use the exact package versions in
`requirements-experiments.txt` on Python 3.10, since PyTorch has no
Intel Mac wheels past 2.2. The core library still needs only numpy.

    make setup
    make test

    make runs       train every run, more than 100 CPU hours, resumable
    make unpack     or skip training and unpack the saved runs
    make results    tables, figures, methods record, and report

Datasets download on first use into `data/`. Each run is saved to
`results/runs/<arch>/<run id>.json`, and finished runs are skipped,
so the runner can be stopped and restarted at any time. The core
phase is the study in the proposal. The more phase adds late faults,
cut gradients, and learning rate blowups. The seeds phase repeats the
proposal's faults on five more seeds.

Finished runs are packed into one compressed file per architecture
with `python -m experiments.bundle`, and the analysis reads those
files when the per run folders are gone. Results land in
`results/REPORT.md`.

## Version 2

The study was planned and run with four signals, and with them 411 of
the 470 broken runs were detected and 3 of 150 healthy runs flagged.
A fifth signal, the share of a layer's units that have been dead since
before training, was added after those results were in. With five
signals 459 of the 470 were detected and 6 of 150 healthy runs flagged.
The signal and its rule were chosen on runs outside the scored ones and
committed before the scored runs were measured, and those runs were then
scored once. Both versions and the limits of the comparison are in the
last section of `results/REPORT.md`.

    python -m experiments.never_fired --workers 8    measure the fifth signal, about nine hours
    python -m experiments.never_fired --pack         pack the measurements for git
    python -m experiments.v2                         score both versions

To watch LHD stop a broken run live:

    python -m experiments.demo --arch cnn --fault frozen --layer conv2
