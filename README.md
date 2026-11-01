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
      signals.py         the four per-layer signals
      recorder.py        logs every layer's signals each epoch
      baseline.py        healthy band from healthy runs only
      detector.py        fires after `patience` epochs out of band
      diagnosis.py       names the likely cause from the signal pattern
      monitor.py         ties the three together for a live run
      replay.py          runs a saved history through LHD after the fact
      faults.py          plants a frozen layer or dead units
      torch_adapter.py   lets LHD watch any PyTorch model
    experiments/       the study from the proposal
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
    results/           every saved run, the tables, figures, and report

## Running the experiments

PyTorch no longer ships wheels for Intel Macs past 2.2, so the
experiments run on Python 3.10 to 3.12 with torch 2.2 and numpy
below 2. The core library still needs only numpy.

    python3.10 -m venv .venv-exp
    source .venv-exp/bin/activate
    pip install "torch==2.2.2" "torchvision==0.17.2" "numpy<2" pyarrow scipy matplotlib
    pip install -e ".[dev]"
    pytest

    python -m experiments.run --phase core --workers 5
    python -m experiments.run --phase more --workers 5
    python -m experiments.run --phase seeds --workers 5
    python -m experiments.analyze
    python -m experiments.figures
    python -m experiments.report

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

To watch LHD stop a broken run live:

    python -m experiments.demo --arch cnn --fault frozen --layer conv2
