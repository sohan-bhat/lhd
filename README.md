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

The experiments use the exact package versions in
`requirements-experiments.txt` on Python 3.10, since PyTorch has no
Intel Mac wheels past 2.2. The core library still needs only numpy.
The paper is compiled with tectonic.

    make setup
    make test

    make runs       train every run, more than 100 CPU hours, resumable
    make unpack     or skip training and unpack the saved runs
    make results    tables, figures, methods record, and report
    make paper      the paper in paper/lhd.pdf
    make table3     print one table of the paper, from table1 to table5

Datasets download on first use into `data/`. Each run is saved to
`results/runs/<arch>/<run id>.json`, and finished runs are skipped,
so the runner can be stopped and restarted at any time. The core
phase is the study in the proposal. The more phase adds late faults,
cut gradients, and learning rate blowups. The seeds phase repeats the
proposal's faults on five more seeds.

Finished runs are packed into one compressed file per architecture
with `python -m experiments.bundle`, and the analysis reads those
files when the per run folders are gone. Results land in
`results/REPORT.md`, and the paper built from them in `paper/lhd.pdf`.

To watch LHD stop a broken run live:

    python -m experiments.demo --arch cnn --fault frozen --layer conv2
