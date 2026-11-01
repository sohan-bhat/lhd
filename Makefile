PY := .venv-exp/bin/python

.PHONY: setup test runs unpack results paper table1 table2 table3 table4 table5

setup:
	python3.10 -m venv .venv-exp
	.venv-exp/bin/pip install -r requirements-experiments.txt
	.venv-exp/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

runs:
	$(PY) -m experiments.run --phase core --workers 5
	$(PY) -m experiments.run --phase seeds --workers 5
	$(PY) -m experiments.run --phase more --workers 5
	$(PY) -m experiments.bundle

unpack:
	$(PY) -m experiments.bundle --unpack

results:
	$(PY) -m experiments.analyze
	$(PY) -m experiments.export
	$(PY) -m experiments.figures
	$(PY) -m experiments.methods
	$(PY) -m experiments.report

paper:
	$(PY) -m experiments.paper

table1:
	$(PY) -m experiments.paper --table tab:rules

table2:
	$(PY) -m experiments.paper --table tab:networks

table3:
	$(PY) -m experiments.paper --table tab:headline

table4:
	$(PY) -m experiments.paper --table tab:detectors

table5:
	$(PY) -m experiments.paper --table tab:ablation
