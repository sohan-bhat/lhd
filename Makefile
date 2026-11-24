PY := .venv-exp/bin/python
export PYTHONPATH := src

.PHONY: setup test runs unpack results

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
