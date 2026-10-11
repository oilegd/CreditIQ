# Convenience targets (GNU make). Windows users: see README for PowerShell equivalents.
PY ?= python

.PHONY: install inspect prepare train evaluate reports pipeline test lint format app synthetic notebooks clean

install:
	$(PY) -m pip install -e ".[dev,tda]"

inspect:
	$(PY) scripts/inspect_data.py

prepare:
	$(PY) scripts/prepare_data.py

train:
	$(PY) scripts/train_model.py --feature-set all

evaluate:
	$(PY) scripts/evaluate_model.py --feature-set all

reports:
	$(PY) scripts/generate_reports.py

pipeline: inspect prepare train evaluate reports

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

app:
	streamlit run app/streamlit_app.py

synthetic:
	$(PY) scripts/inspect_data.py --synthetic
	$(PY) scripts/prepare_data.py --synthetic
	$(PY) scripts/train_model.py --synthetic --feature-set all
	$(PY) scripts/evaluate_model.py --synthetic --feature-set all --bootstrap-reps 50
	$(PY) scripts/generate_reports.py --synthetic

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf data/interim/* data/processed/* data/synthetic/* models/applicant models/origination reports/metrics/*
