# One-command reproducibility. `make demo` regenerates every artifact in reports/.
.PHONY: help install demo quick smoke test lint api dashboard audit sample mydata clean all

PY ?= python

help:
	@echo "make install    - install pinned dependencies"
	@echo "make demo       - full run on real UCI data -> reports/ + models/"
	@echo "make quick      - same, smaller SHAP sample (faster)"
	@echo "make smoke      - synthetic-data pipeline smoke test (no download)"
	@echo "make test       - run the test suite (leakage + money + smoke)"
	@echo "make api        - serve the scoring API on :8000"
	@echo "make dashboard  - open the merchant dashboard on :8501"
	@echo "make audit      - verify the decision ledger's hash chain"
	@echo "make sample     - regenerate the bring-your-own-data sample export"
	@echo "make mydata FILE=path.csv - run the whole pipeline on your own file"
	@echo "make all        - install, test, demo"

install:
	$(PY) -m pip install -r requirements.txt

demo:
	$(PY) run_demo.py

quick:
	$(PY) run_demo.py --quick

smoke:
	$(PY) run_demo.py --synthetic --quick

sample:
	$(PY) scripts/make_sample_export.py

# Usage: make mydata FILE=data/sample/merchant_export_sample.csv
mydata:
	$(PY) run_demo.py --data-file $(FILE)

test:
	$(PY) -m pytest tests/ -q

api:
	$(PY) -m uvicorn app.api:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run app/dashboard.py

audit:
	$(PY) -c "import sys; sys.path.insert(0, 'src'); from returnrisk.audit import DecisionLedger; from returnrisk.config import load_config; import json; print(json.dumps(DecisionLedger.from_config(load_config()).stats(), indent=2))"

clean:
	rm -rf reports models audit reports_user models_user audit_user data/datasets .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

all: install test demo
