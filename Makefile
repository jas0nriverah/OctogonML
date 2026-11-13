.PHONY: demo serve test lint benchmark check verify start
PYTHON ?= .venv/bin/python
check:
	./start.sh --check
verify:
	./scripts/verify.sh
start:
	./start.sh
demo:
	$(PYTHON) -m cagemetrics.cli demo
serve:
	$(PYTHON) -m cagemetrics.cli serve
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check src tests migrations scripts
benchmark:
	$(PYTHON) -m cagemetrics.cli benchmark --requests 300 --concurrency 1 --warmup 20
	$(PYTHON) -m cagemetrics.cli benchmark --requests 500 --concurrency 10 --warmup 20
