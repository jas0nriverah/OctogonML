.PHONY: demo serve test lint benchmark
demo:
	cagemetrics demo
serve:
	cagemetrics serve
test:
	pytest
lint:
	ruff check src tests migrations
benchmark:
	cagemetrics benchmark --requests 300 --concurrency 1 --warmup 20
	cagemetrics benchmark --requests 500 --concurrency 10 --warmup 20
