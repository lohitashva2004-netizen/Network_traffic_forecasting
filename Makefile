.PHONY: install test run clean

install:
	pip install -r requirements-dev.txt

test:
	pytest -q

run:
	python scripts/run_experiments.py

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
