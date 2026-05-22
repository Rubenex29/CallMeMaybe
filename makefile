.ONESHELL:

SHELL := /bin/bash

VENV := venv
PYTHON := python3
MAIN := src/call_me_maybe.py
DEFAULT_DEFINITIONS := data/input/functions_definition.json
DEFAULT_TESTS := data/input/function_calling_tests.json
DEFAULT_OUTPUT := data/output/function_calling_results.json

.PHONY: install run debug clean lint lint-strict

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && \
	$(PYTHON) -m pip install uv && \
	uv sync --directory llm_sdk --active --no-cache && \
	$(PYTHON) -m pip install mypy && \
	$(PYTHON) -m pip install flake8 && \
	$(PYTHON) -m pip install accelerate && \
	$(PYTHON) -m pip install llm_sdk

run:
	uv run $(PYTHON) $(MAIN) \
		--definitions-path $(DEFAULT_DEFINITIONS) \
		--tests-path $(DEFAULT_TESTS) \
		--output-path $(DEFAULT_OUTPUT)

debug:
	uv run $(PYTHON) -m pdb $(MAIN) \
		--definitions-path $(DEFAULT_DEFINITIONS) \
		--tests-path $(DEFAULT_TESTS) \
		--output-path $(DEFAULT_OUTPUT)

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

lint:
	flake8 . --exclude=venv,.venv,llm_sdk
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '(^|/)(\.venv|venv|llm_sdk)(/|$)'

lint-strict:
	flake8 . --exclude=venv,.venv,llm_sdk
	mypy . --strict --exclude '(^|/)(\.venv|venv|llm_sdk)(/|$)'