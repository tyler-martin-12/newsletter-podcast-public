PYTHON ?= python3.11
VENV ?= .venv

.PHONY: install run test lint

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"

run:
	$(VENV)/bin/python -m newsletter_podcast.pipeline --config config.yaml

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy src
