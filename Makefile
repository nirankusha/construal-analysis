.PHONY: install run lint test

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .

run:
	. .venv/bin/activate && python -m construal.cli --in selected.parquet --out artifacts --all

lint:
	. .venv/bin/activate && python -m pip install ruff && ruff check construal

test:
	. .venv/bin/activate && python -m pip install pytest && pytest -q
