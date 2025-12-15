.PHONY: ci test lint format type-check

ci: lint format type-check test

test:
	python -m pytest --cov=src --cov-report=term-missing --cov-report=xml

lint:
	ruff check .

format:
	ruff format --check .

type-check:
	mypy .
