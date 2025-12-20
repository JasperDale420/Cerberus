.PHONY: ci test test-ci test-unit test-integration test-contract test-e2e lint format type-check pre-commit security

ci: pre-commit type-check test

test:
	python -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70

test-ci:
	mkdir -p artifacts/test-results
	python -m pytest --junitxml=artifacts/test-results/junit.xml --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70

test-unit:
	python -m pytest -m unit --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70

test-integration:
	python -m pytest -m integration --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70

test-contract:
	python -m pytest -m contract --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70

test-e2e:
	python -m pytest -m e2e --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70

lint:
	ruff check .

format:
	black --check .

type-check:
	mypy

pre-commit:
	pre-commit run --all-files

security:
	bandit -ll -r src
	detect-secrets-hook --baseline .secrets.baseline
