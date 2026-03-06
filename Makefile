.PHONY: ci test test-ci test-unit test-integration test-contract test-e2e lint format type-check pre-commit security test-hmm bootstrap-hmm

ci: pre-commit type-check test

test:
	python -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=68

test-ci:
	mkdir -p artifacts/test-results
	python -m pytest --junitxml=artifacts/test-results/junit.xml --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=68

test-unit:
	python -m pytest -m unit --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=68

test-integration:
	python -m pytest -m integration --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=68

test-contract:
	python -m pytest -m contract --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=68

test-e2e:
	python -m pytest -m e2e --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=68

test-hmm:
	python -m pytest tests/unit/test_hmm_regime_unit.py -q

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

bootstrap-hmm:
	python scripts/bootstrap_hmm_regime.py --config config/config.yaml --input $(DATA)

docker-build:
	docker build -t empire/cerberus:latest .

.PHONY: up down logs logs-follow
up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs cerberus-scheduler

logs-follow:
	docker-compose logs -f cerberus-scheduler
