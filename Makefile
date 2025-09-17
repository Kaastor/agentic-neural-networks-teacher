.PHONY: install lint format typecheck test run-api pre-commit docker-build docker-run

install:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run black .
	poetry run ruff check . --fix

typecheck:
	poetry run mypy .

test:
	poetry run pytest

pre-commit:
	poetry run pre-commit run --all-files

run-api:
	poetry run uvicorn services.api.main:app --reload

docker-build:
	docker build -f services/api/Dockerfile -t tutor-api:local .

docker-run:
	docker run --rm -p 8000:8000 tutor-api:local
