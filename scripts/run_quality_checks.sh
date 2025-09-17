#!/usr/bin/env bash
set -euo pipefail

echo "Running Ruff lint..."
poetry run ruff check .

echo "Running Black format check..."
poetry run black . --check

echo "Running mypy type check..."
poetry run mypy .

echo "Running pytest suite..."
poetry run python -m pytest

echo "All checks passed."
