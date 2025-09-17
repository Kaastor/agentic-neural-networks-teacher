FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1

WORKDIR /app

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root

COPY . .

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
