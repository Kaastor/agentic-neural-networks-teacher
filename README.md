# agentic-neural-networks-teacher

Agentic tutoring platform that delivers a deep backpropagation curriculum with verified, closed-book content.

## Milestone status

- **Milestone 0** – scaffolding ✅
- **Milestone 1** – curriculum graph and canonical content ✅

## Local development

```bash
poetry install
poetry run uvicorn services.api.main:app --reload
```

### HTTP surface (read-only)

- `GET /healthz` – readiness check
- `GET /hello-agent` – milestone 0 agent example
- `GET /concept/{concept_id}` – fully expanded concept payload (sections, objectives, facts, examples, templates)
- `GET /facts?ids=...` – canonical fact lookup (requires one or more `ids` query parameters)

## Tests

```bash
poetry run python -m pytest
```
