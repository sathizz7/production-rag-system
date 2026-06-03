.PHONY: migrate serve ingest query test test-int fmt lint type up down eval obs-up

migrate:   ## Run Alembic migrations against DATABASE_URL (from .env)
	uv run alembic upgrade head

serve:     ## Run the API with the streaming UI at /ui/ (no-make: uv run uvicorn rag.api.app:create_app --factory --port 8000)
	uv run uvicorn rag.api.app:create_app --factory --reload --port 8000

ingest:    ## Ingest PDFs: make ingest CORPUS=./data/raw
	uv run rag-ingest $(CORPUS)

query:     ## Ask a question: make query Q="does nitrogen help maize?"
	curl -s -X POST localhost:8000/query -H "content-type: application/json" -d "{\"query\": \"$(Q)\"}"

test:      ## Unit tests (fast, no Postgres required)
	uv run pytest -m "not integration and not live"

test-int:  ## Integration tests (needs local Postgres+pgvector + TEST_DATABASE_URL in .env)
	uv run pytest -m integration

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy src

# --- Optional: Docker path (reviewers who lack a local Postgres) ---
up:        ## Build and start api + postgres via Docker Compose (optional, reviewer path)
	docker compose up --build -d

down:
	docker compose down

eval:      ## Run the live golden-set smoke eval (needs GEMINI_API_KEY + EVAL_DATABASE_URL; costs money)
	uv run rag-eval                                               # no-make: same line

obs-up:    ## Optional Prometheus+Grafana+Langfuse dashboards
	docker compose -f docker-compose.observability.yml up -d
