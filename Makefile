.PHONY: migrate serve ingest query test test-int fmt lint type up down

migrate:   ## Run Alembic migrations against DATABASE_URL (from .env)
	uv run alembic upgrade head

serve:     ## Run the API locally against local Postgres
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
