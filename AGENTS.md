# Repository Guidelines

## Project Structure & Module Organization
Application code lives in `src/`: FastAPI bootstrapping in `app.py` and `mcp_server.py`, the LangChain agent in `agent_orchestrator.py`, supporting services such as `instagram_service.py` and `analysis_service.py`, and shared utilities. HTTP schemas and dependencies reside in `src/api/`, while persistence helpers and migrations live in `repository.py` and `migrations/`. Keep tests under `tests/`, mirroring the module layout, and place deployment manifests or reference configs inside `config/` and `docs/`.

## Build, Test, and Development Commands
`python main.py server` starts the production-ready MCP/FastAPI service. For rapid iteration run `uvicorn src.app:create_app --reload`. High-level agent jobs go through `python main.py run --task "<instruction>"` and smoke checks via `python main.py test`. Execute the suite with `pytest`; async endpoints are covered through `pytest-asyncio`. Container workflows use `docker build -t iram-agent:latest .` followed by `docker run -p 8000:8000 --env-file .env iram-agent:latest`.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indentation, and descriptive docstrings. Public interfaces should expose type hints and Pydantic models rather than loose dictionaries. Use `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, and keep structured logging through `structlog`. Format with `black` (default settings) and sort imports before committing: `python -m black src tests`.

## Testing Guidelines
Add unit or integration tests alongside touched modules using the `tests/test_<subject>.py` pattern. Prefer deterministic fixtures and mock out external APIs; `tests/test_basic.py` illustrates expected coverage. Use `pytest.mark.asyncio` with `httpx.AsyncClient` when exercising async routes. Run `pytest` and `python main.py test` locally, and ensure new features include both success and failure-path assertions.

## Commit & Pull Request Guidelines
Follow the existing log style—short, imperative summaries with optional area prefixes (`Healthcheck: use /health consistently`). Group related edits together, describe context and risk in the PR, and link tracking issues. Include screenshots or sample JSON for endpoint changes, highlight new environment variables, and call out any follow-up tasks.

## Environment & Secrets Management
Copy `.env.example` to `.env`, populate Instagram, OpenAI, Firecrawl, and database credentials, and exclude secrets from commits. Document any new keys inside `.env.example` and surface them through `IRamConfig` defaults so deployments fail safely when configuration is incomplete.
