# Common commands. See AGENTS.md §8.
# Requires `uv` (https://docs.astral.sh/uv/). Python version comes from .python-version,
# dependencies from pyproject.toml + uv.lock. Every tool runs through `uv run`.

# Keep numeric libraries inside the 6-thread budget (AGENTS.md §4).
export OMP_NUM_THREADS := 6

.PHONY: setup format lint test-fast test

## setup: create / update .venv from uv.lock (includes the dev group)
setup:
	uv sync

## format: auto-format code with ruff
format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

## lint: ruff format check, ruff lint, pyright type check
lint:
	uv run ruff format --check src tests scripts
	uv run ruff check src tests scripts
	uv run pyright

## test-fast: unit tests only (should finish in < 1 minute)
test-fast:
	uv run pytest tests/unit -q

## test: unit tests + integration tests
test:
	uv run pytest tests -q
