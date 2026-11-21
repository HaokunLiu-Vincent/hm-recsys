# Common commands. See AGENTS.md §8.
# Requires `uv` (https://docs.astral.sh/uv/). Python version comes from .python-version,
# dependencies from pyproject.toml + uv.lock. Every tool runs through `uv run`.

# Keep numeric libraries inside the 6-thread budget (AGENTS.md §4).
export OMP_NUM_THREADS := 6

.PHONY: setup format lint test-fast test test-torch-side test-lightgbm-side
.PHONY: image-require-clean image-build image-info image-lint image-test-fast image-test
.PHONY: image-test-torch-side image-test-lightgbm-side image-calibrate image-r6-repeatability

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

# Tests run in TWO processes: torch and LightGBM each load their own OpenMP runtime, and
# both in one process abort or hang (tests/conftest.py). The second run is the Ranker (M7).

## test-fast: unit tests only (should finish in < 1 minute)
test-fast:
	uv run pytest tests/unit -q
	uv run pytest tests/unit -q --lightgbm-side

## test: unit tests + integration tests
test:
	uv run pytest tests -q
	uv run pytest tests -q --lightgbm-side

# The two halves of `make test`, one process each. CI runs them as two steps, so that a
# failure on one side does not hide the result of the other.

## test-torch-side: unit + integration tests of the torch side only
test-torch-side:
	uv run pytest tests -q

## test-lightgbm-side: unit + integration tests of the LightGBM side only
test-lightgbm-side:
	uv run pytest tests -q --lightgbm-side

# ---------------------------------------------------------------- container image
# Stage 3 (docs/mlops.md §8). The image is built ONLY from a clean, committed tree and
# carries its commit: there is no .git inside an image, so hmrec.git_state reads the
# commit from the image instead. Results computed in a container are not bit-identical
# to the laptop; they are accepted by the tolerance of docs/mlops.md §8.3.
#
#   make image-build       build hmrec:dev (and hmrec:<commit>) for this machine's CPU
#   make image-info        what is in the image and what a container can use
#   make image-lint        `make lint` inside the image
#   make image-test        `make test` inside the image (two processes, as on the laptop)
#
# IMAGE_PLATFORM=linux/amd64 builds for another CPU (CI does; emulation is slow).
# IMAGE_RUN_FLAGS adds `docker run` flags, e.g. IMAGE_RUN_FLAGS="--cpus 6 --memory 12g".

IMAGE_NAME ?= hmrec
IMAGE_TAG ?= dev
IMAGE ?= $(IMAGE_NAME):$(IMAGE_TAG)
IMAGE_PLATFORM ?=
IMAGE_RUN_FLAGS ?=
# "=" and not ":=": git is asked only when an image target runs. Inside the image there
# is no repository, and `make lint` / `make test` must still work there.
IMAGE_COMMIT = $(shell git rev-parse HEAD)
IMAGE_PLATFORM_FLAG = $(if $(IMAGE_PLATFORM),--platform $(IMAGE_PLATFORM),)
# --rm: the container is removed when its command ends; nothing of a run is kept in it.
IMAGE_RUN = docker run --rm $(IMAGE_PLATFORM_FLAG) $(IMAGE_RUN_FLAGS) $(IMAGE)

## image-require-clean: stop unless the working tree equals HEAD (no change, no untracked file)
image-require-clean:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "image-build refused: the working tree differs from HEAD:"; \
		git status --short; \
		echo "An image is built from a clean, committed tree only (docs/mlops.md 8.1). Commit first."; \
		exit 1; \
	fi

## image-build: build the image from the committed tree; the commit goes into the image
image-build: image-require-clean
	docker build $(IMAGE_PLATFORM_FLAG) \
		--build-arg GIT_COMMIT=$(IMAGE_COMMIT) \
		--tag $(IMAGE) \
		--tag $(IMAGE_NAME):$(IMAGE_COMMIT) \
		.

## image-info: image id, commit, size; then Python, libraries, OpenMP, BLAS, CPUs and memory
image-info:
	@docker image inspect $(IMAGE) --format \
		'image: $(IMAGE){{println}}image id: {{.Id}}{{println}}registry digests: {{if .RepoDigests}}{{join .RepoDigests ", "}}{{else}}none (not pushed to a registry){{end}}{{println}}commit label: {{index .Config.Labels "org.opencontainers.image.revision"}}{{println}}built for: {{.Os}}/{{.Architecture}}{{println}}created: {{.Created}}{{println}}size: {{.Size}} bytes'
	@echo "--- torch side"
	@$(IMAGE_RUN) python scripts/image_info.py --side torch
	@echo "--- LightGBM side"
	@$(IMAGE_RUN) python scripts/image_info.py --side lightgbm

## image-lint: `make lint` inside the image
image-lint:
	$(IMAGE_RUN) make lint

## image-test-fast: `make test-fast` inside the image
image-test-fast:
	$(IMAGE_RUN) make test-fast

## image-test: `make test` inside the image (unit + integration, torch side then LightGBM side)
image-test:
	$(IMAGE_RUN) make test

## image-test-torch-side: the torch-side half of `make test` inside the image
image-test-torch-side:
	$(IMAGE_RUN) make test-torch-side

## image-test-lightgbm-side: the LightGBM-side half of `make test` inside the image
image-test-lightgbm-side:
	$(IMAGE_RUN) make test-lightgbm-side

# ---------------------------------------------------------------- M10 calibration
# docs/mlops.md §8.3, docs/m10_calibration_plan.md §2. One calibration item per container.
#
#   make image-calibrate ARGS="--run run1 --item recall --channel r1_popularity"
#
# The mounts are fixed here on purpose. Inputs (data/processed, artifacts) are mounted
# READ-ONLY, so no stage-1 file can be written whatever the code does. The only writable
# folders are the two calibration folders. 6 CPUs = num_threads of configs/base.yaml;
# 12 GiB = the project's memory budget: a process above it is stopped by the kernel.
CALIBRATION_CPUS ?= 6
CALIBRATION_MEMORY ?= 12g
CALIBRATION_ARTIFACTS = $(CURDIR)/artifacts/m10_calibration
CALIBRATION_RESULTS = $(CURDIR)/results/m10_calibration
ARGS ?=

## image-calibrate: run one calibration item in a container (ARGS = arguments of scripts/m10_calibrate.py)
image-calibrate:
	@mkdir -p "$(CALIBRATION_ARTIFACTS)" "$(CALIBRATION_RESULTS)"
	docker run --rm $(IMAGE_PLATFORM_FLAG) \
		--cpus $(CALIBRATION_CPUS) --memory $(CALIBRATION_MEMORY) \
		--mount "type=bind,src=$(CURDIR)/data/processed,dst=/app/data/processed,readonly" \
		--mount "type=bind,src=$(CURDIR)/artifacts,dst=/app/artifacts,readonly" \
		--mount "type=bind,src=$(CALIBRATION_ARTIFACTS),dst=/app/artifacts/m10_calibration" \
		--mount "type=bind,src=$(CALIBRATION_RESULTS),dst=/app/results/m10_calibration" \
		$(IMAGE) python scripts/m10_calibrate.py $(ARGS)

## image-r6-repeatability: the R6 fine-tuning reproducer in the image (synthetic data; ARGS = its arguments)
# Two fine-tunes in fresh processes, compared bit by bit (scripts/m10_r6_repeatability.py).
# No data is mounted: the reproducer draws its data from a seed.
#   make image-r6-repeatability ARGS="--setting implemented --rows 8192"
image-r6-repeatability:
	$(IMAGE_RUN) python scripts/m10_r6_repeatability.py $(ARGS)
