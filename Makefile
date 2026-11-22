# Common commands. See AGENTS.md §8.
# Requires `uv` (https://docs.astral.sh/uv/). Python version comes from .python-version,
# dependencies from pyproject.toml + uv.lock. Every tool runs through `uv run`.

# Keep numeric libraries inside the 6-thread budget (AGENTS.md §4).
export OMP_NUM_THREADS := 6

.PHONY: setup format lint test-fast test test-torch-side test-lightgbm-side
.PHONY: image-require-clean image-build image-info image-lint image-test-fast image-test
.PHONY: image-test-torch-side image-test-lightgbm-side image-calibrate image-r6-repeatability
.PHONY: setup-tools dvc data-manifest data-verify bootstrap-verify
.PHONY: mlflow-sync mlflow-check mlflow-ui test-tools
.PHONY: weekly image-weekly replay-compare image-replay-compare

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

# ---------------------------------------------------------------- M12 weekly pipeline
# docs/mlops.md §6. One tick: close the week before the anchor, open the week of the anchor.
#
#   make image-weekly ARGS="--anchor 2020-07-29 --mode eval"     one tick in the container
#   make image-replay-compare                                    the closed weeks against
#                                                                their stage-1 records
#
# A tick runs in the container, from an image built with `make image-build`. The mounts
# are fixed here on purpose, and they are what makes the tick self-contained:
#   read-only   data/processed (the source tables), data/bootstrap (the state at the end
#               of F4), artifacts/features/category_codes.json. These are exactly the
#               DVC-tracked inputs; NO other stage-1 artifact is visible in the container,
#               so a tick cannot read a stage-1 candidate, feature, label or model.
#   writable    artifacts/pipeline (the pipeline's state) and results/m12_replay (its
#               result tables). Nothing else can be written.
# 6 CPUs = num_threads of configs/base.yaml; 12 GiB = the project's memory budget.
# Keep the laptop awake for a long run: caffeinate -i make image-weekly ARGS="..."
PIPELINE_STATE = $(CURDIR)/artifacts/pipeline
REPLAY_RESULTS = $(CURDIR)/results/m12_replay
CATEGORY_CODES = artifacts/features/category_codes.json

## image-weekly: one tick of the weekly pipeline in a container (ARGS = arguments of scripts/weekly.py)
image-weekly:
	@mkdir -p "$(PIPELINE_STATE)" "$(REPLAY_RESULTS)"
	docker run --rm $(IMAGE_PLATFORM_FLAG) \
		--cpus $(CALIBRATION_CPUS) --memory $(CALIBRATION_MEMORY) \
		--mount "type=bind,src=$(CURDIR)/data/processed,dst=/app/data/processed,readonly" \
		--mount "type=bind,src=$(CURDIR)/data/bootstrap,dst=/app/data/bootstrap,readonly" \
		--mount "type=bind,src=$(CURDIR)/$(CATEGORY_CODES),dst=/app/$(CATEGORY_CODES),readonly" \
		--mount "type=bind,src=$(PIPELINE_STATE),dst=/app/artifacts/pipeline" \
		--mount "type=bind,src=$(REPLAY_RESULTS),dst=/app/results/m12_replay" \
		$(IMAGE) python scripts/weekly.py $(ARGS)

## weekly: one tick on this machine, without a container (ARGS as above; the laptop's strict Ranker entry)
weekly:
	uv run python scripts/weekly.py $(ARGS)

## image-replay-compare: compare every closed week of the replay with its stage-1 record (reads no label)
# A container of its own, NOT a tick's: the pair overlap needs the stage-1 candidate files
# and the serve diagnostic the saved Kaggle prediction set, which a tick cannot see. Here
# they are mounted read-only, the pipeline's state is read-only too, and only the replay's
# result tables can be written. The stage-1 result tables (the records) are in the image.
image-replay-compare:
	@mkdir -p "$(REPLAY_RESULTS)"
	docker run --rm $(IMAGE_PLATFORM_FLAG) \
		--cpus $(CALIBRATION_CPUS) --memory $(CALIBRATION_MEMORY) \
		--mount "type=bind,src=$(CURDIR)/artifacts/candidates,dst=/app/artifacts/candidates,readonly" \
		--mount "type=bind,src=$(CURDIR)/artifacts/kaggle/submission,dst=/app/artifacts/kaggle/submission,readonly" \
		--mount "type=bind,src=$(PIPELINE_STATE),dst=/app/artifacts/pipeline,readonly" \
		--mount "type=bind,src=$(REPLAY_RESULTS),dst=/app/results/m12_replay" \
		$(IMAGE) python scripts/m12_replay_compare.py

## replay-compare: the same comparison on this machine, without a container (needs a clean working tree)
replay-compare:
	uv run python scripts/m12_replay_compare.py

# ---------------------------------------------------------------- M11 versioning
# docs/mlops.md §7. DVC and MLflow are in dependency groups of their own ("versioning",
# "tracking"): `make setup` and the container image install neither.
#
# They also get an environment of their own, .venv-tools, and are never installed into
# .venv. Why: .venv is what lint, tests and every run use, and it must stay exactly the
# runtime + dev set of uv.lock. MLflow brings pandas, and with pandas installed the type
# checker reads LightGBM's optional types differently (found on 2026-11-22: `make lint`
# then reports an error in M7 code that did not change).
#
#   make setup-tools       create / update .venv-tools (the project + both groups)
#   make dvc ARGS="pull"   run a DVC command (README, "Data")
#   make data-manifest     write configs/data_manifest.json from the DVC-tracked files
#   make data-verify       compare the DVC-tracked files on disk with the manifest
#                          (some of them: ARGS="--only data/processed")
#   make bootstrap-verify  compare data/bootstrap/ with the manifest inside it
#   make mlflow-sync       copy result tables and run manifests into the local MLflow store
#   make mlflow-check      every metric in the store against its source
#   make mlflow-ui         serve the local MLflow UI (configs/m11_versioning.yaml: host, port)
#   make test-tools        the tracking tests with MLflow installed (a real SQLite store)
#
# Neither tool reports usage to its maker from this repository: DVC's analytics are off in
# .dvc/config and here; MLflow's telemetry is switched off here and in hmrec.tracking.store.
export DVC_NO_ANALYTICS := 1
export MLFLOW_DISABLE_TELEMETRY := true
export DO_NOT_TRACK := true
export MLFLOW_DISABLE_AGENT_HINT := 1
TOOLS_ENVIRONMENT = .venv-tools
TOOLS_GROUPS = --group versioning --group tracking
TOOLS_RUN = UV_PROJECT_ENVIRONMENT=$(TOOLS_ENVIRONMENT) uv run $(TOOLS_GROUPS)

## setup-tools: create / update .venv-tools with DVC and MLflow (never touches .venv)
setup-tools:
	UV_PROJECT_ENVIRONMENT=$(TOOLS_ENVIRONMENT) uv sync $(TOOLS_GROUPS)

## dvc: run a DVC command in the tools environment, e.g. make dvc ARGS="status --cloud"
dvc:
	$(TOOLS_RUN) dvc $(ARGS)

## data-manifest: write the manifest (path, size, sha256, md5) of every DVC-tracked file
data-manifest:
	$(TOOLS_RUN) python scripts/m11_data_manifest.py write

## data-verify: compare DVC-tracked files on disk with the manifest (ARGS="--only data/processed")
# Needs no DVC: it only hashes files, so it runs in .venv.
data-verify:
	uv run python scripts/m11_data_manifest.py verify $(ARGS)

## bootstrap-verify: compare the bootstrap state with the manifest inside it
bootstrap-verify:
	uv run python scripts/m11_bootstrap_state.py --verify

## mlflow-sync: write the records that are missing to the local MLflow store
mlflow-sync:
	$(TOOLS_RUN) python scripts/m11_mlflow.py sync

## mlflow-check: every metric in the local MLflow store against its source table
mlflow-check:
	$(TOOLS_RUN) python scripts/m11_mlflow.py check

## mlflow-ui: serve the local MLflow UI (stop with Ctrl-C)
mlflow-ui:
	$(TOOLS_RUN) python scripts/m11_mlflow.py ui

## test-tools: the tracking tests where MLflow is installed (sync into a real SQLite store)
test-tools:
	$(TOOLS_RUN) pytest tests/unit/test_tracking_store.py tests/unit/test_tracking_check.py tests/unit/test_tracking_records.py -q
