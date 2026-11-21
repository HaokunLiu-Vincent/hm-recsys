# The container image of stage 3 (docs/mlops.md §8.1).
#
# Build it with `make image-build`, never with a bare `docker build`: the Makefile refuses a
# dirty working tree and passes the commit hash, so that the image carries the commit it was
# built from (there is no .git inside an image).
#
# What is NOT in the image: data, models, credentials. Data are mounted (laptop) or read
# from Cloud Storage (cloud). .dockerignore lists what may enter the build at all.
#
# A result computed in this image is NOT bit-identical to the laptop: Linux uses another
# OpenMP runtime (GNU libgomp instead of Homebrew's libomp) and another BLAS (OpenBLAS
# instead of Accelerate). It is accepted by the tolerance of docs/mlops.md §8.3.

# Pinned by digest: the tags are only there for the reader. The digests are those of the
# multi-platform indexes, so the same line builds linux/arm64 (laptop) and linux/amd64 (CI).
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# libgomp1: the GNU OpenMP runtime. The LightGBM wheel does not bring its own and loads this
#           one. Its version is pinned here because another version can change the order of
#           floating-point sums; if Debian replaces this version the build FAILS, which is
#           wanted: the change must be made here, on purpose, and calibrated again.
# git:      the tests of hmrec.git_state build throw-away repositories. The image itself
#           has no .git; its commit comes from the build argument below.
# make:     `make lint` and `make test` run inside the image.
# libatomic1: the prebuilt Node.js that pyright runs on needs it (on arm64 the binary does
#           not start without it). It takes no part in any computation of the project.
ARG LIBGOMP_VERSION=14.2.0-19
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        "libgomp1=${LIBGOMP_VERSION}" \
        git \
        make \
        libatomic1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# The process does not run as root. The code and the environment below belong to root, so
# the running process can read them but cannot change them in place.
RUN useradd --create-home --uid 1000 --shell /bin/bash hmrec

WORKDIR /app

# uv: use the Python of the base image (never download one), copy packages into the
# environment (a hard link cannot cross from the build cache into an image layer) and
# compile to bytecode at build time, because /app is read-only for the running user.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

# Layer 1: the dependencies, exactly as uv.lock pins them (including the dev group: ruff,
# pyright and pytest run inside the image). Only these two files are copied first, so this
# layer is rebuilt only when a dependency changes, not on every code change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# pyright is a Node.js program. The Python package brings pyright itself but not Node.js;
# it downloads Node.js on first use. Do that once here, as the user that will run it, so
# that `make lint` needs no network. The version is the one the laptop uses.
ENV PYRIGHT_PYTHON_NODE_VERSION=26.10.0 \
    PYRIGHT_PYTHON_GLOBAL_NODE=0 \
    PYRIGHT_PYTHON_IGNORE_WARNINGS=1
USER hmrec
RUN /app/.venv/bin/pyright --version
USER root

# Layer 2: the code. .dockerignore decides what "." is: only tracked source folders, never
# data/, artifacts/, .git or .venv.
COPY . .
RUN uv sync --frozen

# The commit this image was built from (docs/mlops.md §8.1). Required: without it the build
# stops here. hmrec.git_state reads the variable instead of asking git.
ARG GIT_COMMIT
RUN echo "${GIT_COMMIT}" | grep --quiet --extended-regexp '^[0-9a-f]{40}$' \
    || (echo "build with --build-arg GIT_COMMIT=\$(git rev-parse HEAD); use make image-build" >&2; exit 1)
LABEL org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.title="hmrec" \
      org.opencontainers.image.description="H&M recommendation system: weekly pipeline (stage 3)"

# At run time:
#   HMREC_IMAGE_GIT_COMMIT   the commit of the image, recorded by every run.
#   UV_NO_SYNC               `uv run` uses the environment as built; it never installs.
#   PYTHONDONTWRITEBYTECODE, RUFF_CACHE_DIR, PYTEST_ADDOPTS
#                            /app is read-only for the user, so caches go to /tmp or are off.
# Thread counts are NOT set here: the Makefile sets OMP_NUM_THREADS=6 and the code sets the
# rest from configs/base.yaml, exactly as on the laptop.
ENV HMREC_IMAGE_GIT_COMMIT="${GIT_COMMIT}" \
    UV_NO_SYNC=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUFF_CACHE_DIR=/tmp/ruff-cache \
    PYTEST_ADDOPTS="-p no:cacheprovider" \
    PATH="/app/.venv/bin:${PATH}"

USER hmrec
CMD ["python", "scripts/image_info.py", "--side", "lightgbm"]
