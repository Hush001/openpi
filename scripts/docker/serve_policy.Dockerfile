# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
ARG http_proxy
ARG https_proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/
COPY ./cache /cache
ENV UV_CACHE_DIR=/cache

WORKDIR /app

# Needed because LeRobot uses git-lfs.
RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang curl


# Install the project's dependencies using the lockfile and settings
RUN python3.11 -m venv /.venv
RUN /.venv/bin/pip install --upgrade pip
RUN /.venv/bin/pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
