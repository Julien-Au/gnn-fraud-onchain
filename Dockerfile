# Reproducible image for gnn-fraud-onchain.
#
# Default build installs the core (pure-Python) environment and runs the CLI.
# For the graph stack (torch + torch-geometric), build with:
#   docker build --build-arg EXTRAS="--extra gnn" -t gnn-fraud .
# The core image stays small and fast; the gnn image is large (pulls torch).

FROM python:3.10-slim

# uv provides the locked, reproducible environment (same as local + CI).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only what the build needs first, for layer caching.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

ARG EXTRAS=""
# --frozen: fail if the lockfile is out of date (reproducibility guarantee).
RUN uv sync --frozen ${EXTRAS}

# Data and results are produced at run time (mounted or downloaded), never baked in.
ENTRYPOINT ["uv", "run", "gnn-fraud"]
CMD ["info"]
