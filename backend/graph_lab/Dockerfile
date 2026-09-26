# Verified ARM64 GB10 runtime -- the same pinned digest as backend/Dockerfile
# (reuses an existing local image; no running LLM service is used). Also
# already ships fastapi==0.135.3, uvicorn==0.44.0, pydantic==2.13.1, numpy,
# and torch -- confirmed by running `python -c "import fastapi, ..."`
# against this exact digest -- which is why this file's own `pip install`
# below uses `--no-deps`, matching `backend/Dockerfile`'s pattern: this
# package's own `pyproject.toml` pins those same versions to document what
# the base image provides, not to reinstall them.
#
# Build context is the repo root (this file copies from `training/`,
# `scripts/`, and `public/data/`, not just `backend/graph_lab/`):
#   docker build -f backend/graph_lab/Dockerfile -t flyarena-graph-lab:local .
FROM nvcr.io/nvidia/vllm@sha256:98494fdaea68ad892c107180b0be41c99caa3c77683175becc4e202b2cb75b27

# --- Node 22, pinned by exact version and sha256-verified against the
# official nodejs.org release checksums (never fetched over an
# unauthenticated/unverified channel) -- needed to run the esbuild-bundled
# graph-lab entries. Matches this repo's own `package.json` `engines`
# constraint (">=22.22.2 <23").
ARG NODE_VERSION=22.22.3
ARG NODE_SHA256=1c4a9933a5e45bc88f54f70b5f91232c127ec49f1a5989d23fb85824c7adf9b7
RUN set -eu; \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-arm64.tar.xz"; \
    echo "${NODE_SHA256}  node-v${NODE_VERSION}-linux-arm64.tar.xz" | sha256sum -c -; \
    tar -xJf "node-v${NODE_VERSION}-linux-arm64.tar.xz" -C /usr/local --strip-components=1; \
    rm "node-v${NODE_VERSION}-linux-arm64.tar.xz"

WORKDIR /app

# --- Python package: the `graph_lab` FastAPI service itself. `--no-deps`:
# see this file's header -- the base image already provides every pinned
# dependency at a compatible version.
COPY backend/graph_lab/pyproject.toml ./graph_lab_src/pyproject.toml
COPY backend/graph_lab/__init__.py backend/graph_lab/service.py backend/graph_lab/jobs.py backend/graph_lab/models.py ./graph_lab_src/
RUN python -m pip install --no-cache-dir --no-deps ./graph_lab_src \
    && rm -rf ./graph_lab_src

# --- esbuild-bundled JS entries/workers, produced locally by
# `npm run graph-lab:bundle` (gitignored, not built inside this
# Dockerfile -- no Node build toolchain, npm registry access, or repo
# checkout is needed at image-build time beyond these already-built
# outputs; run the bundle step before `docker build`).
COPY backend/graph_lab/js /opt/graph-lab/js

# --- Real-graph engine sources for WP2 (Python side). Not yet wired to any
# job kind in WP1 (`service.py`'s `default_runner` 501s `atlas`/`swapset`
# and `rewired:<seed>`) -- copied now so WP2 does not need to touch this
# Dockerfile again. `scripts/analysis/graph_io.py` locates
# `scripts/data/` relative to its own file location
# (`Path(__file__).resolve().parents[1] / "data"`), so the two directories
# must stay siblings under a common `scripts/` parent exactly as in the
# repo -- not flattened. `env_guard.py` (imported by `transfer.py`) and
# `fsutil.py` (imported by `graph_io.py`) are included even though the
# plan's own file list named only `swap_ops.py`/`graph_io.py`/
# `transfer.py`/`rewire.py`/`binfmt.py`: both are real, unlisted transitive
# dependencies (confirmed by reading their imports), and omitting them
# would make this copy import-broken the moment WP2 exercises it.
COPY training/src/flyarena_training /opt/graph-lab/flyarena_training
COPY scripts/data/binfmt.py scripts/data/fsutil.py scripts/data/rewire.py /opt/graph-lab/scripts/data/
COPY scripts/analysis/env_guard.py scripts/analysis/swap_ops.py scripts/analysis/graph_io.py scripts/analysis/transfer.py /opt/graph-lab/scripts/analysis/

# --- Graph artifacts, read-only. Used today by the GPU atlas smoke test
# (`flyarena_training.atlas_cli`, run directly -- see the smoke-test note
# in this WP's report) and, in WP2, by the job engines.
COPY public/data/malecns-arena-v1.bin.gz public/data/malecns-arena-v1.manifest.json public/data/rewiring-null-v1.json public/data/pathway-interventions-v1.json /opt/graph-lab/data/

# `DD_IAST_ENABLED=false` is load-bearing, not copied boilerplate: the
# host's Datadog auto-instrumentation is force-injected into every process
# on this box (`/etc/ld.so.preload`) regardless of this image's own
# `DD_TRACE_ENABLED=false`. With IAST left on, its taint-tracking aspect
# rewrite of `Tensor.expand()` breaks `flyarena_training.sensors`'s
# `_nearest_egocentric_batch` with `RuntimeError: Boolean value of Tensor
# with more than one value is ambiguous` -- reproduced running the GPU
# atlas smoke test in this container before this line was added.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torch-cache \
    TRITON_CACHE_DIR=/tmp/triton-cache \
    DD_TRACE_ENABLED=false \
    DD_CIVISIBILITY_ENABLED=false \
    DD_INSTRUMENTATION_TELEMETRY_ENABLED=false \
    DD_IAST_ENABLED=false \
    GRAPH_LAB_DATA_DIR=/opt/graph-lab/data \
    GRAPH_LAB_BUNDLE_DIR=/opt/graph-lab/js \
    PYTHONPATH=/opt/graph-lab \
    NODE_OPTIONS=

USER 65532:65532
ENTRYPOINT ["python", "-m", "uvicorn", "graph_lab.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "35"]
