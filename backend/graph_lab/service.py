"""Private real-graph lab API: `/api/graph/v1`.

Reuses `flyarena_lab/service.py`'s security patterns directly (Bearer with
`hmac.compare_digest`, a body-limit ASGI middleware, CORS from an explicit
origin list with `*` rejected, one active job / four retained), plus two
things the synthetic lab doesn't need: Private Network Access (PNA) --
the real backend is reached from a public page's private-network fetch,
`.agents/plans/graph-lab/00-overview.md`'s WP5 -- via `CORSMiddleware`'s
own `allow_private_network` (see the security-review note on
`CORSMiddleware` below for why this isn't a hand-rolled middleware) and a
job store that supervises a real child **process** rather than an
in-process engine call (`jobs.Jobs`).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from . import engine_lesion, engine_swapset
from .jobs import Jobs, Runner, parse_graph_id
from .models import AtlasJobRequest, JobRequest, LesionJobRequest, SwapsetJobRequest

#: Refuse an atlas job (a real GPU search) if free GPU memory reported by
#: `torch.cuda.mem_get_info()` is below this (`02-job-engines.md`'s atlas
#: bound: "refuses to start... if free GPU memory is below 2 GiB"). Read
#: with `torch.cuda.mem_get_info()` specifically (not `nvidia-smi`, which
#: reports "Not Supported" for per-process/global memory on this host's
#: GB10 unified-memory architecture -- confirmed empirically; `torch`'s own
#: CUDA driver query works regardless).
GPU_BUSY_THRESHOLD_BYTES = 2 * 1024**3

MODEL_VERSION = "arena-graph-lab-v1"
MAX_BODY_BYTES = 16 * 1024
MIN_TOKEN_LENGTH = 16


class BodyLimit:
    """Byte-for-byte the same buffering approach as `flyarena_lab/service.py`'s
    `BodyLimit`, at 16 KiB instead of 4 KiB (`02-job-engines.md`'s request
    bodies -- up to 32 lesion sets or 50 swaps -- are larger than the
    synthetic lab's)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > MAX_BODY_BYTES:
                return await JSONResponse({"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"}, 413)(
                    scope, receive, send
                )
            chunks.append(message)
            if not message.get("more_body", False):
                break

        async def buffered():
            if chunks:
                return chunks.pop(0)
            return await receive()

        await self.app(scope, buffered, send)


def _parse_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        raise ValueError("GRAPH_LAB_ORIGINS must list explicit origins")
    if not origins:
        raise ValueError("GRAPH_LAB_ORIGINS must list at least one explicit origin")
    return origins


def default_runner(
    bundle_dir: Path,
    data_dir: Path,
    node_bin: str = "node",
    scripts_dir: "Path | None" = None,
    python_bin: "str | None" = None,
    atlas_python_bin: "str | None" = None,
    atlas_device: str = "cuda",
    gpu_free_bytes: "Callable[[], int | None] | None" = None,
) -> Runner:
    """The real production runner: maps a validated job request to an argv
    list that runs the matching esbuild-bundled entry script (or, for
    `atlas`, `graph_lab.engine_atlas`'s own multi-step driver -- see
    below). Wires all three job kinds
    (`.agents/plans/graph-lab/02-job-engines.md`).

    `scripts_dir`/`python_bin`/`atlas_python_bin`/`gpu_free_bytes` are
    injectable the same way `bundle_dir`/`data_dir`/`node_bin` already are
    (WP1's own convention), so tests can point them at fixtures/fakes
    without a real container, GPU, or `training/`-venv path:
    - `scripts_dir` (default `$GRAPH_LAB_SCRIPTS_DIR` or
      `/opt/graph-lab/scripts`): where `scripts/data/rewire.py` and
      `scripts/analysis/swap_ops.py` live (`py_scripts.py`'s own doc
      comment) -- mirrors the repo's own `scripts/{data,analysis}` layout,
      both in the container and locally.
    - `python_bin` (default `sys.executable`): the interpreter that runs
      `graph_lab.engine_atlas` itself -- always an interpreter with
      `graph_lab` importable (the one running this very process, in the
      container and in `uv run pytest` alike).
    - `atlas_python_bin` (default `$GRAPH_LAB_ATLAS_PYTHON_BIN` or
      `python_bin`): the interpreter `engine_atlas.py` shells out to for
      `python -m flyarena_training.atlas_cli` -- the *same* interpreter in
      the container (one merged environment; `Dockerfile`'s own header
      comment), but a distinct `training/`-venv path locally/in tests
      (`graph_lab`'s own venv deliberately has no `torch`; see
      `pyproject.toml`'s header comment).
    - `gpu_free_bytes` (default `_real_gpu_free_bytes`, `torch.cuda.
      mem_get_info()`): read once per atlas submission, before any
      subprocess starts.
    """
    resolved_scripts_dir = scripts_dir or Path(os.environ.get("GRAPH_LAB_SCRIPTS_DIR", "/opt/graph-lab/scripts"))
    resolved_python_bin = python_bin or sys.executable
    resolved_atlas_python_bin = atlas_python_bin or os.environ.get("GRAPH_LAB_ATLAS_PYTHON_BIN", resolved_python_bin)
    check_gpu_free_bytes = gpu_free_bytes or _real_gpu_free_bytes

    def _biological_manifest() -> dict:
        return json.loads((data_dir / "malecns-arena-v1.manifest.json").read_text())

    def _lesion_argv(request: LesionJobRequest, job_dir: Path) -> "list[str]":
        mode, seed = parse_graph_id(request.graph)
        manifest = _biological_manifest()
        if mode == "rewired":
            assert seed is not None  # parse_graph_id guarantees this for mode == "rewired"
            try:
                graph_path, expected_sha256 = engine_lesion.regenerate_rewired_graph(
                    scripts_dir=resolved_scripts_dir, data_dir=data_dir, job_dir=job_dir, seed=seed
                )
            except Exception as error:  # noqa: BLE001 -- any regeneration/verification failure is a real 500, never left to bubble to jobs.py's generic handler with a less specific message.
                raise HTTPException(500, f"Rewired graph regeneration failed: {_rewired_regeneration_error_detail(error)}") from error
        else:
            graph_path = data_dir / manifest["artifact"]
            expected_sha256 = manifest["binarySha256"]
        args = {
            "dataDir": str(data_dir),
            # `entry-lesion.ts`'s own `LesionArgs.mode` only ever accepts
            # `'biological'`/`'disconnected'` -- a materialized `rewired`
            # graph binary is loaded exactly like a biological one
            # (`buildGraphBufferForMode`'s `'biological'`/`'rewired'`
            # branches both just pass their own buffer through
            # unmodified), so a regenerated `rewired:<seed>` graph is
            # served here as `mode: "biological"` pointed at its own
            # tmpfs path -- never `mode: "rewired"`, which this entry
            # would reject.
            "mode": "biological" if mode == "rewired" else mode,
            "graphPath": str(graph_path),
            "expectedSha256": expected_sha256,
            "sets": request.sets,
            "seedStart": request.seed_start,
            "seedCount": request.seed_count,
            "ticks": request.ticks,
        }
        args_path = job_dir / "args.json"
        args_path.write_text(json.dumps(args))
        return [node_bin, str(bundle_dir / "entry-lesion.mjs"), str(args_path)]

    def _atlas_argv(request: AtlasJobRequest, job_dir: Path) -> "list[str]":
        free_bytes = check_gpu_free_bytes()
        if free_bytes is None or free_bytes < GPU_BUSY_THRESHOLD_BYTES:
            raise HTTPException(503, "GPU busy")
        mode, seed = parse_graph_id(request.graph)
        manifest = _biological_manifest()
        biological_path = data_dir / manifest["artifact"]
        rewired_path = None
        expected_sha256 = manifest["binarySha256"] if mode == "biological" else None
        if mode == "rewired":
            assert seed is not None
            try:
                rewired_path, expected_sha256 = engine_lesion.regenerate_rewired_graph(
                    scripts_dir=resolved_scripts_dir, data_dir=data_dir, job_dir=job_dir, seed=seed
                )
            except Exception as error:  # noqa: BLE001 -- any regeneration/verification failure is a real 500, never left to bubble to jobs.py's generic handler with a less specific message.
                raise HTTPException(500, f"Rewired graph regeneration failed: {_rewired_regeneration_error_detail(error)}") from error
        args = {
            "jobDir": str(job_dir),
            "nodeBin": node_bin,
            "bundleDir": str(bundle_dir),
            "dataDir": str(data_dir),
            "atlasPythonBin": resolved_atlas_python_bin,
            "graphPath": str(biological_path),
            "rewiredPath": str(rewired_path) if rewired_path else None,
            "arm": mode,
            "expectedSha256": expected_sha256,
            "graphArtifactSha256": manifest["gzipSha256"],
            "biologicalGraphPath": str(biological_path),
            "biologicalExpectedSha256": manifest["binarySha256"],
            "device": atlas_device,
            "searchSeed": request.search_seed,
            "population": request.population,
            "generations": request.generations,
            "ticks": request.ticks,
        }
        args_path = job_dir / "atlas-args.json"
        args_path.write_text(json.dumps(args))
        return [resolved_python_bin, "-m", "graph_lab.engine_atlas", str(args_path)]

    def _swapset_argv(request: SwapsetJobRequest, job_dir: Path) -> "list[str]":
        swaps = [(swap.a, swap.b, swap.c, swap.d) for swap in request.swaps]
        # Only the fast half runs synchronously here, inside `Jobs.submit()`'s
        # lock: `build_candidate` validates the swap list against the real
        # graph (one swap application, one rebuild) and can still give a
        # clean 422 before the job is accepted. Building the `controls`
        # random controls (up to 100 `random_class_swaps` calls) is the
        # expensive half -- measured at up to ~46s at this request kind's
        # own upper bounds -- and would otherwise block `Jobs.get` (status
        # polling) and cancellation for every other client for that whole
        # window (a dual-review finding); it now runs inside
        # `engine_swapset.run`, this job's own supervised async subprocess,
        # where the existing wall-clock ceiling and cancellation already
        # apply to it.
        try:
            built = engine_swapset.build_candidate(
                scripts_dir=resolved_scripts_dir,
                data_dir=data_dir,
                job_dir=job_dir,
                swaps=swaps,
            )
        except ValueError as error:
            raise HTTPException(422, f"Invalid swap set: {error}") from error
        args = {
            "jobDir": str(job_dir),
            "nodeBin": node_bin,
            "entryPath": str(bundle_dir / "entry-swapset.mjs"),
            "dataDir": str(data_dir),
            "scriptsDir": str(resolved_scripts_dir),
            "biologicalGraph": built["biologicalGraph"],
            "candidateGraph": built["candidateGraph"],
            "sourceIndices": built["sourceIndices"],
            "targetIndices": built["targetIndices"],
            "k": built["k"],
            "controls": request.controls,
            "seedStart": request.seed_start,
            "seedCount": request.seed_count,
            "ticks": request.ticks,
            "publishedNullPath": str(data_dir / "rewiring-null-v1.json"),
        }
        args_path = job_dir / "swapset-args.json"
        args_path.write_text(json.dumps(args))
        return [resolved_python_bin, "-m", "graph_lab.engine_swapset", str(args_path)]

    def runner(request: JobRequest, job_dir: Path) -> list[str]:
        if isinstance(request, LesionJobRequest):
            return _lesion_argv(request, job_dir)
        if isinstance(request, AtlasJobRequest):
            return _atlas_argv(request, job_dir)
        if isinstance(request, SwapsetJobRequest):
            return _swapset_argv(request, job_dir)
        raise HTTPException(400, "Unrecognized job kind")

    return runner


def _bundle_sha256(bundle_dir: Path) -> str | None:
    import json

    bundle_json = bundle_dir / "bundle.json"
    if not bundle_json.is_file():
        return None
    try:
        return json.loads(bundle_json.read_text()).get("bundleSha256")
    except (json.JSONDecodeError, OSError):
        return None


def _graph_sha256(data_dir: Path) -> str | None:
    import json

    manifest_path = data_dir / "malecns-arena-v1.manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text()).get("binarySha256")
    except (json.JSONDecodeError, OSError):
        return None


def _gpu_available() -> bool:
    try:
        import torch  # optional: not a graph_lab dependency; provided by the base image at runtime
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _rewired_regeneration_error_detail(error: Exception) -> str:
    """`engine_lesion.regenerate_rewired_graph` raises `ValueError` for
    every check it does itself (sha mismatches, a missing published-null
    entry) -- those messages only ever echo seeds and sha256 hex digests
    (audited), safe to return verbatim. Anything else (`OSError`/
    `FileNotFoundError` reading a missing/corrupt data file, a
    `json.JSONDecodeError`, ...) can include an absolute server-side path
    in its own `str()` (confirmed: a security review finding) -- those get
    a fixed, generic message instead, with the real exception logged
    server-side via `raise ... from error` (still visible in the server's
    own logs/traceback, never in the HTTP response body)."""
    if isinstance(error, ValueError):
        return str(error)
    logging.exception("graph-lab: rewired graph regeneration failed")
    return "an internal error occurred while preparing the graph"


def _real_gpu_free_bytes() -> "int | None":
    """`None` means "unknown" (no `torch`, no CUDA device, or the query
    itself raised) -- `default_runner`'s atlas branch treats that the same
    as "known busy": refuse to start rather than silently proceeding
    without ever having checked (`gpu_free_bytes` is injectable precisely
    so `test_engines.py`/`test_security.py`, which run in `graph_lab`'s own
    CPU-only venv with no `torch` at all, can exercise both the "GPU free"
    and "GPU busy" branches without a real GPU)."""
    try:
        import torch  # optional: see `_gpu_available`'s own comment
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes)
    except Exception:
        return None


def create_app(
    token: str | None = None,
    runner: Runner | None = None,
    data_dir: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    node_bin: str = "node",
    scripts_dir: "str | Path | None" = None,
    python_bin: "str | None" = None,
    atlas_python_bin: "str | None" = None,
    atlas_device: str = "cuda",
    gpu_free_bytes: "Callable[[], int | None] | None" = None,
) -> FastAPI:
    secret = token if token is not None else os.environ.get("GRAPH_LAB_TOKEN", "")
    resolved_data_dir = Path(data_dir or os.environ.get("GRAPH_LAB_DATA_DIR", "/opt/graph-lab/data"))
    resolved_bundle_dir = Path(bundle_dir or os.environ.get("GRAPH_LAB_BUNDLE_DIR", "/opt/graph-lab/js"))
    active_runner = (
        runner
        if runner is not None
        else default_runner(
            resolved_bundle_dir,
            resolved_data_dir,
            node_bin,
            scripts_dir=Path(scripts_dir) if scripts_dir else None,
            python_bin=python_bin,
            atlas_python_bin=atlas_python_bin,
            atlas_device=atlas_device,
            gpu_free_bytes=gpu_free_bytes,
        )
    )
    jobs = Jobs(active_runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if len(secret) < MIN_TOKEN_LENGTH:
            raise RuntimeError(f"Set GRAPH_LAB_TOKEN to at least {MIN_TOKEN_LENGTH} characters")
        yield
        jobs.close()

    # `docs_url`/`redoc_url`/`openapi_url` all `None`: FastAPI's defaults
    # serve the OpenAPI schema and interactive docs unauthenticated (they
    # sit outside `authorize`'s `Depends`, same as `/health`) -- a real
    # information leak for a private, unpublished API (endpoint names,
    # field names, and every bound in models.py), and `/docs`/`/redoc` also
    # pull their JS from a CDN. A security-review finding: verified these
    # three paths returned 200 with no token before this was added.
    app = FastAPI(
        title="FlyArena private graph lab",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.jobs = jobs
    app.add_middleware(BodyLimit)

    origins = _parse_origins(os.environ.get("GRAPH_LAB_ORIGINS", "http://127.0.0.1:5173,http://127.0.0.1:4173"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        # `allow_private_network=True`: Starlette's own `CORSMiddleware`
        # already implements the Private Network Access (PNA) preflight
        # check natively (both the locally-pinned Starlette this project's
        # own tests run against, and the base image's own bundled Starlette
        # -- the graph-lab container installs with `--no-deps`, so it runs
        # whichever Starlette the base image already has). A hand-rolled
        # ASGI middleware answering the PNA header on top of this was
        # actively *wrong*, not merely redundant: when
        # `Access-Control-Request-Private-Network: true` is present and
        # `allow_private_network` is left at its default `False`,
        # `CORSMiddleware` itself already treats that as a preflight
        # *failure* and returns 400 -- a non-2xx preflight response the
        # browser rejects regardless of any extra header a second
        # middleware might append afterward. A hand-rolled middleware
        # wrapping `CORSMiddleware` from the outside can decorate that 400
        # response with the header, but can never turn it back into the
        # 200 a real browser's PNA fetch needs. `allow_private_network=True`
        # is the only thing that makes `CORSMiddleware` return 200. It is
        # still gated on the origin allowlist exactly like every other CORS
        # decision here: `is_allowed_origin` is checked independently, and
        # a disallowed origin's preflight still fails (400) on "origin"
        # regardless of this flag.
        allow_private_network=True,
    )

    def authorize(authorization: str = Header(default="")) -> None:
        # `hmac.compare_digest` (not `==`): a timing side channel on token
        # comparison would let a network attacker recover the token one
        # byte at a time from response-time differences.
        if not secret or not hmac.compare_digest(authorization.encode(), f"Bearer {secret}".encode()):
            raise HTTPException(401, "A valid bearer token is required")

    @app.get("/api/graph/v1/health")
    def health():
        # Unauthenticated by design (the frontend needs it to show
        # "backend unavailable" before a token is even entered) -- returns
        # only non-sensitive fields: no free-GPU-memory figure, no request
        # counts, no job ids. `.agents/plans/graph-lab/01-service-and-container.md`
        # documents this as an accepted, tailnet-scoped residual exposure.
        return {
            "status": "ok",
            "modelVersion": MODEL_VERSION,
            "bundleSha256": _bundle_sha256(resolved_bundle_dir),
            "graphSha256": _graph_sha256(resolved_data_dir),
            "gpu": {"available": _gpu_available()},
        }

    @app.post("/api/graph/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    def submit(request: JobRequest):
        return jobs.submit(request)

    @app.get("/api/graph/v1/jobs/{identifier}", dependencies=[Depends(authorize)])
    def status(identifier: str):
        return jobs.get(identifier)

    @app.delete("/api/graph/v1/jobs/{identifier}", dependencies=[Depends(authorize)])
    def cancel(identifier: str):
        return jobs.get(identifier, cancel=True)

    return app
