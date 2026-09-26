"""Private real-graph lab API: `/api/graph/v1`.

Reuses `flyarena_lab/service.py`'s security patterns directly (Bearer with
`hmac.compare_digest`, a body-limit ASGI middleware, CORS from an explicit
origin list with `*` rejected, one active job / four retained), plus two
things the synthetic lab doesn't need: a `PrivateNetworkAccess` middleware
(the real backend is reached from a public page's private-network fetch,
`.agents/plans/graph-lab/00-overview.md`'s WP5) and a job store that
supervises a real child **process** rather than an in-process engine call
(`jobs.Jobs`).
"""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from .jobs import Jobs, Runner, parse_graph_id
from .models import AtlasJobRequest, JobRequest, LesionJobRequest, SwapsetJobRequest

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


class PrivateNetworkAccess:
    """Answers a CORS preflight's `Access-Control-Request-Private-Network:
    true` (Chromium's Private Network Access check, sent by a public page
    fetching a tailnet-private address) with `Access-Control-Allow-Private-Network:
    true` -- but only alongside an already-CORS-allowed origin. Runs *after*
    `CORSMiddleware` in the ASGI stack (added first, so it wraps innermost
    and its response passes back out through this one), and only touches
    `OPTIONS` preflights that already carry an `Access-Control-Allow-Origin`
    header from `CORSMiddleware` -- a disallowed origin's preflight is
    passed through unchanged, still with no CORS allowance, and never gets
    the private-network header either.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "OPTIONS":
            return await self.app(scope, receive, send)

        wants_private_network = any(
            name.decode("latin-1").lower() == "access-control-request-private-network"
            and value.decode("latin-1").lower() == "true"
            for name, value in scope.get("headers", [])
        )
        if not wants_private_network:
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                already_allowed = any(
                    name.decode("latin-1").lower() == "access-control-allow-origin" for name, _ in headers
                )
                if already_allowed:
                    headers.append((b"access-control-allow-private-network", b"true"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


def _parse_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        raise ValueError("GRAPH_LAB_ORIGINS must list explicit origins")
    if not origins:
        raise ValueError("GRAPH_LAB_ORIGINS must list at least one explicit origin")
    return origins


def default_runner(bundle_dir: Path, data_dir: Path, node_bin: str = "node") -> Runner:
    """The real production runner: maps a validated job request to an argv
    list that runs the matching esbuild-bundled entry script. WP1 wires only
    `kind: "lesion"` end to end, and only for `graph` values that need no
    Python-side graph preparation (`biological`/`disconnected`) -- `atlas`,
    `swapset`, and `rewired:<seed>` need engine work
    (`.agents/plans/graph-lab/02-job-engines.md`: GPU search + re-eval,
    swap-op graph construction, and `rewire.py` regeneration respectively)
    that is out of scope for this work package and ships in WP2. Those
    kinds are still fully validated by `models.py` (including the
    closed-set `graph` check) and rejected here with a clear, static 501 --
    never silently accepted and never a 500 from an engine that doesn't
    exist yet.
    """
    import json

    def runner(request: JobRequest, job_dir: Path) -> list[str]:
        if isinstance(request, LesionJobRequest):
            mode, seed = parse_graph_id(request.graph)
            if mode == "rewired":
                raise HTTPException(501, "rewired graph regeneration is not wired until WP2")
            manifest_path = data_dir / "malecns-arena-v1.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            args = {
                "dataDir": str(data_dir),
                "mode": mode,
                "graphPath": str(data_dir / manifest["artifact"]),
                "expectedSha256": manifest["binarySha256"],
                "sets": request.sets,
                "seedStart": request.seed_start,
                "seedCount": request.seed_count,
                "ticks": request.ticks,
            }
            args_path = job_dir / "args.json"
            args_path.write_text(json.dumps(args))
            return [node_bin, str(bundle_dir / "entry-lesion.mjs"), str(args_path)]
        if isinstance(request, (AtlasJobRequest, SwapsetJobRequest)):
            raise HTTPException(501, f"job kind {request.kind!r} engine is not wired until WP2")
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


def create_app(
    token: str | None = None,
    runner: Runner | None = None,
    data_dir: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    node_bin: str = "node",
) -> FastAPI:
    secret = token if token is not None else os.environ.get("GRAPH_LAB_TOKEN", "")
    resolved_data_dir = Path(data_dir or os.environ.get("GRAPH_LAB_DATA_DIR", "/opt/graph-lab/data"))
    resolved_bundle_dir = Path(bundle_dir or os.environ.get("GRAPH_LAB_BUNDLE_DIR", "/opt/graph-lab/js"))
    active_runner = runner if runner is not None else default_runner(resolved_bundle_dir, resolved_data_dir, node_bin)
    jobs = Jobs(active_runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if len(secret) < MIN_TOKEN_LENGTH:
            raise RuntimeError(f"Set GRAPH_LAB_TOKEN to at least {MIN_TOKEN_LENGTH} characters")
        yield
        jobs.close()

    app = FastAPI(title="FlyArena private graph lab", lifespan=lifespan)
    app.state.jobs = jobs
    app.add_middleware(BodyLimit)

    origins = _parse_origins(os.environ.get("GRAPH_LAB_ORIGINS", "http://127.0.0.1:5173,http://127.0.0.1:4173"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    # Added after CORSMiddleware so it sits *outside* CORSMiddleware in the
    # middleware stack (FastAPI/Starlette wraps middlewares in the reverse
    # of `add_middleware` call order) and can inspect the CORS response
    # CORSMiddleware already produced.
    app.add_middleware(PrivateNetworkAccess)

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
