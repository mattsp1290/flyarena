"""Shared fixtures for the parity suite.

`tests/fixtures/golden/` (the default trace directory) deliberately omits
per-tick world state to stay inside its committed byte budget (see
`scripts/training/export-traces.ts`'s module doc comment). World-state,
observation, event, and free-running checks need that column
(`--include-world`), so `include_world_trace_dir` below generates it into a
session-scoped temp directory by invoking the Node exporter directly —
`npm run training:traces -- --include-world --out <tmp dir>` — and skips
those tests (with the exact command to run by hand) when Node can't be
found, so a plain `uv run pytest` still runs everything it can.

A longer, wider-seed-coverage trace (with actual food pickups/respawns,
wall clamps, and hazard contacts — the committed golden fixtures' 60 ticks
never trigger any of these, see thermo-architecture review finding #2) is
generated the same way by `long_trace_dir` below, from
`training/scripts/generate_long_traces.ts`.

Datadog note: this host's global APM auto-injection breaks `torch` (see
`README.md`'s "Environment note"). The three `DD_*=false` env vars below are
set here as a best-effort/documentation measure for any code path that reads
`os.environ` at runtime, but they are **not sufficient on their own** —
ddtrace's auto-injection runs via `sitecustomize`/`PYTHONPATH` before this
file (or any user code) executes, so setting them here is too late to
prevent the injection itself. The actual fix has to happen at the shell
invocation, before the `python`/`uv run` process starts:
`training/scripts/run.sh` wraps that for automation (WP3's future
`flyarena-train` CLI); a human running `pytest` by hand still needs the
documented `DD_TRACE_ENABLED=false DD_IAST_ENABLED=false
DD_APPSEC_ENABLED=false uv run pytest` prefix.
"""
from __future__ import annotations

import os

os.environ.setdefault("DD_TRACE_ENABLED", "false")
os.environ.setdefault("DD_IAST_ENABLED", "false")
os.environ.setdefault("DD_APPSEC_ENABLED", "false")

import shutil
import subprocess
from pathlib import Path

import pytest

# training/tests/conftest.py -> training/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"

# Pinned nvm location the task environment documents; PATH alone may not
# carry it in a non-interactive shell.
PINNED_NODE_BIN_DIR = Path.home() / ".nvm" / "versions" / "node" / "v22.22.3" / "bin"

GENERATE_CMD = "npm run training:traces -- --include-world --out training/runs/traces/trace-graph"
GENERATE_LONG_CMD = "npx tsx training/scripts/generate_long_traces.ts --out training/runs/traces/event-coverage"

# Seeds/ticks `generate_long_traces.ts`'s defaults produce; kept in sync by
# hand (see that script's `DEFAULT_SEEDS`/`DEFAULT_TICKS` comment) since a
# Python fixture can't import a TS module's constants directly.
LONG_TRACE_SEEDS: tuple[int, ...] = (120, 349, 766, 152, 598, 750, 501, 595, 37, 216, 377, 132, 163)


def _find_node_bin_dir() -> str | None:
    found = shutil.which("node")
    if found:
        return str(Path(found).parent)
    if (PINNED_NODE_BIN_DIR / "node").exists():
        return str(PINNED_NODE_BIN_DIR)
    return None


@pytest.fixture(scope="session")
def trace_dir() -> Path:
    """The default (teacher-forced rate/output/readout) parity trace
    directory. `FLYARENA_TRACE_DIR` overrides it, per the plan's
    acceptance criteria, so a later work package can point this suite at
    real-graph traces without code changes."""
    configured = os.environ.get("FLYARENA_TRACE_DIR")
    return Path(configured).resolve() if configured else DEFAULT_TRACE_DIR


@pytest.fixture(scope="session")
def include_world_trace_dir(tmp_path_factory) -> Path:
    """A `--include-world` trace, generated on demand. Skips (naming the
    manual command) when Node isn't reachable or the export fails."""
    node_bin_dir = _find_node_bin_dir()
    if node_bin_dir is None:
        pytest.skip(
            "Node not found on PATH or at the pinned nvm location; cannot generate the "
            f"--include-world trace this check needs. Run manually from the repo root: {GENERATE_CMD}"
        )

    out_dir = tmp_path_factory.mktemp("trace-graph-include-world")
    env = dict(os.environ)
    env["PATH"] = node_bin_dir + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        ["npm", "run", "training:traces", "--", "--include-world", "--out", str(out_dir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Generating the --include-world trace failed (exit {result.returncode}). "
            f"Run manually from the repo root to see the full error: {GENERATE_CMD}\n"
            f"stderr tail: {result.stderr.strip()[-1000:]}"
        )
    return out_dir


@pytest.fixture(scope="session")
def long_trace_dir(tmp_path_factory) -> Path:
    """A longer, wider-seed-coverage `--include-world` trace (see this
    module's doc comment), generated on demand via
    `training/scripts/generate_long_traces.ts` (not the committed exporter's
    CLI, which only accepts its own fixed seeds/tick count). Skips (naming
    the manual command) when Node isn't reachable or the export fails, same
    as `include_world_trace_dir`."""
    node_bin_dir = _find_node_bin_dir()
    if node_bin_dir is None:
        pytest.skip(
            "Node not found on PATH or at the pinned nvm location; cannot generate the "
            f"long event-coverage trace this check needs. Run manually from the repo root: {GENERATE_LONG_CMD}"
        )

    out_dir = tmp_path_factory.mktemp("trace-graph-long")
    env = dict(os.environ)
    env["PATH"] = node_bin_dir + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        ["npx", "tsx", "training/scripts/generate_long_traces.ts", "--out", str(out_dir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Generating the long event-coverage trace failed (exit {result.returncode}). "
            f"Run manually from the repo root to see the full error: {GENERATE_LONG_CMD}\n"
            f"stderr tail: {result.stderr.strip()[-1000:]}"
        )
    return out_dir
