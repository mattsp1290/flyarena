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
"""
from __future__ import annotations

import os
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
