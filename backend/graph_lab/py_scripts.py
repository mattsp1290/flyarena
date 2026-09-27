"""Load the repo's `scripts/data`/`scripts/analysis` Python modules
(`binfmt`, `rewire`, `swap_ops`) from wherever this process's `scripts_dir`
points -- `/opt/graph-lab/scripts` in the container
(`backend/graph_lab/Dockerfile`'s own `COPY scripts/data/... scripts/analysis/...`
layout, which mirrors the repo's real `scripts/{data,analysis}` structure
exactly) or the repo's own `scripts/` locally and in tests.

These modules are plain scripts, not an installed package (`rewire.py`
itself does `sys.path.insert(0, str(Path(__file__).resolve().parent))`
before `import binfmt`, and `swap_ops.py` does the same for its own sibling
`data/` directory) -- this file adds the *other* directory each one needs
(`scripts_dir/data` for `rewire`, both `scripts_dir/data` and
`scripts_dir/analysis` for `swap_ops`, which itself imports `binfmt`) to
`sys.path` before `import`ing, the same sys.path-then-import pattern those
modules use for each other, never a hand-copied duplicate of either
module's own logic.

Both `engine_lesion.py` (`rewired:<seed>` regeneration) and
`engine_swapset.py` (swap-set graph construction) load their science
modules through this one place so the two engines can't drift on how they
resolve `scripts_dir` (a bespoke sys.path snippet per engine is exactly the
kind of duplication `.agents/plans/graph-lab/02-job-engines.md`'s "reuse the
existing verification paths; do not reimplement science logic" instruction
warns against).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _ensure_on_path(directory: Path) -> None:
    resolved = str(directory)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_rewire(scripts_dir: Path) -> Any:
    """`scripts/data/rewire.py`: `decode_graph_binary`, `rewire_graph`, and
    (via `rewire.binfmt`) the wire-format primitives."""
    _ensure_on_path(scripts_dir / "data")
    import rewire  # noqa: E402  -- see this module's own doc comment

    return rewire


def load_swap_ops(scripts_dir: Path) -> Any:
    """`scripts/analysis/swap_ops.py`: `valid_swap`, `apply_swap`,
    `apply_explicit_swaps`, `random_class_swaps`, `bridge_mask_of`,
    `edge_set_from_graph` (and, via `swap_ops.binfmt`, the same wire-format
    primitives `load_rewire` exposes)."""
    _ensure_on_path(scripts_dir / "data")
    _ensure_on_path(scripts_dir / "analysis")
    import swap_ops  # noqa: E402  -- see this module's own doc comment

    return swap_ops


def load_transfer(scripts_dir: Path) -> Any:
    """`scripts/analysis/transfer.py`: `OUTPUT_POPULATION_INDEX`
    (`{"thrust": 0, "yaw": 1, "brake": 2}`) -- the swap-set M1000
    reproduction check (`backend/graph_lab/tests/test_reproduction.py`)
    needs the same `THRUST_IDX = OUTPUT_POPULATION_INDEX["thrust"]`
    `scripts/analysis/interventions.py` uses to define class M's own
    target mask, rather than a hand-inlined `0` that could silently drift
    from it."""
    _ensure_on_path(scripts_dir / "data")
    _ensure_on_path(scripts_dir / "analysis")
    import transfer  # noqa: E402  -- see this module's own doc comment

    return transfer


def run_step(argv: "list[str]", *, cwd: Path) -> str:
    """Run one nested subprocess step to completion and return its captured
    stdout. Shared by `engine_atlas.py`'s and `engine_swapset.py`'s own
    multi-step drivers (both a job-engine "main" that IS `jobs.py`'s single
    supervised child process, running further steps as its *own* nested
    children -- see either module's doc comment for the full "why").
    `check=True`: a nonzero exit raises `CalledProcessError`, which the
    caller's own top-level handler reports through its `error` line --
    never lets a nested step's raw stderr (which can quote its own argv,
    and argv can carry request-derived data) reach a client verbatim,
    matching `jobs.py`'s own stderr-never-verbatim policy for the outer
    job. No `start_new_session` here (unlike `jobs.py`'s own top-level
    `Popen`): every nested step stays in the *same* process group jobs.py
    already put the outer process in, so a cancel/timeout's single
    `os.killpg` reaches nested steps too."""
    completed = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, check=True)
    return completed.stdout


def find_error_message(stdout: str) -> "str | None":
    """Scan `stdout` (from the end) for a `{"type": "error", "message":
    "..."}` line, returning just its `message` -- `None` if no such line is
    present (as opposed to `last_json_message`, which raises either way and
    so can't distinguish "found an error line" from "found nothing")."""
    for line in reversed([raw for raw in stdout.splitlines() if raw.strip()]):
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("type") == "error":
            return str(message.get("message", "failed"))
    return None


def describe_step_failure(error: "subprocess.CalledProcessError") -> str:
    """A nested step's `CalledProcessError.stderr`/`.cmd` can quote its own
    argv, which can carry request-derived data (`run_step`'s own doc
    comment) -- never surfaced verbatim. But every graph-lab entry prints
    its own already-sanitized `{"type": "error", "message": "entry-X: ..."}`
    line to *stdout* before exiting non-zero (a dual-review finding: the
    generic "program exited with status N" fallback this replaces threw
    that message away, which is exactly how one real bug -- a `disconnected`
    atlas job's identity mismatch -- stayed hidden behind an uninformative
    error during this WP's own development). Prefer that message when
    present; fall back to the generic, path-free summary otherwise."""
    program = Path(error.cmd[0]).name
    message = find_error_message(error.stdout or "")
    if message is not None:
        return f"{program}: {message}"
    return f"{program} step exited with status {error.returncode}"


def last_json_message(stdout: str, *, source: str) -> "dict[str, Any]":
    """Every graph-lab entry can print progress lines before its final
    `result`/`error` line (`entry-lesion.ts`/`entry-swapset.ts`'s own
    `printProgress`) -- scan from the end for the last line that parses as
    one of this protocol's two terminal shapes, rather than assuming the
    *last* line of output is always it. Raises `RuntimeError` (never
    returns a partial/`None` result) on an `error` message or if no
    terminal message is found at all."""
    for line in reversed([raw for raw in stdout.splitlines() if raw.strip()]):
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        if message.get("type") == "error":
            raise RuntimeError(f"{source}: {message.get('message', 'failed')}")
        if message.get("type") == "result":
            return message["result"]
    raise RuntimeError(f"{source}: produced no result line")
