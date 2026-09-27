"""`.agents/plans/selection-robustness/02-per-selection-chain.md` WP2:
`scripts/selections/run-chain.sh --dry-run-fixture` -- a dual-review finding
on an earlier version of this branch: the plan's change-surface table
requires a test for this flag, and none existed. This does not run any real
compute (that is the whole point of `--dry-run-fixture`); it asserts the
orchestration script's own structural self-check actually checks something
(selection-id validation, and that every command it would run is logged
under `training/runs/selections/<id>/`, never under `public/`/`docs/`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_CHAIN_SCRIPT = REPO_ROOT / "scripts" / "selections" / "run-chain.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUN_CHAIN_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_dry_run_fixture_exits_zero_for_every_predeclared_selection() -> None:
    for selection in ("default", "larger", "smaller", "random-bridge", "alt-sensory-mapping"):
        result = _run(selection, "--dry-run-fixture")
        assert result.returncode == 0, f"selection={selection!r}: {result.stderr}"


def test_dry_run_fixture_rejects_an_unknown_selection_id() -> None:
    result = _run("no-such-selection", "--dry-run-fixture")
    assert result.returncode != 0
    assert "unknown selection id" in result.stderr


def test_dry_run_fixture_emits_every_step_command_under_the_selection_scratch_tree_only() -> None:
    result = _run("larger", "--dry-run-fixture")
    assert result.returncode == 0, result.stderr
    out = result.stdout

    for needle in (
        "rewire_batch.py --in-path",
        "null:evaluate -- --biological",
        "--decoder authored-flip-both",
        "null:report -- --authored",
        "--variant-out",
        "--trained",  # the trained-arm contamination fix (I2): must be present and point under $B
        "transfer.py --index",
        "features.py --index",
        "null:regime-check --",
        "explain.py --selection-mode",
        "coverage.py --graph",
        "interventions.py --biological",
        "--graph-list",
        "intervention:report -- --authored",
        "--stats-only --arena-task default",
    ):
        assert needle in out, f"missing {needle!r} in dry-run output"

    # The hard safety requirement, checked at the level this script owns:
    # every logged command must confine its writes to this selection's own
    # scratch tree, never public/ or docs/.
    assert "training/runs/selections/larger" in out
    assert "public/" not in out
    assert "docs/" not in out


def test_dry_run_fixture_requires_a_selection_argument() -> None:
    result = _run("--dry-run-fixture")
    assert result.returncode != 0
