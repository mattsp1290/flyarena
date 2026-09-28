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

import hashlib
import json
import os
import shutil
import stat
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
        "--decoder authored-flip-thrust",
        "--decoder authored-flip-yaw",
        "variant-flip-thrust.json",
        "variant-flip-yaw.json",
        "--variant-flip-thrust",
        "--variant-flip-yaw",
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


# ---------------------------------------------------------------------------
# Real-mode (non-`--dry-run-fixture`) resumability/failure tests (thermo-
# maintainability review finding I2: `--dry-run-fixture` deliberately never
# runs any producer, so it cannot exercise the skip-when-output-exists
# logic that is this script's actual "resumable" behavior, `check_sha`'s
# mismatch path, or the final `git status -- public docs` guard).
#
# Everything below runs against a fully synthetic FAKE repo built under
# `tmp_path` -- never this worktree's own `training/runs/selections/` tree
# (per this review's own hard constraint: a long real chain may be running
# against that same relative path in a sibling worktree). `run-chain.sh`
# derives its own `REPO_ROOT` from `${BASH_SOURCE[0]}`'s location
# (`REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"`), so
# copying the current script to `<fake_repo>/scripts/selections/
# run-chain.sh` and invoking that copy makes every path it touches
# (`training/runs/selections/...`, the final `git status -- public docs`
# check) resolve entirely inside the fake repo.
#
# No real `npm`/`uv` producer is ever invoked: a `PATH` with no-op stub
# `npm`/`uv` executables is prepended for every real-mode run below, so
# even a step that structurally fails to skip can only run the harmless
# stub, never a real, hours-long producer.
# ---------------------------------------------------------------------------

GRAPH_SHA = "a" * 64
NULL_EXPLANATION_BYTES = b'{"stub":true}\n'
NULL_EXPLANATION_SHA256 = hashlib.sha256(NULL_EXPLANATION_BYTES).hexdigest()

# scripts/analysis/explain.py:109 -- run-chain.sh's step 2b parses this exact
# constant out of the real file rather than hardcoding 0.25 itself (see that
# step's own comment). The fake repo below never executes explain.py (the
# `uv` stub just exits 0), so a one-line stub carrying only this assignment
# is enough for the grep to find it -- below the default bioPercentile
# (BIO_PERCENTILE_BELOW_THRESHOLD), so the fully-stubbed base fixture keeps
# step 2b a no-producer skip, same as every other already-computed step.
DECODER_PERCENTILE_THRESHOLD = 0.25
BIO_PERCENTILE_BELOW_THRESHOLD = 0.1
BIO_PERCENTILE_ABOVE_THRESHOLD = 0.294  # the real random-bridge mirrored value


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_stub_bin(bin_dir: Path) -> None:
    """A `npm`/`uv` on `PATH` that immediately exits 0 and does nothing --
    proves a step attempted to run (its `run()` call still logs `+ ...`
    before executing) without ever running a real producer.

    One exception: when invoked with `--variant-out <path>` (as step
    2a/2b's `null:report` calls always are), it writes a minimal stub JSON
    carrying the fixture's own `GRAPH_SHA` as `sourceGraphSha256` to that
    path. Every other step's `check_sha` call is satisfied by the fixture's
    pre-populated files (never touched by the no-op stub); step 2b's
    single-axis outputs are the one case a test needs to force *not*
    pre-existing (to prove the step actually ran), so the stub has to leave
    something on disk for `check_sha` to read immediately afterward, same
    as a real `null-report.ts --variant-out` run would."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_body = (
        "#!/usr/bin/env bash\n"
        'prev=""\n'
        'for arg in "$@"; do\n'
        '  if [[ "$prev" == "--variant-out" ]]; then\n'
        '    mkdir -p "$(dirname "$arg")"\n'
        f'    printf \'{{"sourceGraphSha256":"{GRAPH_SHA}"}}\' > "$arg"\n'
        "  fi\n"
        '  prev="$arg"\n'
        "done\n"
        "exit 0\n"
    )
    for name in ("npm", "uv"):
        stub = bin_dir / name
        stub.write_text(stub_body)
        _make_executable(stub)


def _build_fake_repo(
    tmp_path: Path, selection: str = "default", bio_percentile: float = BIO_PERCENTILE_BELOW_THRESHOLD
) -> tuple[Path, Path]:
    """A synthetic repo with `run-chain.sh` copied to the real script's own
    relative path, and a fully populated `$B` scratch tree: every
    producer's output already present and cross-stamped with `GRAPH_SHA`
    (including the selection's own manifest's `nullExplanation` entry,
    matching `null-explanation.json`'s actual bytes) so a real (non-dry-run)
    invocation skips every step. `scripts/analysis/explain.py` is stubbed
    with only its `DECODER_PERCENTILE_THRESHOLD` assignment -- step 2b greps
    that line out of the real file, and the stub is never executed (the `uv`
    stub below exits 0 immediately). `bio_percentile` becomes
    `variant-flip-both.json`'s `bioPercentile`, the same field explain.py
    itself reads (scripts/analysis/explain.py:754) -- defaulting below the
    threshold keeps the base fixture's step 2b a no-producer skip, matching
    every other already-computed step. Returns `(fake_repo_root, B)`."""
    fake_repo = tmp_path / "fake-repo"
    script_dir = fake_repo / "scripts" / "selections"
    script_dir.mkdir(parents=True)
    fake_script = script_dir / "run-chain.sh"
    shutil.copy(RUN_CHAIN_SCRIPT, fake_script)
    _make_executable(fake_script)

    analysis_dir = fake_repo / "scripts" / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "explain.py").write_text(f"DECODER_PERCENTILE_THRESHOLD = {DECODER_PERCENTILE_THRESHOLD}\n")

    b_dir = fake_repo / "training" / "runs" / "selections" / selection
    (b_dir / "graphs").mkdir(parents=True)
    (b_dir / "interventions").mkdir(parents=True)

    (b_dir / f"malecns-arena-{selection}.bin.gz").write_bytes(b"stub-graph-bytes")
    manifest_path = b_dir / f"malecns-arena-{selection}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "binarySha256": GRAPH_SHA,
                "nullExplanation": {"artifact": "null-explanation.json", "sha256": NULL_EXPLANATION_SHA256},
            }
        )
    )

    (b_dir / "graphs" / "index.json").write_text(json.dumps({"sourceSha256": GRAPH_SHA}))
    (b_dir / "rewiring-null.json").write_text(json.dumps({"sourceGraphSha256": GRAPH_SHA}))
    (b_dir / "rewiring-null-report.md").write_text("stub\n")
    (b_dir / "variant-flip-both.json").write_text(
        json.dumps({"sourceGraphSha256": GRAPH_SHA, "bioPercentile": bio_percentile})
    )
    (b_dir / "transfer.json").write_text("{}")
    (b_dir / "features.json").write_text("{}")
    (b_dir / "regime.json").write_text("{}")
    (b_dir / "null-explanation.json").write_bytes(NULL_EXPLANATION_BYTES)
    (b_dir / "null-explanation-report.md").write_text("stub\n")
    (b_dir / "coverage.json").write_text(json.dumps({"sourceGraphSha256": GRAPH_SHA}))
    (b_dir / "interventions" / "index.json").write_text(json.dumps({"sourceSha256": GRAPH_SHA}))
    (b_dir / "interventions" / "attribution.json").write_text("{}")
    (b_dir / "interventions" / "authored.json").write_text("{}")
    (b_dir / "intervention-stats.json").write_text("{}")

    return fake_repo, b_dir


def _run_real(fake_repo: Path, selection: str, stub_bin_dir: Path, *extra_args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(fake_repo / "scripts" / "selections" / "run-chain.sh"), selection, *extra_args],
        cwd=fake_repo,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_real_run_skips_every_step_and_invokes_no_producer_when_everything_already_matches(tmp_path) -> None:
    fake_repo, _ = _build_fake_repo(tmp_path)
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    assert "npm run" not in result.stdout
    assert "uv run" not in result.stdout
    assert "chain complete for selection=default" in result.stdout


def test_real_run_does_not_skip_step_2_when_the_report_md_output_is_missing(tmp_path) -> None:
    # The "both outputs must exist to skip" rule for step 2 -- only
    # `rewiring-null.json` (not `rewiring-null-report.md`) is present.
    fake_repo, b_dir = _build_fake_repo(tmp_path)
    (b_dir / "rewiring-null-report.md").unlink()
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    # The skip message is only ever logged from the `then` branch of the
    # skip check -- its absence here, together with both stub invocations
    # below, proves step 2 took the "run" branch instead.
    assert "and .../rewiring-null-report.md exist, skipping" not in result.stdout
    assert "npm run null:evaluate --" in result.stdout
    assert "npm run null:report --" in result.stdout


def test_real_run_does_not_skip_step_3d_when_the_manifest_null_explanation_entry_is_missing(tmp_path) -> None:
    # Item 4 (thermo-methodology I1): both `null-explanation.json` and
    # `null-explanation-report.md` are present, but the manifest carries no
    # `nullExplanation` entry at all -- the file-existence check alone
    # would wrongly skip; the new manifest-key check must not.
    fake_repo, b_dir = _build_fake_repo(tmp_path)
    manifest_path = b_dir / "malecns-arena-default.manifest.json"
    manifest_path.write_text(json.dumps({"binarySha256": GRAPH_SHA}))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    assert (
        "step 3d (explain --selection-mode): both outputs exist and manifest nullExplanation entry matches, "
        "skipping" not in result.stdout
    )
    assert "explain.py --selection-mode" in result.stdout


def test_real_run_does_not_skip_step_3d_when_the_manifest_sha_is_stale(tmp_path) -> None:
    # Same rule, the other failure mode: the manifest's `nullExplanation`
    # entry IS present, but its recorded sha256 no longer matches the
    # actual `null-explanation.json` bytes on disk (e.g. a stale entry left
    # over from a previous, now-superseded run of step 3d).
    fake_repo, b_dir = _build_fake_repo(tmp_path)
    manifest_path = b_dir / "malecns-arena-default.manifest.json"
    stale_manifest = json.loads(manifest_path.read_text())
    stale_manifest["nullExplanation"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(stale_manifest))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    assert (
        "step 3d (explain --selection-mode): both outputs exist and manifest nullExplanation entry matches, "
        "skipping" not in result.stdout
    )
    assert "explain.py --selection-mode" in result.stdout


def test_real_run_check_sha_mismatch_fails(tmp_path) -> None:
    fake_repo, b_dir = _build_fake_repo(tmp_path)
    (b_dir / "graphs" / "index.json").write_text(json.dumps({"sourceSha256": "b" * 64}))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode != 0
    assert "sha mismatch" in result.stderr
    assert "graphs/index.json sourceSha256" in result.stderr


def test_real_run_final_guard_trips_when_public_is_dirty(tmp_path) -> None:
    fake_repo, _ = _build_fake_repo(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=fake_repo, check=True)
    (fake_repo / "public").mkdir()
    (fake_repo / "public" / "stray.txt").write_text("should never have been written here\n")
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode != 0
    assert "public/ or docs/ changed during this chain" in result.stderr


# ---------------------------------------------------------------------------
# Step 2b: the predeclared conditional single-axis follow-up
# (.agents/plans/null-explanation/00-overview.md:35) -- only if the mirrored
# run's bioPercentile is >= explain.py's own DECODER_PERCENTILE_THRESHOLD
# (0.25) does the chain also run authored-flip-thrust/authored-flip-yaw and
# pass their outputs to explain.py's --variant-flip-thrust/--variant-flip-yaw.
# ---------------------------------------------------------------------------


def _explain_invocation_line(stdout: str) -> str:
    return next(line for line in stdout.splitlines() if "explain.py --selection-mode" in line)


def test_real_run_single_axis_steps_run_and_explain_flags_passed_when_threshold_met(tmp_path) -> None:
    fake_repo, b_dir = _build_fake_repo(tmp_path, bio_percentile=BIO_PERCENTILE_ABOVE_THRESHOLD)
    # Force step 3d to actually run (not skip) so its logged command can be
    # inspected for the conditional flags -- same technique as
    # test_real_run_does_not_skip_step_3d_when_the_manifest_null_explanation_entry_is_missing.
    manifest_path = b_dir / "malecns-arena-default.manifest.json"
    manifest_path.write_text(json.dumps({"binarySha256": GRAPH_SHA}))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "meets the predeclared >= " in out
    assert "--decoder authored-flip-thrust" in out
    assert "--decoder authored-flip-yaw" in out
    assert "variant-flip-thrust.json" in out
    assert "variant-flip-yaw.json" in out

    explain_line = _explain_invocation_line(out)
    assert "--variant-flip-thrust" in explain_line
    assert "--variant-flip-yaw" in explain_line
    assert "variant-flip-thrust.json" in explain_line
    assert "variant-flip-yaw.json" in explain_line
    assert (b_dir / "variant-flip-thrust.json").is_file()
    assert (b_dir / "variant-flip-yaw.json").is_file()


def test_real_run_single_axis_steps_skipped_and_no_explain_flags_when_threshold_not_met(tmp_path) -> None:
    fake_repo, b_dir = _build_fake_repo(tmp_path, bio_percentile=BIO_PERCENTILE_BELOW_THRESHOLD)
    manifest_path = b_dir / "malecns-arena-default.manifest.json"
    manifest_path.write_text(json.dumps({"binarySha256": GRAPH_SHA}))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "is below the predeclared >= " in out
    assert "skipping authored-flip-thrust/authored-flip-yaw" in out
    assert "--decoder authored-flip-thrust" not in out
    assert "--decoder authored-flip-yaw" not in out
    assert not (b_dir / "variant-flip-thrust.json").exists()
    assert not (b_dir / "variant-flip-yaw.json").exists()

    explain_line = _explain_invocation_line(out)
    assert "--variant-flip-thrust" not in explain_line
    assert "--variant-flip-yaw" not in explain_line


def test_real_run_step_2b_resume_skips_completed_single_axis_outputs(tmp_path) -> None:
    fake_repo, b_dir = _build_fake_repo(tmp_path, bio_percentile=BIO_PERCENTILE_ABOVE_THRESHOLD)
    (b_dir / "variant-flip-thrust.json").write_text(json.dumps({"sourceGraphSha256": GRAPH_SHA}))
    (b_dir / "variant-flip-yaw.json").write_text(json.dumps({"sourceGraphSha256": GRAPH_SHA}))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "variant-flip-thrust.json exists, skipping" in out
    assert "variant-flip-yaw.json exists, skipping" in out
    assert "--decoder authored-flip-thrust" not in out
    assert "--decoder authored-flip-yaw" not in out
    assert "chain complete for selection=default" in out


def test_real_run_single_axis_triggers_at_exact_25_percent_boundary(tmp_path) -> None:
    # The plan says "at least the 25th percentile" -- exactly meeting the
    # threshold must trigger the single-axis runs, not only exceeding it.
    fake_repo, b_dir = _build_fake_repo(tmp_path, bio_percentile=DECODER_PERCENTILE_THRESHOLD)
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "meets the predeclared >= " in out
    assert "--decoder authored-flip-thrust" in out
    assert "--decoder authored-flip-yaw" in out
    assert (b_dir / "variant-flip-thrust.json").is_file()
    assert (b_dir / "variant-flip-yaw.json").is_file()


def test_real_run_step_2b_check_sha_mismatch_fails(tmp_path) -> None:
    fake_repo, b_dir = _build_fake_repo(tmp_path, bio_percentile=BIO_PERCENTILE_ABOVE_THRESHOLD)
    (b_dir / "variant-flip-thrust.json").write_text(json.dumps({"sourceGraphSha256": "b" * 64}))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode != 0
    assert "sha mismatch" in result.stderr
    assert "variant-flip-thrust.json sourceGraphSha256" in result.stderr


def test_real_run_step_2b_fails_loudly_on_a_missing_bio_percentile_field(tmp_path) -> None:
    # A dual-review finding: jq -r on a missing key prints the string
    # "null", which --argjson happily parses as JSON null, and `null >=
    # 0.25` is just `false` -- silently "not triggered" instead of an
    # error. variant-flip-both.json without a bioPercentile field at all
    # must fail the chain, not quietly skip the single-axis steps.
    fake_repo, b_dir = _build_fake_repo(tmp_path)
    (b_dir / "variant-flip-both.json").write_text(json.dumps({"sourceGraphSha256": GRAPH_SHA}))
    stub_bin = tmp_path / "stub-bin"
    _write_stub_bin(stub_bin)

    result = _run_real(fake_repo, "default", stub_bin)
    assert result.returncode != 0
    assert "no numeric bioPercentile field" in result.stderr
    assert "--decoder authored-flip-thrust" not in result.stdout
    assert "--decoder authored-flip-yaw" not in result.stdout
