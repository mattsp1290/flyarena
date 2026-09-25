"""Provenance verification for `scripts/analysis/explain.py` (WP3): graph
identity (`sourceGraphSha256`/`rewireSourceSha256`) and producer code
identity (`producer.sourceSha256`) for every WP1/WP2 input this study
combines.

Extracted from `explain.py` alongside `explain_stats.py`/`explain_report.py`
(same "keep each file under this repo's 1000-line rule" rationale -- see
`explain_stats.py`'s doc comment) once this study's code-identity provenance
requirement (pinning not just *which graphs* an input was computed from, but
*which version of the producing script*) grew this section well past what
comfortably fits alongside `explain.py`'s CLI/orchestration role. Like
`explain_stats.py`/`explain_report.py`, this module does not import
`explain.py` (one-directional import boundary); `REWIRED_COUNT` is passed in
by the caller rather than imported, the same pattern `explain_stats.
build_family_rank_matrix` already uses.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import graph_io

#: The producer scripts (plus shared helpers) this study's code-identity
#: check pins -- hashed via `graph_io.source_identity_sha256` (the same
#: scheme as `scripts/data/compile.py`'s `compiler_source_sha256()`).
#: `graph_io.py`/`env_guard.py` are listed for both Python producers since
#: both import them; a change to either shared file correctly changes both
#: producer shas.
TRANSFER_SOURCE_FILENAMES: tuple[str, ...] = ("transfer.py", "graph_io.py", "env_guard.py")
FEATURES_SOURCE_FILENAMES: tuple[str, ...] = ("features.py", "graph_io.py", "env_guard.py")
#: `scripts/null/regime-check.ts`'s own producer files, kept in lockstep with
#: that file's `REGIME_SOURCE_FILENAMES` constant. Hashed from raw file bytes
#: only (no TypeScript execution needed), the same cross-language
#: recomputation `tests/unit/malecns-artifact.test.ts` does for `compile.py`.
REGIME_SOURCE_FILENAMES: tuple[str, ...] = ("regime-check.ts", "regime-task.ts", "regime-worker.ts", "null-worker-shared.ts")


def current_transfer_source_sha256(analysis_source_dir) -> str:
    return graph_io.source_identity_sha256(analysis_source_dir, TRANSFER_SOURCE_FILENAMES)


def current_features_source_sha256(analysis_source_dir) -> str:
    return graph_io.source_identity_sha256(analysis_source_dir, FEATURES_SOURCE_FILENAMES)


def current_regime_source_sha256(null_source_dir) -> str:
    return graph_io.source_identity_sha256(null_source_dir, REGIME_SOURCE_FILENAMES)


def _require_matching_source(label: str, value: str, expected: str) -> None:
    if value != expected:
        raise ValueError(f"explain: {label} sourceGraphSha256/rewireSourceSha256 does not match rewiring-null-v1.json")


def _require_matching_producer(label: str, payload: Mapping[str, object], expected_script: str, current_sha: str) -> None:
    """Refuse (clear error) any input whose recorded `producer.sourceSha256`
    does not match the current working tree's code -- the code-identity half
    of this study's provenance guarantee (`_require_matching_source` above
    is the graph-identity half; a `features.json`/`transfer.json`/
    `regime.json` regenerated from stale code against the *same* graphs
    would otherwise pass every graph-identity check silently, exactly the
    failure mode this study's own feature-6 stale-file incident hit).
    `features_exploratory_json` is deliberately never passed through this
    check -- see `verify_provenance`'s doc comment."""
    producer = payload.get("producer")
    if not producer:
        raise ValueError(
            f"explain: {label} has no 'producer' block -- regenerate it with the current {expected_script} "
            "(code-identity provenance is required for this input)"
        )
    if producer.get("script") != expected_script:
        raise ValueError(
            f"explain: {label}'s producer.script is {producer.get('script')!r}, expected {expected_script!r}"
        )
    if producer.get("sourceSha256") != current_sha:
        raise ValueError(
            f"explain: {label} was produced by {expected_script} with producer.sourceSha256="
            f"{producer.get('sourceSha256')!r}, but the current {expected_script} (plus its shared helpers) hashes "
            f"to {current_sha!r} -- regenerate {label} from the current code before combining (code-identity "
            "provenance mismatch)"
        )


def require_complete_seed_coverage(label: str, seeds: Sequence[int], rewired_count: int) -> None:
    """A rewired-graph seed missing from `regime.json` or
    `rewiring-null-v1.json` would otherwise be silently dropped: a seed
    absent from `regime.json["rewired"]` is never checked against the
    per-graph regime thresholds (so it can never be excluded, correctly or
    not), and a seed absent from `rewiring-null-v1.json["rewired"]` is
    simply missing from `score_by_seed`, which raises a `KeyError` deep in
    `explain.build_metric` with no context about which input was short. A
    duplicate seed would silently overwrite a dict entry the same way.
    Checked once, loudly, at load time (an edge-case-review finding:
    previously unchecked)."""
    if sorted(seeds) != list(range(rewired_count)):
        raise ValueError(f"explain: {label} does not cover rewired seeds 0..{rewired_count - 1} exactly once")


def verify_provenance(
    rewiring_null: dict,
    variants: Mapping[str, dict],
    transfer_json: dict,
    features_json: dict,
    features_exploratory_json: dict,
    regime_json: dict,
    analysis_source_dir,
    null_source_dir,
) -> None:
    """`variants` is every loaded decoder-variant payload (flip-both, plus
    any provided single-axis ones), checked in one loop (an edge-case-review
    finding: the single-axis path previously had no provenance guarantee).

    Pins two independent things (see `docs/null-explanation-report.md`'s
    Limitations section for the reader-facing version): graph identity
    (`sourceGraphSha256`/`rewireSourceSha256`, every input below) and, for
    `transfer.json`/`features.json`/`regime.json` only, producer *code*
    identity (`_require_matching_producer`, recomputed from the current
    working tree). `features_exploratory_json` is deliberately exempt from
    the code-identity check: it is a pinned, stale-code exploratory input by
    design (feature 6's pre-adjudication run -- rerunning it against current
    code would defeat its purpose as a historical snapshot), so only its
    graph identity and content sha (set in `explain.main()`) are pinned."""
    expected_source = rewiring_null["sourceGraphSha256"]
    expected_rewire = rewiring_null["rewireSourceSha256"]
    payloads: list[tuple[str, dict]] = [(f"variant-{key}", payload) for key, payload in variants.items()]
    payloads += [
        ("transfer.json", transfer_json),
        ("features.json", features_json),
        ("features-exploratory-unrestricted.json", features_exploratory_json),
    ]
    for label, payload in payloads:
        _require_matching_source(f"{label}.sourceGraphSha256", payload["sourceGraphSha256"], expected_source)
        _require_matching_source(f"{label}.rewireSourceSha256", payload["rewireSourceSha256"], expected_rewire)
    _require_matching_source("regime.json.sourceGraphSha256", regime_json["sourceGraphSha256"], expected_source)
    _require_matching_source("regime.json.rewireSourceSha256", regime_json["rewireSourceSha256"], expected_rewire)

    _require_matching_producer(
        "transfer.json", transfer_json, "scripts/analysis/transfer.py", current_transfer_source_sha256(analysis_source_dir)
    )
    _require_matching_producer(
        "features.json", features_json, "scripts/analysis/features.py", current_features_source_sha256(analysis_source_dir)
    )
    _require_matching_producer(
        "regime.json", regime_json, "scripts/null/regime-check.ts", current_regime_source_sha256(null_source_dir)
    )
