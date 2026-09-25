#!/usr/bin/env python3
"""Combine the null-explanation study's inputs into one pinned artifact and
report (`.agents/plans/null-explanation/03-explanation-report.md`, WP3).

Inputs (all already computed by WP1/WP2, none re-simulated here):

- `public/data/rewiring-null-v1.json` -- the authored-decoder null (500
  rewirings' scores, biological's score, rank statistics);
- the mirrored decoder-variant summary (`authored-flip-both`), plus the two
  single-axis summaries *only if* the predeclared condition
  (`.agents/plans/null-explanation/00-overview.md`: mirrored biological
  percentile >= 25%) triggered them -- this study's mirrored run left
  biological at the 0th percentile, so the single-axis runs were correctly
  skipped (bean `flyarena-j0tz`), and `--variant-flip-thrust`/
  `--variant-flip-yaw` are therefore omitted below;
- `scripts/analysis/transfer.py`'s `transfer.json` (T, spectral abscissa,
  condition number, `turnGain`/`approachGain` per graph);
- `scripts/analysis/features.py`'s `features.json` (40 predeclared
  structural features per graph), plus a pre-adjudication
  `features-exploratory-unrestricted.json` run with feature 6
  (`weightedInDegree`) unrestricted -- disclosed in the report as
  exploratory/non-predeclared, never used in the outcome-category
  evaluation (see `explain_report.render_feature6_disclosure`'s doc
  comment);
- `scripts/null/regime-check.ts`'s `regime.json` (per-graph clamp-fraction
  and steady-state-distance samples on 10 held-out seeds).

For each of the ~66 predeclared scalar metrics (24 transfer entries, 2
derived predictors, 40 structural features -- `00-overview.md`'s own tally)
this module computes: biological's value; the null distribution's median,
2.5%, and 97.5% values (same empirical-quantile convention as
`scripts/null/null-stats.ts`'s `nullSummary`); biological's empirical
percentile in the null (same tie rule as that file's `rankStatistics`); and
the Spearman rank correlation between the metric and the authored score
across the 500 rewirings, with a 95% bootstrap CI (10,000 resamples, a fixed
per-metric seed). It then applies the regime gate, the per-graph
ill-conditioned/regime-invalid exclusion (for transfer-kind metrics only --
structural features do not depend on the linear regime), and a 1,000-seeded-
permutation multiple-comparisons calibration, and mechanically evaluates the
overview's predeclared outcome categories.

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DD_IAST_ENABLED=false` and `PYTHONPATH` unset, matching every other
`scripts/analysis/` CLI (`env_guard.assert_single_threaded_blas`, checked
before `numpy` does any work, below).

This module is the CLI/orchestration entry point only: pure statistics
(`quantile_index`, `spearman_rho`, the permutation calibration, ...) live in
`explain_stats.py`, and Markdown report rendering lives in
`explain_report.py` -- both extracted out of what was previously a single
1498-line file, 50% past this repo's own "do not let a file cross 1000
lines without a very strong reason" rule (a thermo-maintainability review
finding; see `explain_stats.py`'s doc comment for the precedent, `scripts/
null/null-report-trained.ts`, this study's Python sibling did not originally
get the same treatment).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402

import graph_io  # noqa: E402
from graph_io import canonical_json_text, sha256_hex, write_canonical_json  # noqa: E402
from features import OBSERVATION_CHANNELS, OUTPUT_POPULATIONS  # noqa: E402
from transfer import ILL_CONDITIONED_THRESHOLD, OBSERVATION_CHANNEL_INDEX, OUTPUT_POPULATION_INDEX  # noqa: E402

from explain_stats import (  # noqa: E402
    _rank,
    bootstrap_spearman_ci,
    build_family_rank_matrix,
    format_pct,
    joint_permutation_chance_rate,
    metric_rng,
    null_range_summary,
    outside_range,
    rank_statistics,
    spearman_rho,
)
from explain_report import render_report_markdown  # noqa: E402
import explain_provenance  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
DOCS_DIR = REPO_ROOT / "docs"

VERSION = 1
REWIRED_COUNT = 500

# ---------------------------------------------------------------------------
# Predeclared thresholds (`.agents/plans/null-explanation/00-overview.md`'s
# "Predeclared outcome categories" and "Key decisions" -- do not change
# without editing that file first, per its "Unresolved decisions" section).
# ---------------------------------------------------------------------------

#: Decoder-convention-driven: the mirrored (or triggered single-axis)
#: variant must move biological to at least this percentile.
DECODER_PERCENTILE_THRESHOLD = 0.25

#: Linear-pathway-driven / structural-feature-associated: the minimum
#: |Spearman rho| a metric must have with score, in addition to biological
#: sitting outside the null's 2.5-97.5% range.
SPEARMAN_RHO_THRESHOLD = 0.3

#: Regime gate: the per-graph (and null-median) rate-clamp fraction must be
#: at or below this to trust that graph's transfer entries.
CLAMP_FRACTION_THRESHOLD = 0.2

#: Regime gate: the per-graph (and null-median) linear steady-state distance
#: must be at or below this.
STEADY_STATE_DISTANCE_THRESHOLD = 0.5

#: Single source of truth with `transfer.py`'s own `illConditioned` flag --
#: a graph whose `(lambda I - g A)` condition number exceeds this is
#: excluded from the transfer-kind correlations (and, for biological, fails
#: the aggregate regime gate outright).
CONDITION_NUMBER_THRESHOLD = ILL_CONDITIONED_THRESHOLD

BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_COUNT = 1_000

#: Fixed base seeds for this module's own (per-metric) bootstrap streams and
#: its permutation calibration -- arbitrary but constant, so a rerun on
#: unchanged inputs reproduces byte-identical output (WP3 acceptance:
#: "Running `explain.py` twice on the Spark gives byte-identical output").
#: Deliberately independent of `rewiring-null-v1.json`'s own
#: `bootstrap.seed` (1314212940): that seed calibrates a different
#: statistic (the score's own bootstrap CI, computed by `null-stats.ts`),
#: and reusing it here would wrongly suggest the two are the same
#: computation.
BOOTSTRAP_BASE_SEED = 0x4E554C4C_45585031  # "NULLEXP1", read as hex digits
PERMUTATION_BASE_SEED = 0x4E554C4C_45585032  # "NULLEXP2"


# ---------------------------------------------------------------------------
# Metric name tables (single source of truth: the exact order + names every
# metric is reported under, in both the JSON and the report). Reuses
# `features.py`'s `OBSERVATION_CHANNELS`/`OUTPUT_POPULATIONS` and
# `transfer.py`'s `OBSERVATION_CHANNEL_INDEX`/`OUTPUT_POPULATION_INDEX` as
# the index maps, rather than re-declaring the channel/population lists a
# third time.
# ---------------------------------------------------------------------------

TRANSFER_METRIC_NAMES: list[str] = [
    f"T:{channel}->{population}" for channel in OBSERVATION_CHANNELS for population in OUTPUT_POPULATIONS
]
DERIVED_METRIC_NAMES: list[str] = ["turnGain", "approachGain"]
FEATURE_METRIC_NAMES: list[str] = (
    [f"pathLength:{channel}->{population}" for channel in OBSERVATION_CHANNELS for population in OUTPUT_POPULATIONS]
    + ["meanPathLength"]
    + [f"excitatoryPathCount:{population}" for population in OUTPUT_POPULATIONS]
    + [f"inhibitoryPathCount:{population}" for population in OUTPUT_POPULATIONS]
    + ["reciprocity"]
    + [f"weightBalance:{population}" for population in OUTPUT_POPULATIONS]
    + ["twoCycleCount", "feedForwardTriangleCount"]
    + [f"weightedInDegree:{population}" for population in OUTPUT_POPULATIONS]
)

assert len(TRANSFER_METRIC_NAMES) == 24, len(TRANSFER_METRIC_NAMES)
assert len(DERIVED_METRIC_NAMES) == 2, len(DERIVED_METRIC_NAMES)
assert len(FEATURE_METRIC_NAMES) == 40, len(FEATURE_METRIC_NAMES)
TOTAL_METRIC_COUNT = len(TRANSFER_METRIC_NAMES) + len(DERIVED_METRIC_NAMES) + len(FEATURE_METRIC_NAMES)
assert TOTAL_METRIC_COUNT == 66, TOTAL_METRIC_COUNT


def extract_transfer_value(graph_result: dict, name: str) -> float | None:
    if graph_result.get("singular") or graph_result.get("T") is None:
        return None
    channel, population = name[len("T:") :].split("->")
    return graph_result["T"][OUTPUT_POPULATION_INDEX[population]][OBSERVATION_CHANNEL_INDEX[channel]]


def extract_derived_value(graph_result: dict, name: str) -> float | None:
    return graph_result.get(name)


def extract_feature_value(graph_result: dict, name: str) -> float | None:
    if name == "meanPathLength":
        return graph_result["meanPathLength"]
    if name.startswith("pathLength:"):
        return graph_result["pathLengths"][name[len("pathLength:") :]]
    if name.startswith("excitatoryPathCount:"):
        return graph_result["excitatoryPathCount"][name.split(":", 1)[1]]
    if name.startswith("inhibitoryPathCount:"):
        return graph_result["inhibitoryPathCount"][name.split(":", 1)[1]]
    if name == "reciprocity":
        return graph_result["reciprocity"]
    if name.startswith("weightBalance:"):
        return graph_result["weightBalance"][name.split(":", 1)[1]]
    if name == "twoCycleCount":
        return graph_result["twoCycleCount"]
    if name == "feedForwardTriangleCount":
        return graph_result["feedForwardTriangleCount"]
    if name.startswith("weightedInDegree:"):
        return graph_result["weightedInDegree"][name.split(":", 1)[1]]
    raise ValueError(f"explain: unknown feature metric name {name!r}")


# ---------------------------------------------------------------------------
# Regime gate + per-graph exclusion
# ---------------------------------------------------------------------------


def per_graph_regime_summary(regime_entry: dict) -> dict:
    """Mean over the 10 held-out episodes -- both `clampFraction` and
    `steadyStateDistance` are already per-episode averages/fractions
    (`scripts/null/regime-task.ts`'s doc comment), so a graph's own summary
    is their mean across its 10 held-out seeds. A predeclared, disclosed
    choice among reasonable aggregations (median/max would also be
    defensible); with every measured value well inside both thresholds (see
    `docs/null-explanation-report.md`'s regime section), the choice does not
    change this study's gate outcome."""
    return {
        "clampFraction": float(np.mean(regime_entry["clampFraction"])),
        "steadyStateDistance": float(np.mean(regime_entry["steadyStateDistance"])),
    }


def _transfer_entry_invalid(entry: Mapping[str, object]) -> bool:
    """A transfer entry is untrustworthy as a steady-state gain if the solve
    was singular, ill-conditioned, or the continuous-time system is not
    stable (`spectralAbscissa >= leakRate`) or the discretized per-substep
    Euler update is not stable (`discretizedSpectralRadius >= 1`) --
    `transfer.py`'s own module docstring hands this off explicitly: "when
    WP3 writes the report's stability prose, it must present both numbers
    side by side ... not report `stable` alone". `T` can be algebraically
    well-defined and well-conditioned for an unstable graph while being
    meaningless as a steady-state gain, so this is checked independently of
    `illConditioned`/`singular` (a rigor-review finding: unchecked before,
    though every graph in this study's actual data is both stable and
    discretized-stable, so this made no numeric difference here)."""
    return bool(
        entry.get("singular")
        or entry.get("illConditioned")
        or not entry.get("stable", False)
        or not entry.get("discretizedStable", False)
    )


def compute_regime(regime_json: dict, transfer_json: dict) -> dict:
    bio_regime = per_graph_regime_summary(regime_json["biological"])
    rewired_regime = {
        f"rewired-{entry['seed']}": per_graph_regime_summary(entry) for entry in regime_json["rewired"]
    }
    null_median_distance = float(np.median([v["steadyStateDistance"] for v in rewired_regime.values()]))
    null_median_clamp = float(np.median([v["clampFraction"] for v in rewired_regime.values()]))

    bio_transfer = transfer_json["graphs"]["biological"]
    bio_transfer_ok = not _transfer_entry_invalid(bio_transfer)

    # The plan's regime-gate sentence ("the median steady-state distance ...
    # is <= 0.5 for biological and for the null median ... also requires the
    # rate-clamp fraction to be <= 20%") reads "for biological and for the
    # null median" as applying to both the distance and the clamp fraction,
    # not to distance alone -- gate on both null-median statistics, not only
    # the distance one (an edge-case-review finding: this module's own doc
    # comment above already claimed the null-median clamp was gated; it
    # previously was not).
    gate_passed = (
        bio_regime["steadyStateDistance"] <= STEADY_STATE_DISTANCE_THRESHOLD
        and null_median_distance <= STEADY_STATE_DISTANCE_THRESHOLD
        and bio_regime["clampFraction"] <= CLAMP_FRACTION_THRESHOLD
        and null_median_clamp <= CLAMP_FRACTION_THRESHOLD
        and bio_transfer_ok
    )

    excluded_graph_ids: list[str] = []
    for graph_id, summary in rewired_regime.items():
        transfer_entry = transfer_json["graphs"][graph_id]
        fails = (
            summary["clampFraction"] > CLAMP_FRACTION_THRESHOLD
            or summary["steadyStateDistance"] > STEADY_STATE_DISTANCE_THRESHOLD
            or _transfer_entry_invalid(transfer_entry)
        )
        if fails:
            excluded_graph_ids.append(graph_id)
    excluded_graph_ids.sort(key=lambda gid: int(gid.split("-")[1]))

    # Publish the stability numbers themselves, not only the pass/fail gate
    # derived from them -- `transfer.py`'s module docstring hands this off
    # explicitly ("when WP3 writes the report's stability prose, it must
    # present both numbers side by side ... not report `stable` alone under
    # a bare 'stable' label"), and a round-2 rigor review found the gate fix
    # alone left this unmet: biological's `discretizedSpectralRadius` is
    # 0.9932, close to the 1.0 instability boundary despite being on the
    # stable side, which a reader weighing the linear-pathway finding should
    # be able to see rather than only "not unstable".
    rewired_transfer_entries = [transfer_json["graphs"][gid] for gid in rewired_regime]
    stability = {
        "bio": {
            "stable": bool(bio_transfer["stable"]),
            "spectralAbscissa": bio_transfer["spectralAbscissa"],
            "leakRate": bio_transfer["leakRate"],
            "discretizedStable": bool(bio_transfer["discretizedStable"]),
            "discretizedSpectralRadius": bio_transfer["discretizedSpectralRadius"],
        },
        "nullMedian": {
            "spectralAbscissa": float(np.median([e["spectralAbscissa"] for e in rewired_transfer_entries])),
            "discretizedSpectralRadius": float(
                np.median([e["discretizedSpectralRadius"] for e in rewired_transfer_entries])
            ),
        },
        "unstableNullCount": sum(
            1 for e in rewired_transfer_entries if not e["stable"] or not e["discretizedStable"]
        ),
    }

    return {
        "bio": bio_regime,
        "nullSampleMedian": {"clampFraction": null_median_clamp, "steadyStateDistance": null_median_distance},
        "gatePassed": bool(gate_passed),
        "excludedGraphIds": excluded_graph_ids,
        "excludedCount": len(excluded_graph_ids),
        "stability": stability,
    }


# ---------------------------------------------------------------------------
# Per-metric statistics
# ---------------------------------------------------------------------------


def build_metric(
    name: str,
    kind: str,
    extractor,
    graphs: Mapping[str, dict],
    score_by_seed: Mapping[int, float],
    excluded_graph_ids: frozenset[str],
    bootstrap_base_seed: int,
) -> dict | None:
    bio_value = extractor(graphs["biological"], name)
    pairs: list[tuple[float, float]] = []
    for seed in range(REWIRED_COUNT):
        graph_id = f"rewired-{seed}"
        if graph_id in excluded_graph_ids:
            continue
        value = extractor(graphs[graph_id], name)
        if value is None:
            continue
        pairs.append((value, score_by_seed[seed]))

    if bio_value is None or len(pairs) < 2:
        return None

    null_values = [v for v, _ in pairs]
    metric_values = np.array([v for v, _ in pairs], dtype=np.float64)
    scores = np.array([s for _, s in pairs], dtype=np.float64)

    range_summary = null_range_summary(null_values)
    rank_stats = rank_statistics(null_values, bio_value)

    # A metric constant across the (non-excluded) null set has an undefined
    # Spearman rho, not a zero one -- `explain_stats._pearson` returns `0.0`
    # as a safe internal sentinel (it can never spuriously clear
    # `SPEARMAN_RHO_THRESHOLD`), but reporting "0.000 [0.000, 0.000]" to a
    # reader would read as a precisely measured null correlation rather than
    # "not computable" (a rigor-review finding). `nullConstant` lets
    # `explain_report.render_metric_stats_table` show "n/a" instead.
    null_constant = bool(np.ptp(metric_values) == 0.0)

    rx = _rank(metric_values)
    ry = _rank(scores)
    rho = spearman_rho(metric_values, scores)
    rng = metric_rng(bootstrap_base_seed, name)
    ci_lo, ci_hi = bootstrap_spearman_ci(rx, ry, BOOTSTRAP_RESAMPLES, rng)

    metric = {
        "name": name,
        "kind": kind,
        "bio": float(bio_value),
        "nullMedian": range_summary["median"],
        "p2_5": range_summary["p2_5"],
        "p97_5": range_summary["p97_5"],
        "bioPercentile": rank_stats["bioPercentile"],
        "spearman": rho,
        "spearmanCi": [ci_lo, ci_hi],
        "nullConstant": null_constant,
    }
    # Stamped from the single module-scope `qualifies` predicate below (not
    # a second, hand-written boolean expression) -- lets `explain_report.
    # render_metric_stats_table` visually flag a qualifying row without
    # itself depending on `explain.py` (this module's own one-directional
    # import boundary; see `explain_report.py`'s doc comment) or
    # re-implementing the gate a third time (a thermo-maintainability review
    # finding, M3, generalized: every qualification decision in this
    # pipeline routes through one function).
    metric["qualifiesBothGates"] = qualifies(metric)
    return metric


def build_all_metrics(
    transfer_graphs: Mapping[str, dict],
    features_graphs: Mapping[str, dict],
    score_by_seed: Mapping[int, float],
    excluded_graph_ids: frozenset[str],
) -> list[dict]:
    metrics: list[dict] = []
    for name in TRANSFER_METRIC_NAMES:
        metric = build_metric(
            name, "transfer", extract_transfer_value, transfer_graphs, score_by_seed, excluded_graph_ids, BOOTSTRAP_BASE_SEED
        )
        if metric is not None:
            metrics.append(metric)
    for name in DERIVED_METRIC_NAMES:
        metric = build_metric(
            name, "derived", extract_derived_value, transfer_graphs, score_by_seed, excluded_graph_ids, BOOTSTRAP_BASE_SEED
        )
        if metric is not None:
            metrics.append(metric)
    for name in FEATURE_METRIC_NAMES:
        # Structural features never depend on the linear regime -- no
        # per-graph exclusion applied (`compute_regime`'s doc comment).
        metric = build_metric(
            name, "feature", extract_feature_value, features_graphs, score_by_seed, frozenset(), BOOTSTRAP_BASE_SEED
        )
        if metric is not None:
            metrics.append(metric)
    return metrics


# ---------------------------------------------------------------------------
# Outcome categories (`00-overview.md`'s "Predeclared outcome categories")
# ---------------------------------------------------------------------------


def qualifies(metric: Mapping[str, object]) -> bool:
    """The predeclared linear-pathway/structural-feature gate
    (`.agents/plans/null-explanation/00-overview.md`): biological sits
    outside the null's 2.5-97.5% range AND `|rho| >= SPEARMAN_RHO_THRESHOLD`.
    Module-scope, not a closure local to `evaluate_categories` (M3):
    `evaluate_categories`, `main()`'s `definitionSensitive` computation, and
    `build_metric`'s `qualifiesBothGates` stamp all call this one function,
    so there is exactly one implementation of "qualifies" in this pipeline."""
    return outside_range(metric["bio"], metric["p2_5"], metric["p97_5"]) and abs(metric["spearman"]) >= SPEARMAN_RHO_THRESHOLD


def _direction_consistent(metric: Mapping[str, object]) -> bool:
    """Whether a qualifying metric's sign is consistent with explaining
    biological's *low* score: a positive rho with biological in the null's
    low tail (a metric that rises with score, and biological sits below the
    range), or a negative rho with biological in the high tail (a metric
    that falls with score, and biological sits above the range). `qualifies`
    only checks that biological is outside the range and `|rho| >=`
    threshold -- it does not check this, so a metric could in principle
    qualify while actually predicting a *higher* score for biological (a
    rigor-review finding). Neither of this study's two published triggers is
    affected (both are direction-consistent), but the check and its
    disclosure are added so `build_summary_sentence` never asserts a
    direction the data does not support."""
    bio_low = metric["bio"] < metric["p2_5"]
    return (metric["spearman"] > 0) == bio_low


def evaluate_categories(
    decoder_bio_percentile: float,
    transfer_and_derived_metrics: Sequence[dict],
    structural_metrics: Sequence[dict],
    regime: Mapping[str, object],
) -> dict:
    """Pure function: every input is already-computed data, no I/O, so this
    is exercised directly by `tests_python/test_explain.py` on synthetic
    inputs (each category triggered alone, several together, none; a regime
    gate failure yields `regime-invalid` and never `linear-pathway`)."""
    categories: list[str] = []
    triggers: list[tuple[str, float, dict | None]] = []

    decoder_triggered = decoder_bio_percentile >= DECODER_PERCENTILE_THRESHOLD
    if decoder_triggered:
        categories.append("decoderConvention")
        triggers.append(("decoderConvention", decoder_bio_percentile, None))

    linear_candidates = [m for m in transfer_and_derived_metrics if qualifies(m)]
    regime_gate_passed = bool(regime["gatePassed"])
    linear_triggered = bool(linear_candidates) and regime_gate_passed
    # The plan's rule is unconditional: "Otherwise [if the regime gate
    # fails] the transfer analysis is reported as regime-invalid
    # (inconclusive)" -- whenever the gate fails, the whole transfer
    # analysis is inconclusive, independent of whether any individual
    # metric happens to qualify. Gating this on `bool(linear_candidates)`
    # (as an earlier version did) meant a gate failure with zero qualifying
    # candidates -- including the degenerate case where biological's own
    # transfer solve is singular, so every transfer/derived metric is
    # dropped entirely -- was silently reported as a clean "no linear
    # finding" rather than "could not be evaluated" (an edge-case-review
    # finding).
    regime_invalid = not regime_gate_passed
    linear_detail: dict | None = None
    if linear_triggered:
        linear_detail = max(linear_candidates, key=lambda m: abs(m["spearman"]))
        linear_detail = {**linear_detail, "directionConsistent": _direction_consistent(linear_detail)}
        categories.append("linearPathway")
        triggers.append(("linearPathway", abs(linear_detail["spearman"]), linear_detail))

    structural_candidates = [m for m in structural_metrics if qualifies(m)]
    structural_detail: dict | None = None
    if structural_candidates:
        structural_detail = max(structural_candidates, key=lambda m: abs(m["spearman"]))
        structural_detail = {**structural_detail, "directionConsistent": _direction_consistent(structural_detail)}
        categories.append("structuralFeature")
        triggers.append(("structuralFeature", abs(structural_detail["spearman"]), structural_detail))

    ranked = [t[0] for t in sorted(triggers, key=lambda t: t[1], reverse=True)]
    unexplained = not categories

    # Every metric that independently passes both predeclared gates, across
    # both families, not only the single representative each triggered
    # category surfaces via `max(..., key=abs(spearman))` above -- the
    # Finding narrative names this full set, not just the per-category
    # exemplar. Sorted by |rho| descending, matching `ranked`'s convention.
    qualifying_metrics = sorted(
        linear_candidates + structural_candidates, key=lambda m: abs(m["spearman"]), reverse=True
    )

    return {
        "categories": categories,
        "ranked": ranked,
        "unexplained": unexplained,
        "regimeInvalid": regime_invalid,
        "decoderBioPercentile": decoder_bio_percentile,
        "linearDetail": linear_detail,
        "structuralDetail": structural_detail,
        "qualifyingMetrics": qualifying_metrics,
    }


def build_summary_sentence(finding: Mapping[str, object]) -> str:
    categories = finding["categories"]
    if not categories:
        if finding["regimeInvalid"]:
            return (
                "No single predeclared factor explains it under this model: the decoder-convention check did not "
                "move biological into the qualifying range, and the linear-pathway analysis could not be evaluated "
                "as a positive finding because the regime gate failed, so it is reported as regime-invalid "
                "(inconclusive)."
            )
        return "No single predeclared factor explains it under this model."

    def describe(kind_label: str, detail: Mapping[str, object]) -> str:
        base = (
            f"the {kind_label} {detail['name']} sits outside the null's 2.5-97.5% range "
            f"(rank correlation with score rho={detail['spearman']:.3f})"
        )
        if not detail.get("directionConsistent", True):
            base += (
                " -- but its sign predicts a HIGHER score for biological, not the observed low one "
                "(direction-inconsistent; reported for completeness, not as an explanation of the low score)"
            )
        return base

    parts: list[str] = []
    if "decoderConvention" in categories:
        parts.append(
            "flipping the authored decoder's thrust and yaw sign conventions moves biological to the "
            f"{format_pct(finding['decoderBioPercentile'])}"
        )
    if "linearPathway" in categories:
        parts.append(describe("linear transfer entry", finding["linearDetail"]))
    if "structuralFeature" in categories:
        parts.append(describe("structural feature", finding["structuralDetail"]))
    return "Biological's low score is associated with: " + "; ".join(parts) + " -- a descriptive correlation, not a causal claim."


def build_qualifying_metrics_note(qualifying_metrics: Sequence[Mapping[str, object]]) -> str:
    """Full disclosure sentence for the Finding section: names *every*
    metric passing both predeclared gates, not only the single
    representative each outcome category surfaces via its own
    `max(..., key=abs(spearman))` exemplar. Generated mechanically from
    `qualifying_metrics` (computed from `explanation["metrics"]` in
    `evaluate_categories`) -- never a hand-picked subset."""
    if not qualifying_metrics:
        return (
            "No metric passes both predeclared gates (outside the null's 2.5-97.5% range and |rho| at or above "
            "the threshold)."
        )
    parts = [f"{m['name']} (rho={m['spearman']:.3f})" for m in qualifying_metrics]
    plural = "metric" if len(qualifying_metrics) == 1 else "metrics"
    return (
        f"{len(qualifying_metrics)} {plural} independently pass both predeclared gates (outside the null's "
        "2.5-97.5% range and |rho| at or above the threshold): " + "; ".join(parts) + "."
    )


# ---------------------------------------------------------------------------
# Provenance -- graph identity + producer code identity, both verified by
# `explain_provenance.verify_provenance` (extracted to its own module for
# the same "keep each file under this repo's 1000-line rule" reason as
# `explain_stats.py`/`explain_report.py`; see that module's doc comment).
# The two thin wrappers below bind `REWIRED_COUNT`/`REPO_ROOT` (this
# module's own globals) into `explain_provenance`'s otherwise-parameterized
# functions, so callers below read like the single-module version did.
# `REPO_ROOT` (not the narrower `ANALYSIS_SOURCE_DIR`/`NULL_SOURCE_DIR` this
# replaced) is what each producer's real import-graph closure is walked
# and hashed relative to, since that closure now spans `scripts/analysis/`,
# `scripts/data/`, `scripts/training/`, and `src/lib/...`.
# ---------------------------------------------------------------------------


def current_transfer_source_sha256() -> str:
    return explain_provenance.current_transfer_source_sha256(REPO_ROOT)


def current_features_source_sha256() -> str:
    return explain_provenance.current_features_source_sha256(REPO_ROOT)


def current_regime_source_sha256() -> str:
    return explain_provenance.current_regime_source_sha256(REPO_ROOT)


def _require_complete_seed_coverage(label: str, seeds: Sequence[int]) -> None:
    explain_provenance.require_complete_seed_coverage(label, seeds, REWIRED_COUNT)


def verify_provenance(
    rewiring_null: dict,
    variants: Mapping[str, dict],
    transfer_json: dict,
    features_json: dict,
    features_exploratory_json: dict,
    regime_json: dict,
) -> None:
    explain_provenance.verify_provenance(
        rewiring_null,
        variants,
        transfer_json,
        features_json,
        features_exploratory_json,
        regime_json,
        REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewiring-null", type=Path, default=PUBLIC_DATA_DIR / "rewiring-null-v1.json")
    parser.add_argument("--variant-flip-both", type=Path, required=True)
    parser.add_argument("--variant-flip-thrust", type=Path, default=None)
    parser.add_argument("--variant-flip-yaw", type=Path, default=None)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--features-exploratory-unrestricted",
        type=Path,
        required=True,
        help=(
            "pre-adjudication features.json computed with feature 6 (weightedInDegree) unrestricted "
            "(any presynaptic neuron, not only input-labeled ones) -- disclosed in the report as "
            "exploratory/non-predeclared, never used in the outcome-category evaluation "
            "(.agents/plans/null-explanation/02-transfer-and-features.md's feature-6 adjudication note)"
        ),
    )
    parser.add_argument("--regime", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PUBLIC_DATA_DIR / "null-explanation-v1.json")
    parser.add_argument("--report-out", type=Path, default=DOCS_DIR / "null-explanation-report.md")
    parser.add_argument("--manifest", type=Path, default=PUBLIC_DATA_DIR / "malecns-arena-v1.manifest.json")
    parser.add_argument(
        "--skip-manifest-update",
        action="store_true",
        help="do not touch the shipped manifest (for tests against a temp copy)",
    )
    return parser.parse_args(argv)


def _load(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


def _sort_keys_deep(value: object) -> object:
    if isinstance(value, list):
        return [_sort_keys_deep(v) for v in value]
    if isinstance(value, dict):
        return {key: _sort_keys_deep(value[key]) for key in sorted(value.keys())}
    return value


def check_manifest_round_trips(manifest_path: Path) -> dict:
    """Verify `manifest_path` re-serializes byte-identically before anything
    is written -- the manifest is otherwise Python-written
    (`scripts/data/compile.py`) and this file is also Python, so the round
    trip holds unconditionally here (no JS float/non-ASCII formatting
    mismatch to guard against, unlike `scripts/null/null-report.ts`'s
    equivalent check). Returns the parsed manifest so `update_manifest`
    (or a caller running this as a preflight before any file is touched,
    per `main`'s write ordering) does not have to re-read and re-parse it."""
    original_text = manifest_path.read_text()
    manifest = json.loads(original_text)
    round_tripped = json.dumps(_sort_keys_deep(manifest), indent=2, sort_keys=True) + "\n"
    if round_tripped != original_text:
        raise ValueError(
            f"explain: re-serializing {manifest_path} without any change produced different bytes -- refusing to "
            "write, to avoid silently rewriting unrelated manifest bytes"
        )
    return manifest


def update_manifest(manifest_path: Path, entry: dict, manifest: dict | None = None) -> None:
    """Add/overwrite `nullExplanation` in place, matching
    `scripts/null/null-report.ts`'s `updateManifestWithRewiringNull`
    convention exactly (sorted keys, 2-space indent, trailing newline).
    `manifest`, if given, is the already-round-trip-checked dict from
    `check_manifest_round_trips` (`main` runs that check as a preflight,
    before any file is written); if omitted, this re-reads and re-checks
    `manifest_path` itself, for any other caller (e.g. a test) that wants
    the guard and the write in one call."""
    if manifest is None:
        manifest = check_manifest_round_trips(manifest_path)
    manifest = dict(manifest)
    manifest["nullExplanation"] = entry
    write_canonical_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    rewiring_null = _load(args.rewiring_null)
    variant_flip_both = _load(args.variant_flip_both)
    transfer_json = _load(args.transfer)
    features_json = _load(args.features)
    features_exploratory_json = _load(args.features_exploratory_unrestricted)
    regime_json = _load(args.regime)

    decoder_bio_percentile = variant_flip_both["bioPercentile"]
    single_axis_paths = {"flipThrust": args.variant_flip_thrust, "flipYaw": args.variant_flip_yaw}
    provided_single_axis_paths = {k: v for k, v in single_axis_paths.items() if v is not None}
    triggered = decoder_bio_percentile >= DECODER_PERCENTILE_THRESHOLD
    # Both-or-neither, and only on the branch the predeclared rule actually
    # calls for: a partial pair would publish an incomplete axis-attribution
    # story, and single-axis files supplied when the mirrored run did *not*
    # trigger them would silently look like they were part of the
    # predeclared procedure when they were not (a dual-review finding).
    if triggered and set(provided_single_axis_paths) != {"flipThrust", "flipYaw"}:
        raise ValueError(
            "explain: the mirrored variant's biological percentile "
            f"({decoder_bio_percentile * 100:.1f}%) meets the predeclared >= "
            f"{DECODER_PERCENTILE_THRESHOLD * 100:.0f}% threshold that triggers the single-axis runs "
            "(.agents/plans/null-explanation/00-overview.md) -- both --variant-flip-thrust and "
            "--variant-flip-yaw are required, not just one"
        )
    if not triggered and provided_single_axis_paths:
        raise ValueError(
            "explain: --variant-flip-thrust/--variant-flip-yaw were supplied, but the mirrored variant's "
            f"biological percentile ({decoder_bio_percentile * 100:.1f}%) did not meet the predeclared >= "
            f"{DECODER_PERCENTILE_THRESHOLD * 100:.0f}% threshold that triggers them -- omit these flags "
            "when the predeclared rule did not call for them"
        )
    provided_single_axis = {key: _load(path) for key, path in provided_single_axis_paths.items()}

    variants_loaded: dict[str, dict] = {"flipBoth": variant_flip_both, **provided_single_axis}
    verify_provenance(rewiring_null, variants_loaded, transfer_json, features_json, features_exploratory_json, regime_json)
    _require_complete_seed_coverage("regime.json rewired", [entry["seed"] for entry in regime_json["rewired"]])
    _require_complete_seed_coverage("rewiring-null-v1.json rewired", [entry["seed"] for entry in rewiring_null["rewired"]])

    variant_shas = {"flipBoth": sha256_hex(args.variant_flip_both.read_bytes())}
    variants: dict = {
        "flipBoth": {
            "bioPercentile": variant_flip_both["bioPercentile"],
            "pLow": variant_flip_both["pLow"],
            "pHigh": variant_flip_both["pHigh"],
            "nullMean": variant_flip_both["null"]["mean"],
            "bioScore": variant_flip_both["biological"]["score"],
        }
    }
    for key, payload in provided_single_axis.items():
        variant_shas[key] = sha256_hex(provided_single_axis_paths[key].read_bytes())
        variants[key] = {
            "bioPercentile": payload["bioPercentile"],
            "pLow": payload["pLow"],
            "pHigh": payload["pHigh"],
            "nullMean": payload["null"]["mean"],
            "bioScore": payload["biological"]["score"],
        }
    if not provided_single_axis:
        variants["singleAxisSkipped"] = True

    regime = compute_regime(regime_json, transfer_json)
    excluded = frozenset(regime["excludedGraphIds"])

    score_by_seed = {entry["seed"]: entry["score"] for entry in rewiring_null["rewired"]}
    metrics = build_all_metrics(transfer_json["graphs"], features_json["graphs"], score_by_seed, excluded)

    metrics_by_kind: dict[str, list[dict]] = {"transfer": [], "derived": [], "feature": []}
    for metric in metrics:
        metrics_by_kind[metric["kind"]].append(metric)

    # Feature 6's exploratory, non-predeclared unrestricted reading: the
    # same `weightedInDegree:*` metrics, computed by the same `build_metric`
    # pipeline, from `--features-exploratory-unrestricted` instead of the
    # frozen `--features`. No per-graph regime exclusion (features never
    # depend on the linear regime, same as the predeclared feature metrics).
    exploratory_metrics = [
        metric
        for name in (f"weightedInDegree:{population}" for population in OUTPUT_POPULATIONS)
        if (
            metric := build_metric(
                name, "feature", extract_feature_value, features_exploratory_json["graphs"], score_by_seed, frozenset(), BOOTSTRAP_BASE_SEED
            )
        )
        is not None
    ]

    finding_internal = evaluate_categories(
        decoder_bio_percentile,
        metrics_by_kind["transfer"] + metrics_by_kind["derived"],
        metrics_by_kind["feature"],
        regime,
    )
    structural_detail = finding_internal["structuralDetail"]
    # Reuses the single module-scope `qualifies` predicate (M3) instead of a
    # second hand-written copy of `evaluate_categories`'s own boolean.
    definition_sensitive = bool(
        structural_detail is not None
        and structural_detail["name"].startswith("weightedInDegree:")
        and not any(qualifies(m) for m in exploratory_metrics)
    )
    finding = {
        "categories": finding_internal["categories"],
        "ranked": finding_internal["ranked"],
        "regimeInvalid": finding_internal["regimeInvalid"],
        "definitionSensitive": definition_sensitive,
        "qualifyingMetrics": [
            {"name": m["name"], "kind": m["kind"], "spearman": m["spearman"]}
            for m in finding_internal["qualifyingMetrics"]
        ],
        "summarySentence": build_summary_sentence(finding_internal),
        "qualifyingMetricsNote": build_qualifying_metrics_note(finding_internal["qualifyingMetrics"]),
    }

    ordered_names = TRANSFER_METRIC_NAMES + DERIVED_METRIC_NAMES + FEATURE_METRIC_NAMES
    metrics_by_name = {m["name"]: m for m in metrics}
    ordered_metrics = [metrics_by_name[name] for name in ordered_names if name in metrics_by_name]
    constant_metric_count = sum(1 for m in ordered_metrics if m["nullConstant"])

    transfer_family = build_family_rank_matrix(
        metrics_by_kind["transfer"] + metrics_by_kind["derived"],
        lambda g, n: extract_transfer_value(g, n) if n.startswith("T:") else extract_derived_value(g, n),
        transfer_json["graphs"],
        excluded,
        REWIRED_COUNT,
    )
    feature_family = build_family_rank_matrix(
        metrics_by_kind["feature"], extract_feature_value, features_json["graphs"], frozenset(), REWIRED_COUNT
    )
    full_scores = np.array([score_by_seed[s] for s in range(REWIRED_COUNT)], dtype=np.float64)
    chance_rate = joint_permutation_chance_rate(
        [transfer_family, feature_family],
        full_scores,
        SPEARMAN_RHO_THRESHOLD,
        PERMUTATION_COUNT,
        np.random.default_rng(PERMUTATION_BASE_SEED),
    )

    calibration = {
        "metricsTested": len(ordered_metrics),
        "constantMetricCount": constant_metric_count,
        "permutations": PERMUTATION_COUNT,
        "chanceHits": round(chance_rate * PERMUTATION_COUNT),
        "chanceRate": chance_rate,
    }

    host = {"arch": regime_json["host"]["arch"], "node": regime_json["host"]["node"]}
    for label, payload in (("variant-flip-both", variant_flip_both), ("regime.json", regime_json), *(
        (f"variant-{key}", payload) for key, payload in provided_single_axis.items()
    )):
        if payload["host"] != host:
            raise ValueError(f"explain: {label}'s host {payload['host']} does not match {host}")

    # `transfer.json`/`features.json` are Python producers -- cross-check
    # their `producer.host.arch` against *each other* only, not against the
    # Node-recorded `host` above: `platform.machine()` and Node's
    # `process.arch` report the same physical ARM64 host differently
    # ("aarch64" vs "arm64", confirmed empirically on this study's own
    # Spark), so a cross-runtime check would spuriously refuse a consistent
    # run ("cross-check where applicable" means same-runtime producers only
    # -- see the report's Limitations section).
    transfer_producer_host = transfer_json["producer"]["host"]
    features_producer_host = features_json["producer"]["host"]
    if transfer_producer_host["arch"] != features_producer_host["arch"]:
        raise ValueError(
            f"explain: transfer.json's producer host arch ({transfer_producer_host['arch']!r}) does not match "
            f"features.json's ({features_producer_host['arch']!r}) -- these two Python producers should always "
            "run on the same host"
        )

    explanation = {
        "version": VERSION,
        "sources": {
            "rewiringNullSha256": sha256_hex(args.rewiring_null.read_bytes()),
            "variantShas": variant_shas,
            "transferSha": sha256_hex(args.transfer.read_bytes()),
            "featuresSha": sha256_hex(args.features.read_bytes()),
            "regimeSha": sha256_hex(args.regime.read_bytes()),
            # Producer code-identity blocks, recorded (not merely verified)
            # in the published artifact so a reader can see exactly what
            # code produced each input.
            "producers": {
                "transfer": transfer_json["producer"],
                "features": features_json["producer"],
                "regime": regime_json["producer"],
            },
        },
        "thresholds": {
            "decoderPercentile": DECODER_PERCENTILE_THRESHOLD,
            "spearmanRho": SPEARMAN_RHO_THRESHOLD,
            "clampFraction": CLAMP_FRACTION_THRESHOLD,
            "steadyStateDistance": STEADY_STATE_DISTANCE_THRESHOLD,
            "conditionNumber": CONDITION_NUMBER_THRESHOLD,
        },
        "variants": variants,
        "regime": regime,
        "metrics": ordered_metrics,
        "exploratory": {
            "featureSixUnrestricted": {
                "sourceSha256": sha256_hex(args.features_exploratory_unrestricted.read_bytes()),
                "metrics": exploratory_metrics,
            }
        },
        "calibration": calibration,
        "finding": finding,
        "host": host,
    }

    # Build everything that can still raise (the report render, and the
    # manifest's round-trip guard) *before* writing any file, so a failure
    # here never leaves a new artifact on disk next to a stale manifest sha
    # or a missing report -- the exact partial-publish state the "manifest
    # sha256 equals the artifact bytes" acceptance criterion exists to rule
    # out (a dual-review finding: an earlier version wrote the artifact
    # first, then ran the manifest guard, then rendered the report, any of
    # which could raise after the artifact was already on disk).
    artifact_text = canonical_json_text(explanation)
    artifact_sha256 = sha256_hex(artifact_text.encode("utf-8"))
    report_markdown = render_report_markdown(explanation, rewiring_null)
    manifest_dict = None if args.skip_manifest_update else check_manifest_round_trips(args.manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    graph_io.fsutil.atomic_write_text(args.out, artifact_text)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    graph_io.fsutil.atomic_write_text(args.report_out, report_markdown)
    if not args.skip_manifest_update:
        update_manifest(args.manifest, {"artifact": args.out.name, "sha256": artifact_sha256}, manifest_dict)

    print(
        f"explain: wrote {args.out} ({len(ordered_metrics)} metrics, sha256 {artifact_sha256[:12]}...) and "
        f"{args.report_out}"
    )


if __name__ == "__main__":
    main()
