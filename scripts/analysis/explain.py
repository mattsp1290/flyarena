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
  evaluation (see `render_feature6_disclosure`'s doc comment);
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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import graph_io  # noqa: E402
from graph_io import canonical_json_text, sha256_hex, write_canonical_json  # noqa: E402
from features import OBSERVATION_CHANNELS, OUTPUT_POPULATIONS  # noqa: E402
from transfer import ILL_CONDITIONED_THRESHOLD, OBSERVATION_CHANNEL_INDEX, OUTPUT_POPULATION_INDEX  # noqa: E402

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
# Empirical quantiles / rank statistics -- mirrors
# `scripts/null/null-stats.ts`'s `nullSummary`/`rankStatistics` exactly (the
# same low-tail-floor / high-tail-ceil-minus-one convention), so this
# module's per-metric percentile and range statistics are computed the same
# way as the rest of this study's null artifacts (`03-explanation-report.md`:
# "same tie rule as `scripts/null/null-stats.ts`").
# ---------------------------------------------------------------------------


def quantile_index(n: int, p: float) -> int:
    if p <= 0.5:
        return int(np.floor(p * n))
    return min(n - 1, int(np.ceil(p * n)) - 1)


def null_range_summary(values: Sequence[float]) -> dict:
    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)
    if n == 0:
        raise ValueError("explain: null_range_summary requires at least one value")
    median = (
        sorted_values[(n - 1) // 2]
        if n % 2 == 1
        else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    )
    return {
        "median": float(median),
        "p2_5": float(sorted_values[quantile_index(n, 0.025)]),
        "p97_5": float(sorted_values[quantile_index(n, 0.975)]),
    }


def rank_statistics(null_values: Sequence[float], bio_value: float) -> dict:
    n = len(null_values)
    if n == 0:
        raise ValueError("explain: rank_statistics requires a non-empty null set")
    k_below = sum(1 for v in null_values if v < bio_value)
    k_equal = sum(1 for v in null_values if v == bio_value)
    return {"kBelow": k_below, "kEqual": k_equal, "bioPercentile": (k_below + 0.5 * k_equal) / n}


def outside_range(bio_value: float, p2_5: float, p97_5: float) -> bool:
    return bio_value < p2_5 or bio_value > p97_5


# ---------------------------------------------------------------------------
# Spearman correlation + bootstrap CI
#
# Ranks use `pandas.Series.rank(method="average")` (average-tie ranking,
# `scipy`-equivalent) rather than a plain double-argsort -- several feature
# metrics are small integer counts (`twoCycleCount`, path lengths) with real
# ties, and an untied rank would silently misstate rho for those (a concern
# the WP2 feature-6 adjudication debate's own standalone script flagged and
# did not resolve; `scipy` is not a project dependency, so this reuses the
# `pandas` this repo already depends on instead of adding one).
#
# The bootstrap CI resamples the *already-ranked* pairs (not raw values) and
# does not re-rank within each resample -- the standard, vectorizable way to
# bootstrap a rank correlation's CI, and honest to disclose: a resample with
# repeated indices has repeated rank values, so this is a bootstrap of "the
# correlation of this fixed rank-transformed sample," not a full re-ranking
# bootstrap. For a 500-point, effectively-tie-free biological/null sample
# this distinction is immaterial; it is called out here and in the report's
# limitations because a fully faithful bootstrap is not what is computed.
# ---------------------------------------------------------------------------


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.sqrt(np.sum(x_c * x_c) * np.sum(y_c * y_c)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_c * y_c) / denom)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(x), _rank(y))


def metric_rng(base_seed: int, label: str) -> np.random.Generator:
    """A deterministic, per-metric-independent RNG stream: every metric's
    bootstrap CI is an independent function of `(base_seed, its own name)`,
    the same design `scripts/training/stats.ts`'s `conditionRng` uses for
    this study's other bootstrap CIs (`scripts/null/null-stats.ts`'s doc
    comment), reimplemented here in Python via a sha256 digest of the label
    rather than that file's own string-hash + `xoshiro`-style generator
    (not reused across languages -- this module owns its own, equally
    deterministic, construction)."""
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed)


def bootstrap_spearman_ci(
    rank_x: np.ndarray, rank_y: np.ndarray, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    n = rank_x.shape[0]
    idx = rng.integers(0, n, size=(resamples, n))
    rx = rank_x[idx]
    ry = rank_y[idx]
    rx_c = rx - rx.mean(axis=1, keepdims=True)
    ry_c = ry - ry.mean(axis=1, keepdims=True)
    num = np.sum(rx_c * ry_c, axis=1)
    den = np.sqrt(np.sum(rx_c * rx_c, axis=1) * np.sum(ry_c * ry_c, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rhos = np.where(den > 0, num / den, 0.0)
    sorted_rhos = np.sort(rhos)
    lo = sorted_rhos[quantile_index(resamples, 0.025)]
    hi = sorted_rhos[quantile_index(resamples, 0.975)]
    return float(lo), float(hi)


def permutation_chance_rate(
    rank_matrix: np.ndarray,
    rank_score: np.ndarray,
    threshold: float,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    """`00-overview.md`'s multiple-comparisons calibration: across
    `permutations` random re-pairings of the score vector against the fixed
    metric matrix, the fraction of permutations where *at least one* of
    `rank_matrix`'s columns reaches `|rho| >= threshold` by chance alone.
    `rank_matrix` is `(n_graphs, n_metrics)`, already rank-transformed (see
    `spearman_rho`'s doc comment for why ranking, not raw values, is used);
    `rank_score` is `(n_graphs,)`. Vectorized over all metrics per
    permutation (a Python loop only over the 1,000 permutations, not over
    each of the ~66 metrics inside it)."""
    metric_c = rank_matrix - rank_matrix.mean(axis=0, keepdims=True)
    metric_denom = np.sqrt(np.sum(metric_c * metric_c, axis=0))
    score_c = rank_score - rank_score.mean()
    score_denom = float(np.sqrt(np.sum(score_c * score_c)))
    hits = 0
    n = rank_score.shape[0]
    for _ in range(permutations):
        perm = rng.permutation(n)
        permuted = score_c[perm]
        num = metric_c.T @ permuted
        denom = metric_denom * score_denom
        with np.errstate(invalid="ignore", divide="ignore"):
            rhos = np.where(denom > 0, num / denom, 0.0)
        if np.max(np.abs(rhos)) >= threshold:
            hits += 1
    return hits / permutations


def joint_permutation_chance_rate(
    families: Sequence[tuple[np.ndarray, np.ndarray]],
    full_scores: np.ndarray,
    threshold: float,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    """The real multiple-comparisons calibration across *every* metric
    family at once: `families` is `[(rank_matrix, seed_indices), ...]`, one
    pair per metric family (transfer/derived, structural-feature), where
    `rank_matrix` is that family's `(n_family, n_metrics)` array of metric
    ranks (fixed, computed once from the real, unpermuted scores via
    `_family_rank_matrix`) and `seed_indices` are the 0-based seeds each row
    corresponds to, indexing into `full_scores` (length `REWIRED_COUNT`).
    One shared permutation of the *full* score vector is drawn per
    iteration, then subset (and re-ranked, since a subset's own average-tie
    ranks differ from the full array's) per family -- both families see the
    same underlying seed-to-score re-pairing, not two independently permuted
    draws -- and a permutation counts as a hit if *any* family reaches the
    threshold, which is the actual "at least one of the ~66 metrics" union
    probability `00-overview.md` describes.

    Replaces an earlier version that ran two independent single-family
    permutation loops (`permutation_chance_rate`, above) and reported their
    `max`, mislabeled "the more conservative (larger) chance rate" -- for a
    union of two families, P(A or B) >= max(P(A), P(B)), so `max`
    *understates* the true union rate (a dual-review finding, both
    reviewers independently)."""
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for rank_matrix, seed_indices in families:
        if rank_matrix.shape[1] == 0:
            continue
        metric_c = rank_matrix - rank_matrix.mean(axis=0, keepdims=True)
        metric_denom = np.sqrt(np.sum(metric_c * metric_c, axis=0))
        prepared.append((metric_c, metric_denom, seed_indices))
    if not prepared:
        return 0.0
    n_full = full_scores.shape[0]
    hits = 0
    for _ in range(permutations):
        permuted_full = full_scores[rng.permutation(n_full)]
        hit = False
        for metric_c, metric_denom, seed_indices in prepared:
            ry = _rank(permuted_full[seed_indices])
            y_c = ry - ry.mean()
            y_denom = float(np.sqrt(np.sum(y_c * y_c)))
            num = metric_c.T @ y_c
            denom = metric_denom * y_denom
            with np.errstate(invalid="ignore", divide="ignore"):
                rhos = np.where(denom > 0, num / denom, 0.0)
            if np.max(np.abs(rhos)) >= threshold:
                hit = True
                break
        hits += int(hit)
    return hits / permutations


def build_family_rank_matrix(
    metrics: Sequence[dict], extractor, graphs: Mapping[str, dict], excluded_ids: frozenset[str]
) -> tuple[np.ndarray, np.ndarray]:
    """`(rank_matrix, seed_indices)` for `joint_permutation_chance_rate`:
    `rank_matrix[i, j]` is metric `j`'s average rank at seed `seed_indices[i]`
    (restricted to non-excluded seeds), for every metric in `metrics`.
    Raises rather than silently letting a `None` extracted value become
    `NaN` through `numpy`/`pandas` -- an all-`NaN`-propagated column would
    otherwise make that column's rho `NaN` in every permutation, and because
    `NaN >= threshold` is `False`, would silently make that column *never*
    register a hit, understating the chance rate without raising anywhere
    (an edge-case-review finding). This study's real `features.json` has no
    `None`s among non-excluded rewired graphs, so this path was previously
    unexercised, not previously safe."""
    seed_indices = np.array([s for s in range(REWIRED_COUNT) if f"rewired-{s}" not in excluded_ids], dtype=np.int64)
    columns = []
    for metric in metrics:
        raw = [extractor(graphs[f"rewired-{s}"], metric["name"]) for s in seed_indices]
        if any(v is None for v in raw):
            raise ValueError(
                f"explain: {metric['name']} has a missing value on a non-excluded rewiring -- the permutation "
                "calibration requires a complete column"
            )
        columns.append(_rank(np.asarray(raw, dtype=np.float64)))
    rank_matrix = np.stack(columns, axis=1) if columns else np.zeros((seed_indices.shape[0], 0))
    return rank_matrix, seed_indices


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

    return {
        "bio": bio_regime,
        "nullSampleMedian": {"clampFraction": null_median_clamp, "steadyStateDistance": null_median_distance},
        "gatePassed": bool(gate_passed),
        "excludedGraphIds": excluded_graph_ids,
        "excludedCount": len(excluded_graph_ids),
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
    # Spearman rho, not a zero one -- `_pearson` returns `0.0` as a safe
    # internal sentinel (it can never spuriously clear `SPEARMAN_RHO_
    # THRESHOLD`), but reporting "0.000 [0.000, 0.000]" to a reader would
    # read as a precisely measured null correlation rather than "not
    # computable" (a rigor-review finding). `nullConstant` lets
    # `render_metric_stats_table` show "n/a" instead.
    null_constant = bool(np.ptp(metric_values) == 0.0)

    rx = _rank(metric_values)
    ry = _rank(scores)
    rho = _pearson(rx, ry)
    rng = metric_rng(bootstrap_base_seed, name)
    ci_lo, ci_hi = bootstrap_spearman_ci(rx, ry, BOOTSTRAP_RESAMPLES, rng)

    return {
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

    def qualifies(metric: dict) -> bool:
        return outside_range(metric["bio"], metric["p2_5"], metric["p97_5"]) and abs(metric["spearman"]) >= SPEARMAN_RHO_THRESHOLD

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

    return {
        "categories": categories,
        "ranked": ranked,
        "unexplained": unexplained,
        "regimeInvalid": regime_invalid,
        "decoderBioPercentile": decoder_bio_percentile,
        "linearDetail": linear_detail,
        "structuralDetail": structural_detail,
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}th percentile" if value not in (0.0, 1.0) else (
        "0th percentile" if value == 0.0 else "100th percentile"
    )


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


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _require_matching_source(label: str, value: str, expected: str) -> None:
    if value != expected:
        raise ValueError(f"explain: {label} sourceGraphSha256/rewireSourceSha256 does not match rewiring-null-v1.json")


def _require_complete_seed_coverage(label: str, seeds: Sequence[int]) -> None:
    """A rewired-graph seed missing from `regime.json` or
    `rewiring-null-v1.json` would otherwise be silently dropped: a seed
    absent from `regime.json["rewired"]` is never checked against the
    per-graph regime thresholds (so it can never be excluded, correctly or
    not), and a seed absent from `rewiring-null-v1.json["rewired"]` is
    simply missing from `score_by_seed`, which raises a `KeyError` deep in
    `build_metric` with no context about which input was short. A duplicate
    seed would silently overwrite a dict entry the same way. Checked once,
    loudly, at load time (an edge-case-review finding: previously
    unchecked)."""
    if sorted(seeds) != list(range(REWIRED_COUNT)):
        raise ValueError(f"explain: {label} does not cover rewired seeds 0..{REWIRED_COUNT - 1} exactly once")


def verify_provenance(
    rewiring_null: dict,
    variants: Mapping[str, dict],
    transfer_json: dict,
    features_json: dict,
    features_exploratory_json: dict,
    regime_json: dict,
) -> None:
    """`variants` is every loaded decoder-variant payload (flip-both, plus
    any provided single-axis ones) -- checked in one loop rather than a
    fixed `variant_flip_both` parameter, so the (previously unchecked)
    single-axis path gets the same provenance guarantee (an edge-case-review
    finding)."""
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


# ---------------------------------------------------------------------------
# Report markdown
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_transfer_matrix(metrics_by_name: Mapping[str, dict], field: str) -> str:
    """`metrics_by_name.get(...)` (not direct indexing): if biological's own
    transfer solve is singular, `build_all_metrics` drops every `T:*` metric
    entirely (`build_metric` returns `None` when `bio_value is None`), so a
    direct `metrics_by_name[name]` would raise `KeyError` here -- reported
    as "n/a" instead (an edge-case-review finding: this study's actual data
    never singular, so this path was previously unexercised and untested)."""
    header = "| channel \\ population | " + " | ".join(OUTPUT_POPULATIONS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in OUTPUT_POPULATIONS) + " |"
    rows = [header, sep]
    for channel in OBSERVATION_CHANNELS:
        cells = [
            _fmt(metrics_by_name.get(f"T:{channel}->{population}", {}).get(field), 6)
            for population in OUTPUT_POPULATIONS
        ]
        rows.append(f"| {channel} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_metric_stats_table(metrics: Sequence[dict]) -> str:
    header = "| metric | biological | null median | null 2.5% | null 97.5% | bio percentile | rho | rho 95% CI |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for metric in sorted(metrics, key=lambda m: abs(m["spearman"]), reverse=True):
        # A metric constant across the null has an undefined (not zero)
        # Spearman rho -- shown as "n/a" rather than the internal `0.0`
        # sentinel `build_metric` stores (a rigor-review finding).
        if metric.get("nullConstant"):
            rho_cell, ci_cell = "n/a (constant in null)", "n/a"
        else:
            rho_cell = f"{metric['spearman']:.3f}"
            ci_cell = f"[{metric['spearmanCi'][0]:.3f}, {metric['spearmanCi'][1]:.3f}]"
        rows.append(
            f"| {metric['name']} | {_fmt(metric['bio'])} | {_fmt(metric['nullMedian'])} | {_fmt(metric['p2_5'])} | "
            f"{_fmt(metric['p97_5'])} | {metric['bioPercentile'] * 100:.1f}% | {rho_cell} | {ci_cell} |"
        )
    return "\n".join(rows)


def render_feature6_disclosure(explanation: dict) -> list[str]:
    """The feature-6 (`weightedInDegree`) adjudication disclosure, built
    entirely from computed values in `explanation["exploratory"]` (never
    hardcoded prose numbers) -- a dual-review finding: an earlier version
    hardcoded the exploratory statistics as prose and asserted the restricted
    reading was "decided before any result was seen", which the repository's
    own timestamps and the adjudication debate's write-ups (both computed
    and compared each reading's outcome, including its Spearman rho against
    score, before the restricted reading was adopted) contradict. This
    version states what actually happened and computes every number from a
    sha-pinned input (`--features-exploratory-unrestricted`) instead of an
    unpinned, session-local `/tmp` citation."""
    exploratory = explanation["exploratory"]["featureSixUnrestricted"]
    restricted_by_name = {
        m["name"]: m
        for m in explanation["metrics"]
        if m["kind"] == "feature" and m["name"].startswith("weightedInDegree:")
    }
    exploratory_by_name = {m["name"]: m for m in exploratory["metrics"]}
    thresholds = explanation["thresholds"]

    lines = [
        "**Feature 6 adjudication.** `weightedInDegree` (mean weighted in-degree per output population) was "
        "first implemented and run **unrestricted** (counting edges from any presynaptic neuron), matching one "
        "reading of the plan's ambiguous \"input->output weighted in-degree\" wording. A review flagged that "
        "wording as ambiguous against features 1/2's own restrictive use of \"input\" (channel-mapped neurons "
        "only); the resulting adjudication computed **both** readings' full statistics -- including each reading's "
        "rank correlation with score across all 500 rewirings -- before the input-restricted reading was adopted "
        "on plan-text grounds (bean `flyarena-r37r`'s log). Because both readings' outcomes were visible before "
        "the decision, this was not a fully outcome-blind pre-registration, and the `structuralFeature` finding "
        "below should be read with that limitation in mind, not as a clean, one-shot predeclared test."
    ]
    lines.append("")
    lines.append(
        "The unrestricted reading is disclosed here as **exploratory, non-predeclared**: it is not part of the "
        "frozen 40-feature list and plays no role in the outcome-category evaluation. Both readings, computed by "
        "this same pipeline (`exploratory.featureSixUnrestricted.sourceSha256` = "
        f"`{exploratory['sourceSha256'][:12]}...`):"
    )
    lines.append("")
    lines.append(
        "| population | restricted (predeclared) bio | restricted rho | unrestricted (exploratory) bio | "
        "unrestricted rho |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for population in OUTPUT_POPULATIONS:
        restricted = restricted_by_name[f"weightedInDegree:{population}"]
        unrestricted = exploratory_by_name[f"weightedInDegree:{population}"]
        lines.append(
            f"| {population} | {_fmt(restricted['bio'])} | {restricted['spearman']:.3f} | "
            f"{_fmt(unrestricted['bio'])} | {unrestricted['spearman']:.3f} |"
        )
    lines.append("")
    max_unrestricted_rho = max(abs(m["spearman"]) for m in exploratory["metrics"])
    lines.append(
        f"The unrestricted reading's strongest population correlation is \\|rho\\| = {max_unrestricted_rho:.3f}, "
        f"below the predeclared {thresholds['spearmanRho']} threshold on every population -- under the "
        "unrestricted reading, feature 6 would not itself qualify for the structural-feature-associated category "
        "on any population."
    )
    if explanation["finding"].get("definitionSensitive"):
        lines.append("")
        lines.append(
            "**This report's `structuralFeature` finding is definition-sensitive**: it is triggered by a "
            "`weightedInDegree` entry, and the finding would not hold under the unrestricted reading above."
        )
    return lines


def render_report_markdown(explanation: dict, rewiring_null: dict) -> str:
    """`rewiring_null` (the already-loaded, already-sha-verified
    `rewiring-null-v1.json`, `explanation["sources"]["rewiringNullSha256"]`'s
    own source) supplies the authored-condition baseline numbers quoted in
    the "Question" section and the decoder-convention table's first row --
    read from that file directly rather than hardcoded, so this report can
    never drift from the artifact it is describing."""
    metrics_by_name = {m["name"]: m for m in explanation["metrics"]}
    transfer_metrics = [m for m in explanation["metrics"] if m["kind"] == "transfer"]
    derived_metrics = [m for m in explanation["metrics"] if m["kind"] == "derived"]
    feature_metrics = [m for m in explanation["metrics"] if m["kind"] == "feature"]
    finding = explanation["finding"]
    regime = explanation["regime"]
    variants = explanation["variants"]
    calibration = explanation["calibration"]
    authored_bio_score = rewiring_null["biological"]["score"]
    authored_null_mean = rewiring_null["null"]["mean"]
    authored_null_std = rewiring_null["null"]["std"]
    authored_bio_percentile = rewiring_null["bioPercentile"]
    authored_p_low = rewiring_null["pLow"]
    authored_p_high = rewiring_null["pHigh"]

    lines: list[str] = []
    lines.append("# Explaining the null result (under this model)")
    lines.append("")
    lines.append(
        "**Question.** `docs/rewiring-null-report.md` found the biological MaleCNS graph scoring below all 500 "
        f"degree-preserving rewirings under the authored decoder (biological {authored_bio_score:.4f}, null mean "
        f"{authored_null_mean:.4f}, sd {authored_null_std:.4f}, {authored_bio_percentile * 100:.1f}th percentile, "
        f"`p_low = {authored_p_low:.4f}`). This report tests three predeclared, descriptive explanations for "
        "that result under this model only -- the authored encoder, this rate-model dynamics, and this arena -- "
        "and makes no claim about the real fly."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Three predeclared analyses (`.agents/plans/null-explanation/00-overview.md`), evaluated only after all "
        "three finished (see the feature-6 disclosure under \"Structural features\" below for one qualification "
        "to the feature list's predeclaration):"
    )
    lines.append("")
    lines.append(
        "1. **Decoder-convention check.** Re-score biological and all 500 rewirings with the authored decoder's "
        "thrust and yaw signs both flipped (`authored-flip-both`). Predeclared rule: only run the two single-axis "
        "variants if the mirrored run moves biological to at least the 25th percentile."
    )
    lines.append(
        "2. **Linear transfer analysis.** For each graph, the steady-state linear transfer matrix "
        "`T = O(lambda I - g A)^-1 B` (3 outputs x 8 input channels), gated by a linear-regime validity check."
    )
    lines.append(
        "3. **Structural feature attribution.** 40 predeclared graph features, each compared against the null and "
        "rank-correlated with score."
    )
    lines.append("")
    lines.append("**Predeclared thresholds:**")
    lines.append("")
    thresholds = explanation["thresholds"]
    lines.append("| Threshold | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Decoder-convention bio percentile | >= {thresholds['decoderPercentile'] * 100:.0f}% |")
    lines.append(f"| Spearman \\|rho\\| (linear-pathway / structural-feature) | >= {thresholds['spearmanRho']} |")
    lines.append(f"| Regime gate: rate-clamp fraction | <= {thresholds['clampFraction'] * 100:.0f}% |")
    lines.append(f"| Regime gate: steady-state distance | <= {thresholds['steadyStateDistance']} |")
    lines.append(f"| Regime gate: condition number | <= {thresholds['conditionNumber']:.0e} |")
    lines.append("")
    lines.append(
        f"**Multiple comparisons.** {calibration['metricsTested']} metrics are tested (24 transfer entries, 2 "
        f"derived predictors, 40 structural features; {calibration['constantMetricCount']} of these are constant "
        "across the null and so can never reach the |rho| threshold). Correlations are reported descriptively, "
        "without per-metric significance testing; a permutation calibration "
        f"({calibration['permutations']} seeded permutations of the score, applied jointly to every metric family "
        "at once) found that at least one of the tested metrics reaches "
        f"\\|rho\\| >= {thresholds['spearmanRho']} by chance alone in {calibration['chanceHits']} of "
        f"{calibration['permutations']} permutations ({calibration['chanceRate'] * 100:.1f}%) -- this chance rate "
        "applies to any triggered linear-pathway or structural-feature finding below."
    )
    lines.append("")

    lines.append("## Decoder-convention check")
    lines.append("")
    flip_both = variants["flipBoth"]
    lines.append("| Condition | Biological score | Null mean | Bio percentile | p_low | p_high |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| authored, opponent parked | {authored_bio_score:.4f} | {authored_null_mean:.4f} | "
        f"{authored_bio_percentile * 100:.1f}% | {authored_p_low:.4f} | {authored_p_high:.4f} |"
    )
    lines.append(
        f"| authored (thrust and yaw flipped), opponent parked | {flip_both['bioScore']:.4f} | "
        f"{flip_both['nullMean']:.4f} | {flip_both['bioPercentile'] * 100:.1f}% | {flip_both['pLow']:.4f} | "
        f"{flip_both['pHigh']:.4f} |"
    )
    single_axis_labels = {
        "flipThrust": "authored (thrust flipped), opponent parked",
        "flipYaw": "authored (yaw flipped), opponent parked",
    }
    for key, label in single_axis_labels.items():
        if key in variants:
            entry = variants[key]
            lines.append(
                f"| {label} | {entry['bioScore']:.4f} | {entry['nullMean']:.4f} | "
                f"{entry['bioPercentile'] * 100:.1f}% | {entry['pLow']:.4f} | {entry['pHigh']:.4f} |"
            )
    lines.append("")
    if variants.get("singleAxisSkipped"):
        lines.append(
            f"Mirrored biological percentile ({flip_both['bioPercentile'] * 100:.1f}%) stayed below the predeclared "
            f"{thresholds['decoderPercentile'] * 100:.0f}% threshold, so the single-axis variants "
            "(`authored-flip-thrust`, `authored-flip-yaw`) were skipped per the predeclared rule "
            "(`.agents/plans/null-explanation/00-overview.md`)."
        )
    lines.append("")

    lines.append("## Linear transfer analysis")
    lines.append("")
    lines.append(
        "`T[channel, population]`: steady-state gain from a unit-held input on that channel to that output "
        "population, exact within the model's rate/input clamps."
    )
    lines.append("")
    lines.append("**Biological `T`:**")
    lines.append("")
    lines.append(render_transfer_matrix(metrics_by_name, "bio"))
    lines.append("")
    lines.append("**Null median `T`:**")
    lines.append("")
    lines.append(render_transfer_matrix(metrics_by_name, "nullMedian"))
    lines.append("")
    lines.append("**Transfer entry statistics (sorted by \\|rho\\|):**")
    lines.append("")
    lines.append(render_metric_stats_table(transfer_metrics))
    lines.append("")
    lines.append("**Derived predictors** (`turnGain = T[yaw,foodBearing] - T[yaw,hazardBearing]`, "
                  "`approachGain = T[thrust,foodDistance]`):")
    lines.append("")
    lines.append(render_metric_stats_table(derived_metrics))
    lines.append("")

    lines.append("## Regime check")
    lines.append("")
    lines.append(
        "Authored episodes on 10 held-out seeds (`30001..30010`) for biological, disconnected, and all 500 "
        "rewirings, measuring the fraction of neuron-substeps with an active rate clamp and the linear steady-state "
        "distance `||r_t - r*(u_t)|| / ||r*(u_t)||`."
    )
    lines.append("")
    lines.append("| | rate-clamp fraction | steady-state distance |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| biological | {regime['bio']['clampFraction'] * 100:.2f}% | {regime['bio']['steadyStateDistance']:.4f} |")
    lines.append(
        f"| null (median over 500 rewirings) | {regime['nullSampleMedian']['clampFraction'] * 100:.2f}% | "
        f"{regime['nullSampleMedian']['steadyStateDistance']:.4f} |"
    )
    lines.append("")
    lines.append(
        f"{regime['excludedCount']} of 500 rewirings were individually excluded from the transfer-kind "
        "correlations above for failing their own per-graph regime threshold "
        f"({', '.join(regime['excludedGraphIds']) if regime['excludedGraphIds'] else 'none'})."
    )
    lines.append("")
    if regime["gatePassed"]:
        lines.append(
            "The aggregate regime gate **passed**: biological's steady-state distance "
            f"({regime['bio']['steadyStateDistance']:.4f}) and the null median's "
            f"({regime['nullSampleMedian']['steadyStateDistance']:.4f}) are both at or below the "
            f"{thresholds['steadyStateDistance']} threshold; biological's rate-clamp fraction "
            f"({regime['bio']['clampFraction'] * 100:.2f}%) and the null median's "
            f"({regime['nullSampleMedian']['clampFraction'] * 100:.2f}%) are both at or below "
            f"{thresholds['clampFraction'] * 100:.0f}%; and biological's transfer solve is not singular, "
            "ill-conditioned, or unstable. This licenses treating the linear analysis as applicable to both "
            "biological and the null sample under this model; it does not by itself certify that any single "
            "transfer entry explains the score -- that still requires the outside-range-and-\\|rho\\|-threshold "
            "test above."
        )
    else:
        lines.append(
            "The aggregate regime gate **failed**: biological's or the null median's steady-state distance or "
            "rate-clamp fraction exceeded threshold, or biological's own transfer solve was singular, "
            "ill-conditioned, or unstable. Per the predeclared rule, the linear transfer analysis is therefore "
            "reported as **regime-invalid (inconclusive)** and is never reported as a positive `linearPathway` "
            "finding, regardless of any individual transfer entry's statistics above."
        )
    lines.append("")

    lines.append("## Structural features")
    lines.append("")
    lines.append(
        "40 predeclared graph features (fixed before any analysis ran; the frozen list is not edited after the "
        "first *production* run against it -- see the feature-6 disclosure below for what happened before that)."
    )
    lines.append("")
    lines.extend(render_feature6_disclosure(explanation))
    lines.append("")
    lines.append(render_metric_stats_table(feature_metrics))
    lines.append("")

    lines.append("## Finding")
    lines.append("")
    lines.append(finding["summarySentence"])
    lines.append("")
    if finding["categories"]:
        lines.append(f"Categories that hold, ranked by effect size: {', '.join(finding['ranked'])}.")
    else:
        lines.append("No predeclared category holds.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        f"- **{calibration['metricsTested']} metrics tested.** No per-metric significance testing is performed; "
        f"correlations are descriptive. The permutation calibration above found a {calibration['chanceRate'] * 100:.1f}% "
        "chance that at least one metric reaches the |rho| threshold by chance alone, and that rate applies to any "
        "triggered linear-pathway or structural-feature finding in this report."
    )
    lines.append(
        "- **Regime-invalid is never reported as a positive finding.** If the aggregate regime gate fails, the "
        "linear-pathway analysis is reported as regime-invalid (inconclusive), never as a positive finding, "
        "regardless of any individual transfer entry's statistics."
    )
    lines.append(
        "- **This model only.** Every analysis here describes the authored decoder, this rate-model dynamics, and "
        "this arena running on the measured biological topology versus 500 degree-preserving rewirings of it. "
        "Nothing here is a claim about the real fly's neural function or behavior, and no rewiring's topology is "
        "claimed to be causally \"worse\" or \"better\" than biological's."
    )
    lines.append(
        "- **The linear analysis is valid only to the measured regime extent.** `T` is the model's exact "
        "fixed-point gain when no rate/input clamp is active; the regime check quantifies how close the real, "
        "clamped, discretized simulation actually sits to that fixed point, and the linear-pathway category is "
        "gated on that check, not assumed."
    )
    lines.append(
        "- **Correlation is not causation.** A rank correlation between a structural or transfer metric and score "
        "across the 500 rewirings describes an association within this null model's sample, not a causal "
        "mechanism."
    )
    lines.append(
        "- **Bootstrap CIs are approximate.** Each metric's 95% Spearman CI resamples the already rank-transformed "
        "pairs and does not re-rank within each resample -- a bootstrap of the rank-transformed sample's Pearson "
        "correlation, not a fully faithful re-ranking bootstrap. The CIs are descriptive only and play no role in "
        "any outcome-category decision (only the point estimate and the predeclared |rho| threshold do)."
    )
    lines.append(
        "- **A metric constant across the null (`n/a (constant in null)` in the tables above) has an undefined, "
        "not zero, Spearman correlation** and can never trigger the |rho| threshold; it is still counted toward "
        "the metrics-tested total above."
    )
    lines.append("- **No biological claim.** See \"This model only\" above.")
    lines.append("")
    return "\n".join(lines)


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
    definition_sensitive = bool(
        structural_detail is not None
        and structural_detail["name"].startswith("weightedInDegree:")
        and not any(
            outside_range(m["bio"], m["p2_5"], m["p97_5"]) and abs(m["spearman"]) >= SPEARMAN_RHO_THRESHOLD
            for m in exploratory_metrics
        )
    )
    finding = {
        "categories": finding_internal["categories"],
        "ranked": finding_internal["ranked"],
        "regimeInvalid": finding_internal["regimeInvalid"],
        "definitionSensitive": definition_sensitive,
        "summarySentence": build_summary_sentence(finding_internal),
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
    )
    feature_family = build_family_rank_matrix(
        metrics_by_kind["feature"], extract_feature_value, features_json["graphs"], frozenset()
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

    explanation = {
        "version": VERSION,
        "sources": {
            "rewiringNullSha256": sha256_hex(args.rewiring_null.read_bytes()),
            "variantShas": variant_shas,
            "transferSha": sha256_hex(args.transfer.read_bytes()),
            "featuresSha": sha256_hex(args.features.read_bytes()),
            "regimeSha": sha256_hex(args.regime.read_bytes()),
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
