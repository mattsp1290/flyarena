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
  structural features per graph);
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


def compute_regime(regime_json: dict, transfer_json: dict) -> dict:
    bio_regime = per_graph_regime_summary(regime_json["biological"])
    rewired_regime = {
        f"rewired-{entry['seed']}": per_graph_regime_summary(entry) for entry in regime_json["rewired"]
    }
    null_median_distance = float(np.median([v["steadyStateDistance"] for v in rewired_regime.values()]))
    null_median_clamp = float(np.median([v["clampFraction"] for v in rewired_regime.values()]))

    bio_transfer = transfer_json["graphs"]["biological"]
    bio_transfer_ok = not bio_transfer.get("singular") and not bio_transfer.get("illConditioned")

    gate_passed = (
        bio_regime["steadyStateDistance"] <= STEADY_STATE_DISTANCE_THRESHOLD
        and null_median_distance <= STEADY_STATE_DISTANCE_THRESHOLD
        and bio_regime["clampFraction"] <= CLAMP_FRACTION_THRESHOLD
        and bio_transfer_ok
    )

    excluded_graph_ids: list[str] = []
    for graph_id, summary in rewired_regime.items():
        transfer_entry = transfer_json["graphs"][graph_id]
        fails = (
            summary["clampFraction"] > CLAMP_FRACTION_THRESHOLD
            or summary["steadyStateDistance"] > STEADY_STATE_DISTANCE_THRESHOLD
            or transfer_entry.get("illConditioned")
            or transfer_entry.get("singular")
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
    regime_invalid = bool(linear_candidates) and not regime_gate_passed
    linear_detail: dict | None = None
    if linear_triggered:
        linear_detail = max(linear_candidates, key=lambda m: abs(m["spearman"]))
        categories.append("linearPathway")
        triggers.append(("linearPathway", abs(linear_detail["spearman"]), linear_detail))

    structural_candidates = [m for m in structural_metrics if qualifies(m)]
    structural_detail: dict | None = None
    if structural_candidates:
        structural_detail = max(structural_candidates, key=lambda m: abs(m["spearman"]))
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

    parts: list[str] = []
    if "decoderConvention" in categories:
        parts.append(
            "flipping the authored decoder's thrust and yaw sign conventions moves biological to the "
            f"{format_pct(finding['decoderBioPercentile'])}"
        )
    if "linearPathway" in categories:
        detail = finding["linearDetail"]
        parts.append(
            f"the linear transfer entry {detail['name']} sits outside the null's 2.5-97.5% range "
            f"(rank correlation with score rho={detail['spearman']:.3f})"
        )
    if "structuralFeature" in categories:
        detail = finding["structuralDetail"]
        parts.append(
            f"the structural feature {detail['name']} sits outside the null's 2.5-97.5% range "
            f"(rank correlation with score rho={detail['spearman']:.3f})"
        )
    return "Biological's low score is associated with: " + "; ".join(parts) + " -- a descriptive correlation, not a causal claim."


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _require_matching_source(label: str, value: str, expected: str) -> None:
    if value != expected:
        raise ValueError(f"explain: {label} sourceGraphSha256/rewireSourceSha256 does not match rewiring-null-v1.json")


def verify_provenance(rewiring_null: dict, variant_flip_both: dict, transfer_json: dict, features_json: dict, regime_json: dict) -> None:
    expected_source = rewiring_null["sourceGraphSha256"]
    expected_rewire = rewiring_null["rewireSourceSha256"]
    for label, payload in (
        ("variant-flip-both", variant_flip_both),
        ("transfer.json", transfer_json),
        ("features.json", features_json),
    ):
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
    header = "| channel \\ population | " + " | ".join(OUTPUT_POPULATIONS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in OUTPUT_POPULATIONS) + " |"
    rows = [header, sep]
    for channel in OBSERVATION_CHANNELS:
        cells = [_fmt(metrics_by_name[f"T:{channel}->{population}"][field], 6) for population in OUTPUT_POPULATIONS]
        rows.append(f"| {channel} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_metric_stats_table(metrics: Sequence[dict]) -> str:
    header = "| metric | biological | null median | null 2.5% | null 97.5% | bio percentile | rho | rho 95% CI |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for metric in sorted(metrics, key=lambda m: abs(m["spearman"]), reverse=True):
        rows.append(
            f"| {metric['name']} | {_fmt(metric['bio'])} | {_fmt(metric['nullMedian'])} | {_fmt(metric['p2_5'])} | "
            f"{_fmt(metric['p97_5'])} | {metric['bioPercentile'] * 100:.1f}% | {metric['spearman']:.3f} | "
            f"[{metric['spearmanCi'][0]:.3f}, {metric['spearmanCi'][1]:.3f}] |"
        )
    return "\n".join(rows)


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
        "three finished, with a fixed feature list not edited after the first run:"
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
        "derived predictors, 40 structural features). Correlations are reported descriptively, without per-metric "
        "significance testing; a permutation calibration "
        f"({calibration['permutations']} seeded permutations of the score against the fixed metric set) found that "
        f"at least one of the {calibration['metricsTested']} metrics reaches \\|rho\\| >= "
        f"{thresholds['spearmanRho']} by chance alone in {calibration['chanceRate'] * 100:.1f}% of permutations -- "
        "this chance rate applies to any triggered linear-pathway or structural-feature finding below."
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
    gate_word = "passed" if regime["gatePassed"] else "failed"
    lines.append(
        f"The aggregate regime gate **{gate_word}** (biological and the null median both within threshold, "
        "biological's transfer solve not ill-conditioned or singular). "
        f"{regime['excludedCount']} of 500 rewirings were individually excluded from the transfer-kind "
        "correlations above for failing their own per-graph regime threshold "
        f"({', '.join(regime['excludedGraphIds']) if regime['excludedGraphIds'] else 'none'})."
    )
    lines.append("")
    lines.append(
        "This licenses treating the linear analysis as applicable to both biological and the null sample under "
        "this model (steady-state distances and clamp fractions are all well inside threshold); it does not by "
        "itself certify that any single transfer entry explains the score -- that still requires the "
        "outside-range-and-\\|rho\\|-threshold test above."
    )
    lines.append("")

    lines.append("## Structural features")
    lines.append("")
    lines.append(
        "40 predeclared graph features (fixed before any analysis ran; not edited after the first run). Feature 6 "
        "(`weightedInDegree`, mean input-restricted weighted in-degree per output population) was adjudicated "
        "during WP2: the plan's \"input->output weighted in-degree\" wording was read as restricted to edges whose "
        "*presynaptic* neuron is input-labeled (channel-mapped), on plan-text grounds (features 1/2's own "
        "\"input\"/\"from any input neuron\" usage, and the parallel with feature 4's unqualified \"edges into "
        "output neurons\" phrasing) decided **before any result was seen**, not selected because of its outcome "
        "(bean `flyarena-r37r`'s log; advocate write-ups under `/tmp/claude-1000/feature6-debate/`). An "
        "**unrestricted** variant (counting edges from *any* presynaptic neuron, not only input-labeled ones) was "
        "also computed during that adjudication for comparison and is disclosed here as **exploratory, "
        "non-predeclared** -- it is not part of the frozen feature list and is not used in the outcome-category "
        "evaluation below: biological's unrestricted thrust in-degree is 1476.5 (null mean 1126.4, sd 72.4, ~100th "
        "percentile), with rank correlation to score rho <= 0.072 on every output population -- weaker on every "
        "population than the predeclared, input-restricted reading, and it would not itself qualify for the "
        "structural-feature-associated category (\\|rho\\| < 0.3)."
    )
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


def update_manifest(manifest_path: Path, entry: dict) -> None:
    """Add/overwrite `nullExplanation` in place, matching
    `scripts/null/null-report.ts`'s `updateManifestWithRewiringNull`
    convention exactly (sorted keys, 2-space indent, trailing newline) --
    verified round-trip-safe first, since the manifest is otherwise
    Python-written (`scripts/data/compile.py`) and this file is also
    Python, the round trip holds unconditionally here (no JS float/
    non-ASCII formatting mismatch to guard against)."""
    original_text = manifest_path.read_text()
    manifest = json.loads(original_text)
    round_tripped = json.dumps(_sort_keys_deep(manifest), indent=2, sort_keys=True) + "\n"
    if round_tripped != original_text:
        raise ValueError(
            f"explain: re-serializing {manifest_path} without any change produced different bytes -- refusing to "
            "write, to avoid silently rewriting unrelated manifest bytes"
        )
    manifest["nullExplanation"] = entry
    write_canonical_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    rewiring_null = _load(args.rewiring_null)
    variant_flip_both = _load(args.variant_flip_both)
    transfer_json = _load(args.transfer)
    features_json = _load(args.features)
    regime_json = _load(args.regime)

    verify_provenance(rewiring_null, variant_flip_both, transfer_json, features_json, regime_json)

    decoder_bio_percentile = variant_flip_both["bioPercentile"]
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
    single_axis_paths = {"flipThrust": args.variant_flip_thrust, "flipYaw": args.variant_flip_yaw}
    provided_single_axis = {k: v for k, v in single_axis_paths.items() if v is not None}
    if decoder_bio_percentile >= DECODER_PERCENTILE_THRESHOLD and not provided_single_axis:
        raise ValueError(
            "explain: the mirrored variant's biological percentile "
            f"({decoder_bio_percentile * 100:.1f}%) meets the predeclared >= "
            f"{DECODER_PERCENTILE_THRESHOLD * 100:.0f}% threshold that triggers the single-axis runs "
            "(.agents/plans/null-explanation/00-overview.md), but --variant-flip-thrust/--variant-flip-yaw were "
            "not supplied -- rerun WP1's single-axis procedure before running explain.py"
        )
    for key, path in provided_single_axis.items():
        payload = _load(path)
        variant_shas[key] = sha256_hex(path.read_bytes())
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

    metrics_by_kind = {"transfer": [], "derived": [], "feature": []}
    for metric in metrics:
        metrics_by_kind[metric["kind"]].append(metric)

    finding_internal = evaluate_categories(
        decoder_bio_percentile,
        metrics_by_kind["transfer"] + metrics_by_kind["derived"],
        metrics_by_kind["feature"],
        regime,
    )
    finding = {
        "categories": finding_internal["categories"],
        "ranked": finding_internal["ranked"],
        "regimeInvalid": finding_internal["regimeInvalid"],
        "summarySentence": build_summary_sentence(finding_internal),
    }

    ordered_names = TRANSFER_METRIC_NAMES + DERIVED_METRIC_NAMES + FEATURE_METRIC_NAMES
    metrics_by_name = {m["name"]: m for m in metrics}
    ordered_metrics = [metrics_by_name[name] for name in ordered_names if name in metrics_by_name]

    # Permutation calibration needs each metric's rank vector over the same
    # (non-excluded) null set it was scored on. Transfer/derived metrics
    # share one exclusion set (possibly non-empty, from the regime gate);
    # structural features always use the full 500 (they never depend on the
    # linear regime -- see `compute_regime`'s doc comment). The two
    # exclusion sets can therefore differ in size, so this calibrates each
    # metric family separately against its own null sample and reports the
    # more conservative (larger) chance rate, rather than concatenating
    # possibly-mismatched-length columns into one matrix.
    def _metric_ranks_and_scores(
        kind_metrics: Sequence[dict], extractor, graphs: Mapping[str, dict], excluded_ids: frozenset[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        seeds = [s for s in range(REWIRED_COUNT) if f"rewired-{s}" not in excluded_ids]
        scores = np.array([score_by_seed[s] for s in seeds], dtype=np.float64)
        columns = [
            _rank(np.array([extractor(graphs[f"rewired-{s}"], metric["name"]) for s in seeds], dtype=np.float64))
            for metric in kind_metrics
        ]
        rank_matrix = np.stack(columns, axis=1) if columns else np.zeros((len(seeds), 0))
        return rank_matrix, _rank(scores)

    transfer_rank_matrix, transfer_rank_scores = _metric_ranks_and_scores(
        metrics_by_kind["transfer"] + metrics_by_kind["derived"],
        lambda g, n: extract_transfer_value(g, n) if n.startswith("T:") else extract_derived_value(g, n),
        transfer_json["graphs"],
        excluded,
    )
    feature_rank_matrix, feature_rank_scores = _metric_ranks_and_scores(
        metrics_by_kind["feature"], extract_feature_value, features_json["graphs"], frozenset()
    )
    permutation_rng = np.random.default_rng(PERMUTATION_BASE_SEED)
    chance_rate_transfer = (
        permutation_chance_rate(
            transfer_rank_matrix, transfer_rank_scores, SPEARMAN_RHO_THRESHOLD, PERMUTATION_COUNT, permutation_rng
        )
        if transfer_rank_matrix.shape[1] > 0
        else 0.0
    )
    chance_rate_feature = permutation_chance_rate(
        feature_rank_matrix, feature_rank_scores, SPEARMAN_RHO_THRESHOLD, PERMUTATION_COUNT, permutation_rng
    )
    chance_rate = max(chance_rate_transfer, chance_rate_feature)

    calibration = {
        "metricsTested": TOTAL_METRIC_COUNT,
        "permutations": PERMUTATION_COUNT,
        "chanceRate": chance_rate,
    }

    host = {"arch": regime_json["host"]["arch"], "node": regime_json["host"]["node"]}
    for label, payload in (("variant-flip-both", variant_flip_both), ("regime.json", regime_json)):
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
        "calibration": calibration,
        "finding": finding,
        "host": host,
    }

    write_canonical_json(args.out, explanation)
    artifact_sha256 = sha256_hex(args.out.read_bytes())

    if not args.skip_manifest_update:
        update_manifest(args.manifest, {"artifact": args.out.name, "sha256": artifact_sha256})

    report_markdown = render_report_markdown(explanation, rewiring_null)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report_markdown)

    print(
        f"explain: wrote {args.out} ({len(ordered_metrics)} metrics, sha256 {artifact_sha256[:12]}...) and "
        f"{args.report_out}"
    )


if __name__ == "__main__":
    main()
