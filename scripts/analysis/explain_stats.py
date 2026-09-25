"""Pure numeric statistics for `scripts/analysis/explain.py` (WP3): empirical
quantiles/rank statistics, Spearman correlation + bootstrap CI, and the
multiple-comparisons permutation calibration.

Extracted from `explain.py` (a thermo-maintainability review finding:
`explain.py` was 1498 lines, 50% past this repo's own "do not let a file
cross 1000 lines without a very strong reason" rule -- the same rule
`scripts/null/null-report.ts` hit and was split for, see
`scripts/null/null-report-trained.ts`'s doc comment). Every function here is
pure (no I/O, no `explain.py`-specific domain knowledge -- metric dicts,
category evaluation, provenance, and CLI orchestration all stay in
`explain.py`), which is exactly the property that let each of them already
have its own direct unit test in `tests_python/test_explain_stats.py`,
independent of `explain.main()`. This module deliberately does not import
anything from `explain.py` or `explain_report.py`, to keep the import
boundary one-directional (`explain.py` imports this module and
`explain_report.py`; neither of those imports back).
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Empirical quantiles / rank statistics -- mirrors
# `scripts/null/null-stats.ts`'s `nullSummary`/`rankStatistics` exactly (the
# same low-tail-floor / high-tail-ceil-minus-one convention), so this
# module's per-metric percentile and range statistics are computed the same
# way as the rest of this study's null artifacts (`docs/null-explanation-
# report.md`: "same tie rule as `scripts/null/null-stats.ts`"). Cross-checked
# against a real `npx tsx` run of `null-stats.ts`'s own `nullSummary`/
# `rankStatistics` by `tests_python/test_null_stats_cross_check.py` (a
# thermo-maintainability review finding: the previous test only asserted
# Python's own output against hand-copied expected values, which could not
# catch either implementation drifting from the other).
# ---------------------------------------------------------------------------


def quantile_index(n: int, p: float) -> int:
    if p <= 0.5:
        return int(np.floor(p * n))
    return min(n - 1, int(np.ceil(p * n)) - 1)


def null_range_summary(values: Sequence[float]) -> dict:
    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)
    if n == 0:
        raise ValueError("explain_stats: null_range_summary requires at least one value")
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
        raise ValueError("explain_stats: rank_statistics requires a non-empty null set")
    k_below = sum(1 for v in null_values if v < bio_value)
    k_equal = sum(1 for v in null_values if v == bio_value)
    return {"kBelow": k_below, "kEqual": k_equal, "bioPercentile": (k_below + 0.5 * k_equal) / n}


def outside_range(bio_value: float, p2_5: float, p97_5: float) -> bool:
    return bio_value < p2_5 or bio_value > p97_5


def format_pct(value: float) -> str:
    """`"37.2th percentile"`-style formatting, with the `0th`/`100th`
    boundary cases spelled out exactly (not `"0.0th"`/`"100.0th"`) -- used by
    both `explain.py`'s `build_summary_sentence` (the decoder-convention
    percentile) and `explain_report.py`'s `render_feature6_disclosure` (the
    feature-6 percentile-flip disclosure), which is why this small formatter
    lives here rather than in either of those two modules: it has no
    dependency on either's domain objects, and duplicating it would be the
    same "two implementations of the same formula" drift risk this review
    pass was asked to look for elsewhere."""
    return f"{value * 100:.1f}th percentile" if value not in (0.0, 1.0) else (
        "0th percentile" if value == 0.0 else "100th percentile"
    )


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
    `build_family_rank_matrix`) and `seed_indices` are the 0-based seeds each
    row corresponds to, indexing into `full_scores` (length `REWIRED_COUNT`).
    One shared permutation of the *full* score vector is drawn per
    iteration, then subset (and re-ranked, since a subset's own average-tie
    ranks differ from the full array's) per family -- both families see the
    same underlying seed-to-score re-pairing, not two independently permuted
    draws -- and a permutation counts as a hit if *any* family reaches the
    threshold, which is the actual "at least one of the ~66 metrics" union
    probability `.agents/plans/null-explanation/00-overview.md` describes.

    Replaces an earlier version that ran two independent single-family
    permutation loops and reported their `max`, mislabeled "the more
    conservative (larger) chance rate" -- for a union of two families, P(A or
    B) >= max(P(A), P(B)), so `max` *understates* the true union rate (a
    dual-review finding, both reviewers independently). The earlier
    single-family `permutation_chance_rate` was removed once nothing in
    production still called it (a thermo-maintainability review finding --
    it was kept alive only by its own test)."""
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
    metrics: Sequence[dict], extractor, graphs: Mapping[str, dict], excluded_ids: frozenset[str], rewired_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(rank_matrix, seed_indices)` for `joint_permutation_chance_rate`:
    `rank_matrix[i, j]` is metric `j`'s average rank at seed `seed_indices[i]`
    (restricted to non-excluded seeds), for every metric in `metrics`.
    `rewired_count` is `explain.REWIRED_COUNT` (passed explicitly rather than
    imported, since this module does not import `explain.py` -- see this
    file's own doc comment on the one-directional import boundary). Raises
    rather than silently letting a `None` extracted value become `NaN`
    through `numpy`/`pandas` -- an all-`NaN`-propagated column would
    otherwise make that column's rho `NaN` in every permutation, and because
    `NaN >= threshold` is `False`, would silently make that column *never*
    register a hit, understating the chance rate without raising anywhere
    (an edge-case-review finding). This study's real `features.json` has no
    `None`s among non-excluded rewired graphs, so this path was previously
    unexercised, not previously safe."""
    seed_indices = np.array(
        [s for s in range(rewired_count) if f"rewired-{s}" not in excluded_ids], dtype=np.int64
    )
    columns = []
    for metric in metrics:
        raw = [extractor(graphs[f"rewired-{s}"], metric["name"]) for s in seed_indices]
        if any(v is None for v in raw):
            raise ValueError(
                f"explain_stats: {metric['name']} has a missing value on a non-excluded rewiring -- the "
                "permutation calibration requires a complete column"
            )
        columns.append(_rank(np.asarray(raw, dtype=np.float64)))
    rank_matrix = np.stack(columns, axis=1) if columns else np.zeros((seed_indices.shape[0], 0))
    return rank_matrix, seed_indices
