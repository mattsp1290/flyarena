"""`.agents/plans/null-explanation/03-explanation-report.md`'s WP3 test list:
"Category logic on synthetic inputs (each category triggered alone, several
together, none); a regime gate failure yields `regime-invalid` and never
`linear-pathway`; an ill-conditioned graph is excluded; the permutation
chance rate is deterministic with a fixed seed; percentile tie rule;
deterministic output."

Covers `scripts/analysis/explain.py`'s own domain: the regime gate,
per-metric building, outcome-category evaluation (including the
`qualifies`/`qualifyingMetrics` full-disclosure machinery), provenance
(graph identity + producer code identity), and the CLI/`main()` end-to-end
determinism gate. Pure statistics (`quantile_index`, `spearman_rho`, the
permutation calibration, ...) are tested in `test_explain_stats.py`, and
Markdown rendering in `test_explain_report.py` -- both split out alongside
`explain.py`'s own module split (see `explain_stats.py`'s doc comment).

Every pure-function test below constructs its own small synthetic input --
none reads the real (gitignored) `training/runs/null/` artifacts, so this
file runs standalone and fast, independent of any Spark run. The single
end-to-end test (`test_main_produces_deterministic_output`) builds a full
synthetic 500-rewiring input set (values only, no real graph algebra) and
runs `explain.main()` twice, to exercise the CLI's actual write path (not
just the pure functions) for the "byte-identical rerun" acceptance gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

import explain  # noqa: E402
import features  # noqa: E402
import transfer  # noqa: E402
import ts_import_graph  # noqa: E402
from features import OBSERVATION_CHANNELS, OUTPUT_POPULATIONS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Regime gate + per-graph exclusion ("an ill-conditioned graph is excluded")
# ---------------------------------------------------------------------------


def _regime_entry(seed_or_label, clamp_fraction, steady_state_distance):
    return {
        "seed": seed_or_label,
        "heldOutSeeds": list(range(30001, 30011)),
        "clampFraction": [clamp_fraction] * 10,
        "steadyStateDistance": [steady_state_distance] * 10,
    }


def _transfer_entry(*, singular=False, ill_conditioned=False, condition_number=4.0, stable=True, discretized_stable=True):
    return {
        "singular": singular,
        "illConditioned": ill_conditioned,
        "conditionNumber": condition_number,
        "stable": stable,
        "spectralAbscissa": 0.1,
        "leakRate": 0.35,
        "discretizedStable": discretized_stable,
        "discretizedSpectralRadius": 0.99,
        "T": None if singular else [[0.1] * 8, [0.1] * 8, [0.1] * 8],
        "turnGain": None if singular else 0.05,
        "approachGain": None if singular else 0.02,
    }


def test_regime_gate_passes_and_no_exclusions_when_everything_is_within_threshold():
    regime_json = {
        "biological": _regime_entry("biological", 0.01, 0.1),
        "rewired": [_regime_entry(i, 0.01, 0.1) for i in range(5)],
    }
    transfer_json = {"graphs": {"biological": _transfer_entry(), **{f"rewired-{i}": _transfer_entry() for i in range(5)}}}
    regime = explain.compute_regime(regime_json, transfer_json)
    assert regime["gatePassed"] is True
    assert regime["excludedGraphIds"] == []
    assert regime["excludedCount"] == 0
    # `T`'s own stability numbers are published, not only the derived
    # pass/fail gate (a round-2 rigor-review finding: WP2's `transfer.py`
    # hands off "report both `stable` and `discretizedStable` side by
    # side" -- the gate alone does not satisfy that).
    assert regime["stability"]["bio"]["spectralAbscissa"] == pytest.approx(0.1)
    assert regime["stability"]["bio"]["discretizedSpectralRadius"] == pytest.approx(0.99)
    assert regime["stability"]["unstableNullCount"] == 0


def test_ill_conditioned_rewiring_is_excluded_from_transfer_correlations():
    regime_json = {
        "biological": _regime_entry("biological", 0.01, 0.1),
        "rewired": [_regime_entry(i, 0.01, 0.1) for i in range(5)],
    }
    transfer_graphs = {"biological": _transfer_entry(), **{f"rewired-{i}": _transfer_entry() for i in range(5)}}
    transfer_graphs["rewired-2"] = _transfer_entry(ill_conditioned=True, condition_number=1e9)
    transfer_json = {"graphs": transfer_graphs}

    regime = explain.compute_regime(regime_json, transfer_json)
    assert regime["excludedGraphIds"] == ["rewired-2"]
    assert regime["excludedCount"] == 1
    # The aggregate gate is about *biological's* own regime/conditioning,
    # not a single excluded rewiring -- it still passes here.
    assert regime["gatePassed"] is True


def test_over_threshold_clamp_fraction_and_steady_state_distance_are_excluded():
    regime_json = {
        "biological": _regime_entry("biological", 0.01, 0.1),
        "rewired": [
            _regime_entry(0, 0.01, 0.1),
            _regime_entry(1, 0.25, 0.1),  # over CLAMP_FRACTION_THRESHOLD
            _regime_entry(2, 0.01, 0.6),  # over STEADY_STATE_DISTANCE_THRESHOLD
        ],
    }
    transfer_json = {"graphs": {"biological": _transfer_entry(), **{f"rewired-{i}": _transfer_entry() for i in range(3)}}}
    regime = explain.compute_regime(regime_json, transfer_json)
    assert regime["excludedGraphIds"] == ["rewired-1", "rewired-2"]
    assert regime["excludedCount"] == 2


def test_biological_own_regime_failure_fails_the_aggregate_gate():
    regime_json = {
        "biological": _regime_entry("biological", 0.01, 0.9),  # bio itself over threshold
        "rewired": [_regime_entry(i, 0.01, 0.1) for i in range(5)],
    }
    transfer_json = {"graphs": {"biological": _transfer_entry(), **{f"rewired-{i}": _transfer_entry() for i in range(5)}}}
    regime = explain.compute_regime(regime_json, transfer_json)
    assert regime["gatePassed"] is False


def test_null_median_clamp_fraction_above_threshold_fails_the_gate():
    # Every rewiring individually over the clamp threshold -- the module's
    # own doc comment says the null-median clamp fraction is gated
    # alongside the null-median distance; this pins that it actually is.
    regime_json = {
        "biological": _regime_entry("biological", 0.01, 0.1),
        "rewired": [_regime_entry(i, 0.9, 0.1) for i in range(5)],
    }
    transfer_json = {"graphs": {"biological": _transfer_entry(), **{f"rewired-{i}": _transfer_entry() for i in range(5)}}}
    regime = explain.compute_regime(regime_json, transfer_json)
    assert regime["gatePassed"] is False
    # Every rewiring is also individually excluded (its own clamp fraction
    # exceeds threshold too), so no transfer/derived metric could ever pick
    # up a spurious candidate from an all-excluded null set.
    assert regime["excludedCount"] == 5


def test_unstable_biological_transfer_fails_the_gate():
    regime_json = {
        "biological": _regime_entry("biological", 0.01, 0.1),
        "rewired": [_regime_entry(i, 0.01, 0.1) for i in range(3)],
    }
    transfer_json = {
        "graphs": {
            "biological": _transfer_entry(stable=False),
            **{f"rewired-{i}": _transfer_entry() for i in range(3)},
        }
    }
    regime = explain.compute_regime(regime_json, transfer_json)
    assert regime["gatePassed"] is False


def test_discretized_unstable_rewiring_is_excluded():
    regime_json = {
        "biological": _regime_entry("biological", 0.01, 0.1),
        "rewired": [_regime_entry(i, 0.01, 0.1) for i in range(3)],
    }
    transfer_graphs = {"biological": _transfer_entry(), **{f"rewired-{i}": _transfer_entry() for i in range(3)}}
    transfer_graphs["rewired-1"] = _transfer_entry(discretized_stable=False)
    transfer_json = {"graphs": transfer_graphs}
    regime = explain.compute_regime(regime_json, transfer_json)
    assert regime["excludedGraphIds"] == ["rewired-1"]
    # Biological's own transfer entry is still stable, so the aggregate gate
    # (which only looks at biological's conditioning) still passes.
    assert regime["gatePassed"] is True


def test_build_metric_drops_excluded_graphs_from_the_null_set(monkeypatch):
    monkeypatch.setattr(explain, "REWIRED_COUNT", 5)
    graphs = {
        "biological": _transfer_entry(),
        **{f"rewired-{i}": _transfer_entry() for i in range(5)},
    }
    # Give rewired-2 a wildly different T so it would obviously skew the
    # null distribution if it were not excluded.
    graphs["rewired-2"]["T"][0][0] = 999.0
    score_by_seed = {i: float(i) for i in range(5)}
    metric_all = explain.build_metric(
        "T:foodBearing->thrust", "transfer", explain.extract_transfer_value, graphs, score_by_seed, frozenset(), 1
    )
    metric_excluded = explain.build_metric(
        "T:foodBearing->thrust",
        "transfer",
        explain.extract_transfer_value,
        graphs,
        score_by_seed,
        frozenset({"rewired-2"}),
        1,
    )
    assert metric_all["p97_5"] != metric_excluded["p97_5"]
    assert metric_excluded["p97_5"] == pytest.approx(0.1)


def test_build_metric_stamps_qualifies_both_gates_from_the_single_predicate(monkeypatch):
    # `build_metric` stamps `qualifiesBothGates` by calling the module-scope
    # `explain.qualifies` on its own just-built dict -- not a second,
    # hand-written boolean. Constructed to obviously qualify (bio far
    # outside a tight null range, |rho| large).
    monkeypatch.setattr(explain, "REWIRED_COUNT", 10)
    graphs = {
        "biological": _transfer_entry(),
        **{f"rewired-{i}": _transfer_entry() for i in range(10)},
    }
    for i in range(10):
        graphs[f"rewired-{i}"]["T"][0][0] = 0.1 + i * 0.001  # tight null range around 0.1
    graphs["biological"]["T"][0][0] = 999.0  # far outside the null range
    score_by_seed = {i: float(i) for i in range(10)}
    metric = explain.build_metric(
        "T:foodBearing->thrust", "transfer", explain.extract_transfer_value, graphs, score_by_seed, frozenset(), 1
    )
    assert metric["qualifiesBothGates"] == explain.qualifies(metric)


# ---------------------------------------------------------------------------
# Outcome category evaluation
# ---------------------------------------------------------------------------


def _metric(name, kind, bio, p2_5, p97_5, spearman):
    return {
        "name": name,
        "kind": kind,
        "bio": bio,
        "nullMedian": (p2_5 + p97_5) / 2,
        "p2_5": p2_5,
        "p97_5": p97_5,
        "bioPercentile": 0.0,
        "spearman": spearman,
        "spearmanCi": [spearman - 0.05, spearman + 0.05],
    }


PASSING_REGIME = {"gatePassed": True, "excludedGraphIds": [], "excludedCount": 0}
FAILING_REGIME = {"gatePassed": False, "excludedGraphIds": [], "excludedCount": 0}


def test_none_of_the_categories_trigger():
    finding = explain.evaluate_categories(0.1, [], [], PASSING_REGIME)
    assert finding["categories"] == []
    assert finding["unexplained"] is True
    assert finding["regimeInvalid"] is False
    sentence = explain.build_summary_sentence(finding)
    assert "No single predeclared factor" in sentence


def test_decoder_convention_triggers_alone():
    finding = explain.evaluate_categories(0.3, [], [], PASSING_REGIME)
    assert finding["categories"] == ["decoderConvention"]
    assert finding["unexplained"] is False
    sentence = explain.build_summary_sentence(finding)
    assert "thrust and yaw" in sentence


def test_linear_pathway_triggers_alone_when_regime_gate_passes():
    qualifying = _metric("T:foodBearing->thrust", "transfer", 0.0, 1.0, 2.0, 0.5)
    finding = explain.evaluate_categories(0.0, [qualifying], [], PASSING_REGIME)
    assert finding["categories"] == ["linearPathway"]
    assert finding["regimeInvalid"] is False
    assert finding["linearDetail"]["name"] == "T:foodBearing->thrust"


def test_structural_feature_triggers_alone():
    qualifying = _metric("weightedInDegree:thrust", "feature", 0.0, 90.0, 160.0, 0.4)
    finding = explain.evaluate_categories(0.0, [], [qualifying], PASSING_REGIME)
    assert finding["categories"] == ["structuralFeature"]
    assert finding["structuralDetail"]["name"] == "weightedInDegree:thrust"


def test_several_categories_trigger_together_and_are_ranked_by_effect_size():
    weak_linear = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, 0.31)
    strong_structural = _metric("weightedInDegree:thrust", "feature", 0.0, 90.0, 160.0, 0.9)
    finding = explain.evaluate_categories(0.5, [weak_linear], [strong_structural], PASSING_REGIME)
    assert set(finding["categories"]) == {"decoderConvention", "linearPathway", "structuralFeature"}
    # Ranked by |effect size| descending: structural (0.9) > decoder (0.5) > linear (0.31).
    assert finding["ranked"] == ["structuralFeature", "decoderConvention", "linearPathway"]


def test_regime_gate_failure_yields_regime_invalid_and_never_linear_pathway():
    qualifying = _metric("T:foodBearing->thrust", "transfer", 0.0, 1.0, 2.0, 0.5)
    finding = explain.evaluate_categories(0.0, [qualifying], [], FAILING_REGIME)
    assert "linearPathway" not in finding["categories"]
    assert finding["regimeInvalid"] is True
    sentence = explain.build_summary_sentence(finding)
    assert "regime-invalid" in sentence


def test_regime_gate_failure_does_not_suppress_structural_feature():
    # Structural features never depend on the linear regime -- a failed
    # regime gate must not block a qualifying structural-feature finding,
    # even though `regimeInvalid` (which describes the *transfer* analysis
    # only) is still correctly `True` whenever the gate itself failed,
    # independent of whether any transfer/derived candidate existed.
    qualifying_structural = _metric("weightedInDegree:thrust", "feature", 0.0, 90.0, 160.0, 0.4)
    finding = explain.evaluate_categories(0.0, [], [qualifying_structural], FAILING_REGIME)
    assert finding["categories"] == ["structuralFeature"]
    assert "linearPathway" not in finding["categories"]
    assert finding["regimeInvalid"] is True


def test_near_threshold_spearman_boundary_does_not_qualify():
    just_under = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, 0.2999999)
    finding = explain.evaluate_categories(0.0, [just_under], [], PASSING_REGIME)
    assert finding["categories"] == []


def test_inside_range_does_not_qualify_even_with_high_rho():
    inside_range = _metric("T:a->b", "transfer", 1.5, 1.0, 2.0, 0.9)
    finding = explain.evaluate_categories(0.0, [inside_range], [], PASSING_REGIME)
    assert finding["categories"] == []


def test_direction_consistent_low_bio_positive_rho():
    # bio below the range, positive rho (the metric rises with score) --
    # consistent with explaining a *low* score.
    metric = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, 0.5)
    assert explain._direction_consistent(metric) is True


def test_direction_consistent_high_bio_negative_rho():
    metric = _metric("T:a->b", "transfer", 3.0, 1.0, 2.0, -0.5)
    assert explain._direction_consistent(metric) is True


def test_direction_inconsistent_low_bio_negative_rho():
    # bio below the range but rho is negative (the metric *falls* with
    # score) -- this metric's sign predicts a HIGHER score for biological,
    # not the observed low one.
    metric = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, -0.5)
    assert explain._direction_consistent(metric) is False


def test_linear_pathway_detail_flags_direction_inconsistency_in_summary():
    inconsistent = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, -0.5)
    finding = explain.evaluate_categories(0.0, [inconsistent], [], PASSING_REGIME)
    assert finding["categories"] == ["linearPathway"]
    assert finding["linearDetail"]["directionConsistent"] is False
    sentence = explain.build_summary_sentence(finding)
    assert "direction-inconsistent" in sentence


# ---------------------------------------------------------------------------
# `qualifies` / `qualifyingMetrics` full disclosure (thermo-methodology
# review, Important 1: the Finding must name EVERY metric that passes both
# predeclared gates, not only the per-category `max(..., key=abs(spearman))`
# exemplar `linearDetail`/`structuralDetail` each surface).
# ---------------------------------------------------------------------------


def test_qualifies_is_module_scope_function():
    qualifying = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, 0.5)
    non_qualifying = _metric("T:c->d", "transfer", 1.5, 1.0, 2.0, 0.9)
    assert explain.qualifies(qualifying) is True
    assert explain.qualifies(non_qualifying) is False


def test_qualifying_metrics_lists_every_metric_passing_both_gates_not_just_category_winners():
    # Mirrors this study's real published numbers exactly: two transfer
    # entries and one structural feature independently pass both predeclared
    # gates, but `evaluate_categories` (via `max(..., key=abs(spearman))`)
    # only ever names the single strongest metric per category as its
    # `linearDetail`/`structuralDetail`.
    strong_linear = _metric("T:rightClearance->thrust", "transfer", 0.0, 90.0, 160.0, 0.467)
    weak_linear = _metric("T:forwardClearance->thrust", "transfer", 0.0, 90.0, 160.0, 0.353)
    structural = _metric("weightedInDegree:thrust", "feature", 0.0, 90.0, 160.0, 0.394)
    non_qualifying = _metric("T:speed->brake", "transfer", 1.5, 1.0, 2.0, 0.9)  # inside range -- excluded

    finding = explain.evaluate_categories(0.0, [strong_linear, weak_linear, non_qualifying], [structural], PASSING_REGIME)

    # The category-level exemplars name only one metric each.
    assert finding["linearDetail"]["name"] == "T:rightClearance->thrust"
    assert finding["structuralDetail"]["name"] == "weightedInDegree:thrust"

    # But `qualifyingMetrics` discloses every metric meeting the bar, sorted
    # by |rho| descending -- including the category runner-up
    # (`T:forwardClearance->thrust`), which the exemplar fields above never
    # surface at all.
    names = [m["name"] for m in finding["qualifyingMetrics"]]
    assert names == ["T:rightClearance->thrust", "weightedInDegree:thrust", "T:forwardClearance->thrust"]
    assert "T:speed->brake" not in names


def test_qualifying_metrics_empty_when_nothing_qualifies():
    finding = explain.evaluate_categories(0.0, [], [], PASSING_REGIME)
    assert finding["qualifyingMetrics"] == []


def test_build_qualifying_metrics_note_names_every_qualifying_metric():
    metrics = [
        _metric("T:rightClearance->thrust", "transfer", 0.0, 90.0, 160.0, 0.467),
        _metric("weightedInDegree:thrust", "feature", 0.0, 90.0, 160.0, 0.394),
        _metric("T:forwardClearance->thrust", "transfer", 0.0, 90.0, 160.0, 0.353),
    ]
    note = explain.build_qualifying_metrics_note(metrics)
    for m in metrics:
        assert m["name"] in note
        assert f"{m['spearman']:.3f}" in note
    assert "3 metrics" in note


def test_build_qualifying_metrics_note_singular_wording_for_one_metric():
    note = explain.build_qualifying_metrics_note([_metric("T:a->b", "transfer", 0.0, 1.0, 2.0, 0.5)])
    assert "1 metric " in note
    assert "1 metrics" not in note


def test_build_qualifying_metrics_note_empty_says_none_qualify():
    note = explain.build_qualifying_metrics_note([])
    assert "No metric" in note


# ---------------------------------------------------------------------------
# Seed coverage
# ---------------------------------------------------------------------------


def test_require_complete_seed_coverage_accepts_full_range(monkeypatch):
    monkeypatch.setattr(explain, "REWIRED_COUNT", 5)
    explain._require_complete_seed_coverage("test", [0, 1, 2, 3, 4])  # no raise


def test_require_complete_seed_coverage_rejects_missing_seed(monkeypatch):
    monkeypatch.setattr(explain, "REWIRED_COUNT", 5)
    with pytest.raises(ValueError, match="does not cover"):
        explain._require_complete_seed_coverage("test", [0, 1, 2, 3])


def test_require_complete_seed_coverage_rejects_duplicate_seed(monkeypatch):
    monkeypatch.setattr(explain, "REWIRED_COUNT", 5)
    with pytest.raises(ValueError, match="does not cover"):
        explain._require_complete_seed_coverage("test", [0, 1, 2, 3, 3])


# ---------------------------------------------------------------------------
# End-to-end determinism (byte-identical rerun on a small synthetic input)
# ---------------------------------------------------------------------------


def _synthetic_transfer_graph(rng: np.random.Generator) -> dict:
    return {
        "T": rng.standard_normal((3, 8)).tolist(),
        "singular": False,
        "illConditioned": False,
        "conditionNumber": 5.0,
        "stable": True,
        "spectralAbscissa": 0.1449,
        "leakRate": 0.35,
        "discretizedStable": True,
        "discretizedSpectralRadius": 0.9932,
        "turnGain": float(rng.standard_normal()),
        "approachGain": float(rng.standard_normal()),
    }


def _synthetic_features_graph(rng: np.random.Generator) -> dict:
    return {
        "pathLengths": {
            f"{channel}->{population}": int(rng.integers(1, 4))
            for channel in OBSERVATION_CHANNELS
            for population in OUTPUT_POPULATIONS
        },
        "meanPathLength": float(rng.uniform(1.0, 3.0)),
        "excitatoryPathCount": {population: int(rng.integers(0, 1000)) for population in OUTPUT_POPULATIONS},
        "inhibitoryPathCount": {population: int(rng.integers(0, 1000)) for population in OUTPUT_POPULATIONS},
        "reciprocity": float(rng.uniform(0.0, 1.0)),
        "weightBalance": {population: float(rng.standard_normal() * 100) for population in OUTPUT_POPULATIONS},
        "twoCycleCount": int(rng.integers(0, 1000)),
        "feedForwardTriangleCount": int(rng.integers(0, 1000)),
        "weightedInDegree": {population: float(rng.uniform(0.0, 200.0)) for population in OUTPUT_POPULATIONS},
    }


@pytest.fixture
def synthetic_inputs(tmp_path, monkeypatch):
    """A full, self-consistent, small (20-rewiring) synthetic input set for
    `explain.main()` -- values only (no real graph algebra), built
    deterministically so the two calls in
    `test_main_produces_deterministic_output` see identical inputs.
    `REWIRED_COUNT` is monkeypatched down from 500 so this stays fast.
    `transfer.json`/`features.json`/`regime.json` each carry a `producer`
    block computed from `explain.current_*_source_sha256()` -- i.e. the
    *real* current working tree's code identity, the same way a real
    `transfer.py`/`features.py`/`regime-check.ts` run would -- so
    `verify_provenance`'s code-identity check passes against this fixture
    exactly as it would against real WP2 output regenerated just now."""
    monkeypatch.setattr(explain, "REWIRED_COUNT", 20)
    rng = np.random.default_rng(1234)

    source_sha = "a" * 64
    rewire_sha = "b" * 64
    host = {"arch": "arm64", "node": "v22.22.3"}
    python_host = {"arch": "aarch64", "python": "3.12.3"}

    rewired = []
    transfer_graphs = {"biological": _synthetic_transfer_graph(rng)}
    features_graphs = {"biological": _synthetic_features_graph(rng)}
    features_exploratory_graphs = {"biological": _synthetic_features_graph(rng)}
    regime_rewired = []
    for seed in range(20):
        score = float(rng.standard_normal())
        rewired.append({"seed": seed, "score": score})
        transfer_graphs[f"rewired-{seed}"] = _synthetic_transfer_graph(rng)
        features_graphs[f"rewired-{seed}"] = _synthetic_features_graph(rng)
        features_exploratory_graphs[f"rewired-{seed}"] = _synthetic_features_graph(rng)
        regime_rewired.append(
            {
                "seed": seed,
                "heldOutSeeds": list(range(30001, 30011)),
                "clampFraction": [float(rng.uniform(0.0, 0.05))] * 10,
                "steadyStateDistance": [float(rng.uniform(0.05, 0.2))] * 10,
            }
        )

    rewiring_null = {
        "sourceGraphSha256": source_sha,
        "rewireSourceSha256": rewire_sha,
        "biological": {"score": -0.5},
        "null": {"mean": 1.0, "std": 0.5},
        "bioPercentile": 0.0,
        "pLow": 0.002,
        "pHigh": 1.0,
        "rewired": rewired,
    }
    variant_flip_both = {
        "sourceGraphSha256": source_sha,
        "rewireSourceSha256": rewire_sha,
        "bioPercentile": 0.0,
        "pLow": 0.002,
        "pHigh": 1.0,
        "null": {"mean": 1.5},
        "biological": {"score": -0.6},
        "host": host,
    }
    transfer_json = {
        "sourceGraphSha256": source_sha,
        "rewireSourceSha256": rewire_sha,
        "graphs": transfer_graphs,
        "producer": {
            "script": "scripts/analysis/transfer.py",
            "sourceSha256": explain.current_transfer_source_sha256(),
            "dependencies": transfer.transfer_producer()["dependencies"],
            "host": python_host,
        },
    }
    features_json = {
        "sourceGraphSha256": source_sha,
        "rewireSourceSha256": rewire_sha,
        "graphs": features_graphs,
        "producer": {
            "script": "scripts/analysis/features.py",
            "sourceSha256": explain.current_features_source_sha256(),
            "dependencies": features.features_producer()["dependencies"],
            "host": python_host,
        },
    }
    features_exploratory_json = {
        "sourceGraphSha256": source_sha,
        "rewireSourceSha256": rewire_sha,
        "graphs": features_exploratory_graphs,
        # Deliberately NOT given a producer block matching current code --
        # exempt from the code-identity check by design (see
        # `explain.verify_provenance`'s doc comment). A stale-looking
        # `producer` here confirms that exemption is real, not an accident
        # of the fixture happening to match.
        "producer": {"script": "scripts/analysis/features.py", "sourceSha256": "stale" * 16, "host": python_host},
    }
    regime_json = {
        "sourceGraphSha256": source_sha,
        "rewireSourceSha256": rewire_sha,
        "biological": {
            "heldOutSeeds": list(range(30001, 30011)),
            "clampFraction": [0.01] * 10,
            "steadyStateDistance": [0.1] * 10,
        },
        "rewired": regime_rewired,
        "host": host,
        "producer": {
            "script": "scripts/null/regime-check.ts",
            "sourceSha256": explain.current_regime_source_sha256(),
            "dependencies": ts_import_graph.collect_repo_relative_dependencies(
                REPO_ROOT / "scripts" / "null" / "regime-check.ts", REPO_ROOT
            ),
        },
    }

    paths = {}
    for name, payload in (
        ("rewiring-null.json", rewiring_null),
        ("variant-flip-both.json", variant_flip_both),
        ("transfer.json", transfer_json),
        ("features.json", features_json),
        ("features-exploratory-unrestricted.json", features_exploratory_json),
        ("regime.json", regime_json),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        paths[name] = path
    return paths


def test_main_produces_deterministic_output(tmp_path, synthetic_inputs):
    def run(out_name: str, report_name: str) -> tuple[Path, Path]:
        out_path = tmp_path / out_name
        report_path = tmp_path / report_name
        explain.main(
            [
                "--rewiring-null",
                str(synthetic_inputs["rewiring-null.json"]),
                "--variant-flip-both",
                str(synthetic_inputs["variant-flip-both.json"]),
                "--transfer",
                str(synthetic_inputs["transfer.json"]),
                "--features",
                str(synthetic_inputs["features.json"]),
                "--features-exploratory-unrestricted",
                str(synthetic_inputs["features-exploratory-unrestricted.json"]),
                "--regime",
                str(synthetic_inputs["regime.json"]),
                "--out",
                str(out_path),
                "--report-out",
                str(report_path),
                "--skip-manifest-update",
            ]
        )
        return out_path, report_path

    out_a, report_a = run("out-a.json", "report-a.md")
    out_b, report_b = run("out-b.json", "report-b.md")

    assert out_a.read_bytes() == out_b.read_bytes()
    assert report_a.read_bytes() == report_b.read_bytes()

    payload = json.loads(out_a.read_bytes())
    assert payload["version"] == 1
    assert len(payload["metrics"]) <= explain.TOTAL_METRIC_COUNT
    assert payload["host"] == {"arch": "arm64", "node": "v22.22.3"}
    assert "finding" in payload
    assert "summarySentence" in payload["finding"]
    assert "definitionSensitive" in payload["finding"]
    assert payload["calibration"]["metricsTested"] == len(payload["metrics"])
    assert payload["calibration"]["chanceHits"] == round(payload["calibration"]["chanceRate"] * payload["calibration"]["permutations"])
    assert "exploratory" in payload
    assert len(payload["exploratory"]["featureSixUnrestricted"]["metrics"]) == 3

    # `qualifyingMetrics`/`qualifyingMetricsNote`: the full-disclosure set
    # must exactly match an independent recomputation of `qualifies` over
    # every published metric (not just what happened to trigger a category).
    expected_qualifying_names = {m["name"] for m in payload["metrics"] if explain.qualifies(m)}
    actual_qualifying_names = {m["name"] for m in payload["finding"]["qualifyingMetrics"]}
    assert actual_qualifying_names == expected_qualifying_names
    for m in payload["finding"]["qualifyingMetrics"]:
        assert m["name"] in payload["finding"]["qualifyingMetricsNote"]

    # Every published metric carries the `qualifiesBothGates` stamp, and it
    # agrees with an independent recomputation of `explain.qualifies`.
    for m in payload["metrics"]:
        assert m["qualifiesBothGates"] == explain.qualifies(m)

    # Producer code-identity blocks are recorded in the published artifact
    # (thermo-methodology review, Important 2's "record the input producer
    # shas in the published artifact").
    producers = payload["sources"]["producers"]
    assert producers["transfer"]["sourceSha256"] == explain.current_transfer_source_sha256()
    assert producers["features"]["sourceSha256"] == explain.current_features_source_sha256()
    assert producers["regime"]["sourceSha256"] == explain.current_regime_source_sha256()

    # The report's "Question" paragraph lead clause (a thermo-methodology
    # review finding, C3): `synthetic_inputs`' `rewiring-null.json` has
    # `bioPercentile: 0.0` over 20 rewirings (`REWIRED_COUNT` monkeypatched
    # down from the real study's 500), so the fixed "below all 500"
    # phrasing this bug hardcoded would already have been wrong here even
    # before considering `random-bridge` -- confirms the derived clause
    # reads the fixture's real null size, not a hardcoded constant.
    report_text = report_a.read_text()
    assert "scoring below all 20 degree-preserving rewirings" in report_text
    assert "scoring below all 500 degree-preserving rewirings" not in report_text


def test_report_question_paragraph_derives_from_nonzero_bio_percentile(tmp_path, synthetic_inputs):
    # `random-bridge`'s real, committed regression: biological at the 24.8th
    # percentile (not the 0th), which the old hardcoded "scoring below all
    # 500 ... rewirings" clause stated as fact regardless. Mutate the
    # fixture's `rewiring-null.json` to a non-zero percentile and confirm
    # the rendered report no longer claims "below all".
    rewiring_null_path = synthetic_inputs["rewiring-null.json"]
    rewiring_null = json.loads(rewiring_null_path.read_text())
    rewiring_null["bioPercentile"] = 0.248
    rewiring_null_path.write_text(json.dumps(rewiring_null))

    out_path = tmp_path / "out.json"
    report_path = tmp_path / "report.md"
    explain.main(
        [
            "--rewiring-null",
            str(rewiring_null_path),
            "--variant-flip-both",
            str(synthetic_inputs["variant-flip-both.json"]),
            "--transfer",
            str(synthetic_inputs["transfer.json"]),
            "--features",
            str(synthetic_inputs["features.json"]),
            "--features-exploratory-unrestricted",
            str(synthetic_inputs["features-exploratory-unrestricted.json"]),
            "--regime",
            str(synthetic_inputs["regime.json"]),
            "--out",
            str(out_path),
            "--report-out",
            str(report_path),
            "--skip-manifest-update",
        ]
    )
    report_text = report_path.read_text()
    assert "scoring at the 24.8th percentile of 20 degree-preserving rewirings" in report_text
    assert "below all" not in report_text
    # The parenthetical percentile figure quoted later in the same
    # sentence must agree with (not contradict) the lead clause.
    assert "24.8th percentile, `p_low" in report_text


# ---------------------------------------------------------------------------
# Single-axis variant validation (I7: both-or-neither, trigger-consistent)
# ---------------------------------------------------------------------------


def _run_main(tmp_path, synthetic_inputs, extra_args: list[str]) -> None:
    explain.main(
        [
            "--rewiring-null",
            str(synthetic_inputs["rewiring-null.json"]),
            "--variant-flip-both",
            str(synthetic_inputs["variant-flip-both.json"]),
            "--transfer",
            str(synthetic_inputs["transfer.json"]),
            "--features",
            str(synthetic_inputs["features.json"]),
            "--features-exploratory-unrestricted",
            str(synthetic_inputs["features-exploratory-unrestricted.json"]),
            "--regime",
            str(synthetic_inputs["regime.json"]),
            "--out",
            str(tmp_path / "out.json"),
            "--report-out",
            str(tmp_path / "report.md"),
            "--skip-manifest-update",
            *extra_args,
        ]
    )


def test_single_axis_supplied_when_not_triggered_raises(tmp_path, synthetic_inputs):
    # The synthetic fixture's mirrored variant has bioPercentile 0.0, well
    # under the predeclared 25% trigger -- supplying single-axis files here
    # is exactly the "ran variants the predeclared rule didn't call for"
    # case that must be rejected, not silently accepted.
    single_axis = json.loads(synthetic_inputs["variant-flip-both.json"].read_text())
    single_axis_path = tmp_path / "flip-thrust.json"
    single_axis_path.write_text(json.dumps(single_axis))
    with pytest.raises(ValueError, match="predeclared rule did not call"):
        _run_main(tmp_path, synthetic_inputs, ["--variant-flip-thrust", str(single_axis_path)])


def test_single_axis_partial_pair_when_triggered_raises(tmp_path, synthetic_inputs, monkeypatch):
    # Rewrite the mirrored variant to trigger the single-axis condition
    # (bioPercentile >= 0.25), then supply only one of the two required
    # single-axis files.
    flip_both = json.loads(synthetic_inputs["variant-flip-both.json"].read_text())
    flip_both["bioPercentile"] = 0.5
    synthetic_inputs["variant-flip-both.json"].write_text(json.dumps(flip_both))
    single_axis_path = tmp_path / "flip-thrust.json"
    single_axis_path.write_text(json.dumps(flip_both))
    with pytest.raises(ValueError, match="both --variant-flip-thrust and --variant-flip-yaw are required"):
        _run_main(tmp_path, synthetic_inputs, ["--variant-flip-thrust", str(single_axis_path)])


def test_single_axis_both_supplied_when_triggered_succeeds(tmp_path, synthetic_inputs):
    flip_both = json.loads(synthetic_inputs["variant-flip-both.json"].read_text())
    flip_both["bioPercentile"] = 0.5
    synthetic_inputs["variant-flip-both.json"].write_text(json.dumps(flip_both))
    thrust_path = tmp_path / "flip-thrust.json"
    yaw_path = tmp_path / "flip-yaw.json"
    thrust_path.write_text(json.dumps(flip_both))
    yaw_path.write_text(json.dumps(flip_both))
    _run_main(tmp_path, synthetic_inputs, ["--variant-flip-thrust", str(thrust_path), "--variant-flip-yaw", str(yaw_path)])
    payload = json.loads((tmp_path / "out.json").read_bytes())
    assert "flipThrust" in payload["variants"]
    assert "flipYaw" in payload["variants"]
    assert "singleAxisSkipped" not in payload["variants"]


def test_single_axis_wrong_provenance_is_rejected(tmp_path, synthetic_inputs):
    flip_both = json.loads(synthetic_inputs["variant-flip-both.json"].read_text())
    flip_both["bioPercentile"] = 0.5
    synthetic_inputs["variant-flip-both.json"].write_text(json.dumps(flip_both))
    bad = dict(flip_both)
    bad["sourceGraphSha256"] = "c" * 64  # does not match rewiring-null-v1.json's source
    thrust_path = tmp_path / "flip-thrust.json"
    yaw_path = tmp_path / "flip-yaw.json"
    thrust_path.write_text(json.dumps(bad))
    yaw_path.write_text(json.dumps(flip_both))
    with pytest.raises(ValueError, match="does not match"):
        _run_main(tmp_path, synthetic_inputs, ["--variant-flip-thrust", str(thrust_path), "--variant-flip-yaw", str(yaw_path)])


def test_regime_seed_gap_is_rejected(tmp_path, synthetic_inputs):
    regime = json.loads(synthetic_inputs["regime.json"].read_text())
    regime["rewired"].pop()  # drop one seed -- coverage is now incomplete
    synthetic_inputs["regime.json"].write_text(json.dumps(regime))
    with pytest.raises(ValueError, match="does not cover rewired seeds"):
        _run_main(tmp_path, synthetic_inputs, [])


# ---------------------------------------------------------------------------
# Provenance: producer code identity (thermo-methodology review, Important
# 2 -- `verify_provenance` must REFUSE any input whose recorded
# `producer.sourceSha256` doesn't match the current source).
# ---------------------------------------------------------------------------


def test_transfer_json_stale_producer_sha_is_rejected(tmp_path, synthetic_inputs):
    transfer_json = json.loads(synthetic_inputs["transfer.json"].read_text())
    transfer_json["producer"]["sourceSha256"] = "0" * 64  # does not match current transfer.py
    synthetic_inputs["transfer.json"].write_text(json.dumps(transfer_json))
    with pytest.raises(ValueError, match="code-identity provenance mismatch"):
        _run_main(tmp_path, synthetic_inputs, [])


def test_features_json_stale_producer_sha_is_rejected(tmp_path, synthetic_inputs):
    features_json = json.loads(synthetic_inputs["features.json"].read_text())
    features_json["producer"]["sourceSha256"] = "0" * 64
    synthetic_inputs["features.json"].write_text(json.dumps(features_json))
    with pytest.raises(ValueError, match="code-identity provenance mismatch"):
        _run_main(tmp_path, synthetic_inputs, [])


def test_regime_json_stale_producer_sha_is_rejected(tmp_path, synthetic_inputs):
    regime_json = json.loads(synthetic_inputs["regime.json"].read_text())
    regime_json["producer"]["sourceSha256"] = "0" * 64
    synthetic_inputs["regime.json"].write_text(json.dumps(regime_json))
    with pytest.raises(ValueError, match="code-identity provenance mismatch"):
        _run_main(tmp_path, synthetic_inputs, [])


def test_transfer_json_missing_producer_block_is_rejected(tmp_path, synthetic_inputs):
    transfer_json = json.loads(synthetic_inputs["transfer.json"].read_text())
    del transfer_json["producer"]
    synthetic_inputs["transfer.json"].write_text(json.dumps(transfer_json))
    with pytest.raises(ValueError, match="has no 'producer' block"):
        _run_main(tmp_path, synthetic_inputs, [])


def test_features_exploratory_unrestricted_is_exempt_from_producer_check(tmp_path, synthetic_inputs):
    # The fixture already gives this input a deliberately-stale-looking
    # producer sha (see `synthetic_inputs`'s doc comment) -- confirms
    # `verify_provenance` really does skip the code-identity check for it,
    # not merely that this test forgot to break it.
    exploratory = json.loads(synthetic_inputs["features-exploratory-unrestricted.json"].read_text())
    assert exploratory["producer"]["sourceSha256"] == "stale" * 16
    _run_main(tmp_path, synthetic_inputs, [])  # must not raise


def test_transfer_features_producer_host_arch_mismatch_is_rejected(tmp_path, synthetic_inputs):
    transfer_json = json.loads(synthetic_inputs["transfer.json"].read_text())
    transfer_json["producer"]["host"] = {"arch": "x86_64", "python": "3.12.3"}
    synthetic_inputs["transfer.json"].write_text(json.dumps(transfer_json))
    with pytest.raises(ValueError, match="producer host arch"):
        _run_main(tmp_path, synthetic_inputs, [])


# ---------------------------------------------------------------------------
# --selection-mode (.agents/plans/selection-robustness/02-per-selection-chain.md
# WP2): --features-exploratory-unrestricted becomes optional, and the
# public/docs refusal guard applies to --out/--report-out/--manifest.
# ---------------------------------------------------------------------------


def _run_main_selection_mode(tmp_path, synthetic_inputs, out_path, report_path, extra_args=()):
    explain.main(
        [
            "--selection-mode",
            "--rewiring-null",
            str(synthetic_inputs["rewiring-null.json"]),
            "--variant-flip-both",
            str(synthetic_inputs["variant-flip-both.json"]),
            "--transfer",
            str(synthetic_inputs["transfer.json"]),
            "--features",
            str(synthetic_inputs["features.json"]),
            "--regime",
            str(synthetic_inputs["regime.json"]),
            "--out",
            str(out_path),
            "--report-out",
            str(report_path),
            "--skip-manifest-update",
            *extra_args,
        ]
    )


def test_selection_mode_omits_exploratory_and_records_a_reason(tmp_path, synthetic_inputs):
    out_path = tmp_path / "out.json"
    _run_main_selection_mode(tmp_path, synthetic_inputs, out_path, tmp_path / "report.md")
    payload = json.loads(out_path.read_bytes())
    assert payload["exploratory"] is None
    assert "no per-selection counterpart" in payload["exploratoryOmittedReason"]
    assert payload["finding"]["definitionSensitive"] is None


def test_selection_mode_runs_twice_byte_identical_with_exploratory_omitted(tmp_path, synthetic_inputs):
    out_a, report_a = tmp_path / "out-a.json", tmp_path / "report-a.md"
    out_b, report_b = tmp_path / "out-b.json", tmp_path / "report-b.md"
    _run_main_selection_mode(tmp_path, synthetic_inputs, out_a, report_a)
    _run_main_selection_mode(tmp_path, synthetic_inputs, out_b, report_b)
    assert out_a.read_bytes() == out_b.read_bytes()
    assert report_a.read_bytes() == report_b.read_bytes()


def test_without_selection_mode_features_exploratory_unrestricted_is_still_required(tmp_path, synthetic_inputs):
    with pytest.raises(SystemExit):
        explain.main(
            [
                "--rewiring-null",
                str(synthetic_inputs["rewiring-null.json"]),
                "--variant-flip-both",
                str(synthetic_inputs["variant-flip-both.json"]),
                "--transfer",
                str(synthetic_inputs["transfer.json"]),
                "--features",
                str(synthetic_inputs["features.json"]),
                "--regime",
                str(synthetic_inputs["regime.json"]),
                "--out",
                str(tmp_path / "out.json"),
                "--report-out",
                str(tmp_path / "report.md"),
                "--skip-manifest-update",
            ]
        )


def test_selection_mode_still_requires_exploratory_json_shas_for_verify_provenance_direct_call(synthetic_inputs):
    rewiring_null = json.loads(synthetic_inputs["rewiring-null.json"].read_text())
    variant_flip_both = json.loads(synthetic_inputs["variant-flip-both.json"].read_text())
    transfer_json = json.loads(synthetic_inputs["transfer.json"].read_text())
    features_json = json.loads(synthetic_inputs["features.json"].read_text())
    regime_json = json.loads(synthetic_inputs["regime.json"].read_text())
    # Omitting features_exploratory_json without selection_mode=True must be
    # refused even calling verify_provenance directly (not only via the CLI's
    # own argparse gate) -- explain_provenance.py's own defense in depth.
    with pytest.raises(ValueError, match="required unless selection_mode"):
        explain.verify_provenance(
            rewiring_null, {"flipBoth": variant_flip_both}, transfer_json, features_json, None, regime_json
        )
    # selection_mode=True accepts it (no producer/graph-identity checks run
    # against a None exploratory input).
    explain.verify_provenance(
        rewiring_null,
        {"flipBoth": variant_flip_both},
        transfer_json,
        features_json,
        None,
        regime_json,
        selection_mode=True,
    )


def test_selection_mode_guard_refuses_public_data_out_for_non_shipped_graph(tmp_path, synthetic_inputs, monkeypatch):
    # A fake "shipped" tree with a manifest describing a DIFFERENT graph sha
    # than the fixture's own ("a" * 64) -- exercises the refusal without
    # touching the real repo's public/data or docs directories.
    fake_public_data = tmp_path / "shipped" / "public" / "data"
    fake_docs = tmp_path / "shipped" / "docs"
    fake_public_data.mkdir(parents=True)
    fake_docs.mkdir(parents=True)
    (fake_public_data / "malecns-arena-v1.manifest.json").write_text(json.dumps({"binarySha256": "f" * 64}))
    monkeypatch.setattr(explain, "PUBLIC_DATA_DIR", fake_public_data)
    monkeypatch.setattr(explain, "DOCS_DIR", fake_docs)

    with pytest.raises(ValueError, match="resolves under"):
        _run_main_selection_mode(tmp_path, synthetic_inputs, fake_public_data / "null-explanation-selection-larger.json", tmp_path / "report.md")
    assert not (fake_public_data / "null-explanation-selection-larger.json").exists()

    with pytest.raises(ValueError, match="resolves under"):
        _run_main_selection_mode(tmp_path, synthetic_inputs, tmp_path / "out.json", fake_docs / "selection-larger-report.md")


def test_selection_mode_guard_covers_manifest_too(tmp_path, synthetic_inputs, monkeypatch):
    # Every other selection-mode guard test passes --skip-manifest-update
    # (a dual-review gap flagged directly), so `guard(args.manifest,
    # "--manifest")` (explain.py, right after the --out/--report-out guard
    # calls) was never actually exercised. The guard runs before any
    # manifest file is read (explain.main() calls it immediately after
    # loading rewiring_null, well before check_manifest_round_trips), so
    # --manifest need not even point at a real file for this to prove the
    # guard fires.
    fake_public_data = tmp_path / "shipped" / "public" / "data"
    fake_docs = tmp_path / "shipped" / "docs"
    fake_public_data.mkdir(parents=True)
    fake_docs.mkdir(parents=True)
    (fake_public_data / "malecns-arena-v1.manifest.json").write_text(json.dumps({"binarySha256": "f" * 64}))
    monkeypatch.setattr(explain, "PUBLIC_DATA_DIR", fake_public_data)
    monkeypatch.setattr(explain, "DOCS_DIR", fake_docs)

    manifest_path = fake_public_data / "malecns-arena-selection-larger.manifest.json"
    with pytest.raises(ValueError, match="resolves under"):
        explain.main(
            [
                "--selection-mode",
                "--rewiring-null",
                str(synthetic_inputs["rewiring-null.json"]),
                "--variant-flip-both",
                str(synthetic_inputs["variant-flip-both.json"]),
                "--transfer",
                str(synthetic_inputs["transfer.json"]),
                "--features",
                str(synthetic_inputs["features.json"]),
                "--regime",
                str(synthetic_inputs["regime.json"]),
                "--out",
                str(tmp_path / "out.json"),
                "--report-out",
                str(tmp_path / "report.md"),
                "--manifest",
                str(manifest_path),
            ]
        )
    assert not manifest_path.exists()


def test_selection_mode_guard_refuses_public_data_out_even_when_graph_sha_matches_shipped(
    tmp_path, synthetic_inputs, monkeypatch
):
    # A dual-review finding: an earlier version of the guard returned early
    # whenever the scored graph's sha happened to match the shipped one,
    # even in --selection-mode -- so `explain.py --selection-mode` run
    # against the shipped graph, with default --out/--report-out, would
    # silently overwrite the shipped null-explanation-v1.json with
    # `exploratory: null`/`finding.definitionSensitive: null`, which the
    # browser's shape validator rejects. Selection mode must refuse
    # public/docs unconditionally -- it is never the canonical publish path.
    fake_public_data = tmp_path / "shipped" / "public" / "data"
    fake_docs = tmp_path / "shipped" / "docs"
    fake_public_data.mkdir(parents=True)
    fake_docs.mkdir(parents=True)
    (fake_public_data / "malecns-arena-v1.manifest.json").write_text(json.dumps({"binarySha256": "a" * 64}))
    monkeypatch.setattr(explain, "PUBLIC_DATA_DIR", fake_public_data)
    monkeypatch.setattr(explain, "DOCS_DIR", fake_docs)

    out_path = fake_public_data / "null-explanation-v1.json"
    with pytest.raises(ValueError, match="resolves under"):
        _run_main_selection_mode(tmp_path, synthetic_inputs, out_path, tmp_path / "report.md")
    assert not out_path.exists()


def test_non_selection_mode_allows_public_data_out_when_graph_sha_matches_shipped(
    tmp_path, synthetic_inputs, monkeypatch
):
    # The sha-based allowance is for the ordinary (non-selection-mode)
    # republish path only, where writing the shipped graph's own artifacts
    # back to their shipped paths is the entire point (the normal
    # `npm run` … / `uv run python scripts/analysis/explain.py` flow, with
    # no --selection-mode, republishing null-explanation-v1.json).
    fake_public_data = tmp_path / "shipped" / "public" / "data"
    fake_docs = tmp_path / "shipped" / "docs"
    fake_public_data.mkdir(parents=True)
    fake_docs.mkdir(parents=True)
    (fake_public_data / "malecns-arena-v1.manifest.json").write_text(json.dumps({"binarySha256": "a" * 64}))
    monkeypatch.setattr(explain, "PUBLIC_DATA_DIR", fake_public_data)
    monkeypatch.setattr(explain, "DOCS_DIR", fake_docs)

    out_path = fake_public_data / "null-explanation-v1.json"
    # Exercised directly against the guard (the ordinary, non-selection-mode
    # call site's exact arguments) -- a full `explain.main()` run would also
    # need a real `--manifest` round-trip fixture, which is incidental to
    # what this test is checking.
    explain.explain_selection_mode.guard_selection_scratch_target(
        out_path,
        "--out",
        "a" * 64,
        public_data_dir=fake_public_data,
        docs_dir=fake_docs,
        selection_mode=False,
    )  # must not raise


def test_guard_selection_scratch_target_fails_safe_on_a_corrupt_shipped_manifest(tmp_path) -> None:
    # A thermo-methodology review finding: an earlier version of this guard
    # only handled a MISSING shipped manifest (`.exists()`) -- a
    # present-but-corrupt one (e.g. a half-written `git merge` conflict or
    # an interrupted editor save) raised an uncaught `JSONDecodeError`
    # instead of the documented "fails safe: unreadable/missing/corrupt ...
    # treated as 'graph not shipped'" refusal. This never crosses a write
    # boundary either way (both the crash and the graceful refusal happen
    # before anything is written), but the graceful refusal is what the doc
    # comment promises, so it's what must actually happen.
    fake_public_data = tmp_path / "shipped" / "public" / "data"
    fake_docs = tmp_path / "shipped" / "docs"
    fake_public_data.mkdir(parents=True)
    fake_docs.mkdir(parents=True)
    (fake_public_data / "malecns-arena-v1.manifest.json").write_text("{not valid json")

    out_path = fake_public_data / "null-explanation-selection-larger.json"
    with pytest.raises(ValueError, match="resolves under"):
        explain.explain_selection_mode.guard_selection_scratch_target(
            out_path,
            "--out",
            "a" * 64,
            public_data_dir=fake_public_data,
            docs_dir=fake_docs,
            selection_mode=False,
        )
    assert not out_path.exists()
