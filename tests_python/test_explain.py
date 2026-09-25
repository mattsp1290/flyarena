"""`.agents/plans/null-explanation/03-explanation-report.md`'s WP3 test list:
"Category logic on synthetic inputs (each category triggered alone, several
together, none); a regime gate failure yields `regime-invalid` and never
`linear-pathway`; an ill-conditioned graph is excluded; the permutation
chance rate is deterministic with a fixed seed; percentile tie rule;
deterministic output."

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
from features import OBSERVATION_CHANNELS, OUTPUT_POPULATIONS  # noqa: E402


# ---------------------------------------------------------------------------
# Empirical quantiles / rank statistics ("percentile tie rule")
# ---------------------------------------------------------------------------


def test_quantile_index_matches_null_stats_ts_convention():
    # Mirrors `scripts/null/null-stats.ts`'s own `quantileIndex` doc comment
    # examples: n=500 -> floor(0.025*500)=12, min(499, ceil(0.975*500)-1)=487.
    assert explain.quantile_index(500, 0.025) == 12
    assert explain.quantile_index(500, 0.975) == 487
    # p == 0.5 uses the low-tail floor branch.
    assert explain.quantile_index(500, 0.5) == 250
    # High-tail branch never exceeds n - 1.
    assert explain.quantile_index(4, 0.975) == 3


def test_null_range_summary_median_and_quantiles():
    values = [float(v) for v in range(1, 501)]  # 1..500, evenly spaced
    summary = explain.null_range_summary(values)
    assert summary["median"] == pytest.approx(250.5)
    assert summary["p2_5"] == pytest.approx(values[12])
    assert summary["p97_5"] == pytest.approx(values[487])


def test_rank_statistics_tie_rule():
    # 3 below, 2 equal, 5 total -> bioPercentile = (3 + 0.5*2)/5 = 0.8.
    null_values = [1.0, 2.0, 3.0, 5.0, 5.0]
    stats = explain.rank_statistics(null_values, 5.0)
    assert stats["kBelow"] == 3
    assert stats["kEqual"] == 2
    assert stats["bioPercentile"] == pytest.approx(0.8)


def test_outside_range():
    assert explain.outside_range(0.0, 1.0, 2.0) is True
    assert explain.outside_range(3.0, 1.0, 2.0) is True
    assert explain.outside_range(1.5, 1.0, 2.0) is False
    assert explain.outside_range(1.0, 1.0, 2.0) is False  # boundary is inside


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------


def test_spearman_rho_perfect_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert explain.spearman_rho(x, y) == pytest.approx(1.0)
    assert explain.spearman_rho(x, y[::-1]) == pytest.approx(-1.0)


def test_spearman_rho_degenerate_constant_input_is_zero_not_nan():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explain.spearman_rho(x, y) == 0.0


def test_spearman_rho_ties_use_average_rank():
    # x has a tie at rank (1,2) -> average rank 1.5 each; verified against
    # a hand-computed Spearman value (equivalent to scipy.stats.spearmanr
    # with tie correction).
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    rho = explain.spearman_rho(x, y)
    assert rho == pytest.approx(0.9486832980505138, rel=1e-9)


# ---------------------------------------------------------------------------
# Deterministic RNG / bootstrap / permutation calibration
# ---------------------------------------------------------------------------


def test_metric_rng_is_deterministic_and_label_dependent():
    rng_a1 = explain.metric_rng(42, "foo")
    rng_a2 = explain.metric_rng(42, "foo")
    rng_b = explain.metric_rng(42, "bar")
    assert rng_a1.integers(0, 1_000_000, size=5).tolist() == rng_a2.integers(0, 1_000_000, size=5).tolist()
    # A different label gives a different stream (not a hard mathematical
    # guarantee, but true with overwhelming probability for a sha256-derived
    # seed -- a collision here would indicate a bug, not bad luck).
    rng_a3 = explain.metric_rng(42, "foo")
    assert rng_a3.integers(0, 1_000_000, size=5).tolist() != rng_b.integers(0, 1_000_000, size=5).tolist()


def test_bootstrap_spearman_ci_deterministic_with_fixed_seed():
    rank_x = np.arange(50.0)
    rank_y = np.arange(50.0) + np.array([0.0, 1.0] * 25)
    ci_a = explain.bootstrap_spearman_ci(rank_x, rank_y, 2_000, explain.metric_rng(7, "m"))
    ci_b = explain.bootstrap_spearman_ci(rank_x, rank_y, 2_000, explain.metric_rng(7, "m"))
    assert ci_a == ci_b


def test_permutation_chance_rate_deterministic_with_fixed_seed():
    rng = np.random.default_rng(0)
    n = 200
    score = rng.standard_normal(n)
    # One metric strongly correlated with score, one pure noise.
    metric_matrix = np.stack([score + rng.standard_normal(n) * 0.01, rng.standard_normal(n)], axis=1)
    rank_matrix = np.stack([explain._rank(metric_matrix[:, i]) for i in range(2)], axis=1)
    rank_score = explain._rank(score)

    rate_a = explain.permutation_chance_rate(rank_matrix, rank_score, 0.3, 500, np.random.default_rng(123))
    rate_b = explain.permutation_chance_rate(rank_matrix, rank_score, 0.3, 500, np.random.default_rng(123))
    assert rate_a == rate_b
    # Under permutation the true association is broken, so a high-|rho|
    # column should be rare, not near-certain.
    assert 0.0 <= rate_a <= 0.2


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


def _transfer_entry(*, singular=False, ill_conditioned=False, condition_number=4.0):
    return {
        "singular": singular,
        "illConditioned": ill_conditioned,
        "conditionNumber": condition_number,
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
    # regime gate must not block a qualifying structural-feature finding.
    qualifying_structural = _metric("weightedInDegree:thrust", "feature", 0.0, 90.0, 160.0, 0.4)
    finding = explain.evaluate_categories(0.0, [], [qualifying_structural], FAILING_REGIME)
    assert finding["categories"] == ["structuralFeature"]
    assert finding["regimeInvalid"] is False  # no linear candidate existed to be invalidated


def test_near_threshold_spearman_boundary_does_not_qualify():
    just_under = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, 0.2999999)
    finding = explain.evaluate_categories(0.0, [just_under], [], PASSING_REGIME)
    assert finding["categories"] == []


def test_inside_range_does_not_qualify_even_with_high_rho():
    inside_range = _metric("T:a->b", "transfer", 1.5, 1.0, 2.0, 0.9)
    finding = explain.evaluate_categories(0.0, [inside_range], [], PASSING_REGIME)
    assert finding["categories"] == []


# ---------------------------------------------------------------------------
# End-to-end determinism (byte-identical rerun on a small synthetic input)
# ---------------------------------------------------------------------------


def _synthetic_transfer_graph(rng: np.random.Generator) -> dict:
    return {
        "T": rng.standard_normal((3, 8)).tolist(),
        "singular": False,
        "illConditioned": False,
        "conditionNumber": 5.0,
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
    `REWIRED_COUNT` is monkeypatched down from 500 so this stays fast."""
    monkeypatch.setattr(explain, "REWIRED_COUNT", 20)
    rng = np.random.default_rng(1234)

    source_sha = "a" * 64
    rewire_sha = "b" * 64
    host = {"arch": "arm64", "node": "v22.22.3"}

    rewired = []
    transfer_graphs = {"biological": _synthetic_transfer_graph(rng)}
    features_graphs = {"biological": _synthetic_features_graph(rng)}
    regime_rewired = []
    for seed in range(20):
        score = float(rng.standard_normal())
        rewired.append({"seed": seed, "score": score})
        transfer_graphs[f"rewired-{seed}"] = _synthetic_transfer_graph(rng)
        features_graphs[f"rewired-{seed}"] = _synthetic_features_graph(rng)
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
    }
    features_json = {
        "sourceGraphSha256": source_sha,
        "rewireSourceSha256": rewire_sha,
        "graphs": features_graphs,
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
    }

    paths = {}
    for name, payload in (
        ("rewiring-null.json", rewiring_null),
        ("variant-flip-both.json", variant_flip_both),
        ("transfer.json", transfer_json),
        ("features.json", features_json),
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
