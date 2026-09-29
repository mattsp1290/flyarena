"""Direct unit tests for `scripts/analysis/explain_report.py`'s Markdown
renderers -- extracted from `test_explain.py` alongside `explain_report.py`'s
own extraction from `explain.py` (see that module's doc comment). The full
report's end-to-end shape (every section present, deterministic across two
runs) is still covered by `test_explain.py::test_main_produces_deterministic_output`;
these tests exercise each renderer in isolation on small synthetic inputs.
"""

from __future__ import annotations

import explain_report


def _metric(name, kind, bio, p2_5, p97_5, spearman, *, qualifies=None):
    metric = {
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
    if qualifies is not None:
        metric["qualifiesBothGates"] = qualifies
    return metric


def test_render_transfer_matrix_handles_missing_metric_without_crashing():
    # Simulates biological's transfer solve being singular: every `T:*`
    # metric is dropped, so `metrics_by_name` has none of them.
    rendered = explain_report.render_transfer_matrix({}, "bio")
    assert "n/a" in rendered
    assert "foodBearing" in rendered


def test_render_metric_stats_table_shows_null_constant_as_na():
    constant_metric = _metric("pathLength:a->b", "feature", 2.0, 1.0, 1.0, 0.0)
    constant_metric["nullConstant"] = True
    rendered = explain_report.render_metric_stats_table([constant_metric])
    assert "n/a (constant in null)" in rendered
    assert "0.000" not in rendered


def test_render_metric_stats_table_bolds_rows_that_qualify_both_gates():
    # A thermo-methodology review finding (item 4): rows independently
    # passing both predeclared gates are visually flagged, using the
    # `qualifiesBothGates` field `explain.build_metric` stamps from the
    # single module-scope `explain.qualifies` predicate -- the renderer
    # itself never recomputes the gate.
    qualifying = _metric("T:rightClearance->thrust", "transfer", 0.0, 90.0, 160.0, 0.467, qualifies=True)
    non_qualifying = _metric("T:speed->brake", "transfer", 1.5, 1.0, 2.0, 0.9, qualifies=False)
    rendered = explain_report.render_metric_stats_table([qualifying, non_qualifying])
    assert "**T:rightClearance->thrust**" in rendered
    assert "**T:speed->brake**" not in rendered
    assert "| T:speed->brake |" in rendered


def test_render_metric_stats_table_does_not_bold_when_field_absent():
    # Category-evaluation-style synthetic fixtures (no `qualifiesBothGates`
    # key at all) must not crash or spuriously bold.
    metric = _metric("T:a->b", "transfer", 0.0, 1.0, 2.0, 0.5)
    rendered = explain_report.render_metric_stats_table([metric])
    assert "**T:a->b**" not in rendered
    assert "| T:a->b |" in rendered


# ---------------------------------------------------------------------------
# `_null_comparison_clause`: the "Question" paragraph's lead clause must be
# derived from the real biological percentile/null size, not the fixed
# "scoring below all 500 ... rewirings" string that was only ever true at
# the 0th percentile (a thermo-methodology review finding, C3 -- committed
# proof: `random-bridge/null-explanation-report.md`'s 24.8th-percentile
# report shipped the "below all 500" clause anyway, contradicting the
# percentile quoted two words later in the same sentence).
# ---------------------------------------------------------------------------


def test_null_comparison_clause_at_zero_percentile_says_below_all():
    # This is the shipped default's exact case (bioPercentile 0.0, 500
    # rewirings): the fixed phrasing this bug hardcoded is correct here,
    # and the byte-identity requirement on the published default artifact
    # depends on this exact string never changing for this input.
    assert explain_report._null_comparison_clause(0.0, 500) == "scoring below all 500 degree-preserving rewirings"


def test_null_comparison_clause_nonzero_percentile_says_at_the_nth_percentile_of():
    # random-bridge's real, committed number: 24.8th percentile of 500.
    assert (
        explain_report._null_comparison_clause(0.248, 500)
        == "scoring at the 24.8th percentile of 500 degree-preserving rewirings"
    )
    assert "below all" not in explain_report._null_comparison_clause(0.248, 500)


def test_null_comparison_clause_tied_with_null_floor_is_not_treated_as_below_all():
    # `explain_stats.rank_statistics`'s `bioPercentile = (kBelow + 0.5 *
    # kEqual) / n`: biological *tied* with the single lowest null value
    # (kBelow=0, kEqual=1) gives a tiny but strictly non-zero percentile
    # (0.5/500 = 0.001), not 0.0 -- "at the null floor" is a distinct case
    # from "below all N", and must not collapse into the same "below all"
    # wording just because it is numerically close to zero.
    floor_percentile = 0.5 / 500
    assert floor_percentile != 0.0
    clause = explain_report._null_comparison_clause(floor_percentile, 500)
    assert clause == "scoring at the 0.1th percentile of 500 degree-preserving rewirings"
    assert "below all" not in clause
