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
