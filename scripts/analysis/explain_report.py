"""Markdown report rendering for `scripts/analysis/explain.py` (WP3):
`render_report_markdown` and its per-section helpers, plus the two
standalone table/disclosure renderers they call.

Extracted from `explain.py` (see `explain_stats.py`'s doc comment for the
same "1498 lines, 50% past this repo's 1000-line rule" rationale, shared by
both extractions). Every function here takes an already-built `explanation`
dict (and, where needed, the already-loaded `rewiring_null` dict) and
returns text -- no I/O, no statistics computation, no category-evaluation
logic. `render_report_markdown` itself was previously a single 270-line
function building every report section inline; it is now a thin
concatenation of one render helper per section (`render_method_section`
through `render_limitations_section`), matching the pattern
`render_transfer_matrix`/`render_metric_stats_table` already established in
this file -- a thermo-maintainability review finding (M1b): each section is
now unit-testable on its own, the way the two table renderers already were.

This module deliberately does not import `explain.py` (one-directional
import boundary, matching `scripts/null/null-report-trained.ts`'s own doc
comment: "This module deliberately does NOT import anything back from
`null-report.ts`... to keep the import boundary one-directional"). It only
imports `explain_stats.format_pct` (a pure formatter with no `explain.py`
dependency) and `features.py`/`transfer.py`'s own channel/population name
tables.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from explain_stats import format_pct
from features import OBSERVATION_CHANNELS, OUTPUT_POPULATIONS


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_transfer_matrix(metrics_by_name: Mapping[str, dict], field: str) -> str:
    """`metrics_by_name.get(...)` (not direct indexing): if biological's own
    transfer solve is singular, `explain.build_all_metrics` drops every
    `T:*` metric entirely (`explain.build_metric` returns `None` when
    `bio_value is None`), so a direct `metrics_by_name[name]` would raise
    `KeyError` here -- reported as "n/a" instead (an edge-case-review
    finding: this study's actual data never singular, so this path was
    previously unexercised and untested)."""
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
        # Rows that independently pass both predeclared gates (`explain.
        # qualifies`, stamped onto every metric as `qualifiesBothGates` by
        # `explain.build_metric`) are bolded -- a thermo-methodology review
        # finding: with the Finding narrative previously naming only one
        # per-category exemplar, a reader skimming a 24- or 40-row table had
        # no visual cue that a second (or third) row independently qualifies
        # too. Reads the precomputed field rather than recomputing the gate
        # here, so there is still exactly one implementation of "qualifies"
        # (`explain.qualifies`) in the whole pipeline.
        name_cell = f"**{metric['name']}**" if metric.get("qualifiesBothGates") else metric["name"]
        rows.append(
            f"| {name_cell} | {_fmt(metric['bio'])} | {_fmt(metric['nullMedian'])} | {_fmt(metric['p2_5'])} | "
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
    lines.append("")
    # The biological percentile *flip*, not only the rho comparison above --
    # a thermo-methodology review finding (suggestion 1): the restricted and
    # unrestricted readings don't just disagree on effect size, for
    # `weightedInDegree:thrust` they disagree on which tail biological sits
    # in entirely (0th percentile restricted vs. 100th percentile
    # unrestricted), which is a much more vivid illustration of
    # "definition-sensitive" than the rho-vs-threshold comparison alone.
    # Computed generically for every population from the pinned exploratory
    # input, not hardcoded to `thrust`.
    lines.append(
        "**Biological's percentile under each reading** (not just the rho comparison above -- a definition change "
        "can flip which tail biological sits in, not merely weaken the effect size):"
    )
    lines.append("")
    for population in OUTPUT_POPULATIONS:
        restricted = restricted_by_name[f"weightedInDegree:{population}"]
        unrestricted = exploratory_by_name[f"weightedInDegree:{population}"]
        flip_note = (
            " -- **the two readings disagree on direction, not just magnitude**"
            if {restricted["bioPercentile"], unrestricted["bioPercentile"]} == {0.0, 1.0}
            else ""
        )
        lines.append(
            f"- `weightedInDegree:{population}`: restricted {format_pct(restricted['bioPercentile'])} "
            f"(bio {_fmt(restricted['bio'])} vs null {_fmt(restricted['p2_5'])}-{_fmt(restricted['p97_5'])}) -> "
            f"unrestricted {format_pct(unrestricted['bioPercentile'])} (bio {_fmt(unrestricted['bio'])} vs null "
            f"{_fmt(unrestricted['p2_5'])}-{_fmt(unrestricted['p97_5'])}){flip_note}"
        )
    if explanation["finding"].get("definitionSensitive"):
        lines.append("")
        lines.append(
            "**This report's `structuralFeature` finding is definition-sensitive**: it is triggered by a "
            "`weightedInDegree` entry, and the finding would not hold under the unrestricted reading above."
        )
    return lines


def render_method_section(explanation: dict) -> list[str]:
    lines: list[str] = ["## Method", ""]
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
    calibration = explanation["calibration"]
    transfer_metrics = [m for m in explanation["metrics"] if m["kind"] == "transfer"]
    derived_metrics = [m for m in explanation["metrics"] if m["kind"] == "derived"]
    feature_metrics = [m for m in explanation["metrics"] if m["kind"] == "feature"]
    lines.append(
        f"**Multiple comparisons.** {calibration['metricsTested']} metrics are tested ({len(transfer_metrics)} "
        f"transfer entries, {len(derived_metrics)} derived predictors, {len(feature_metrics)} structural features"
        f"{' (predeclared count: 24/2/40; fewer here because some were not computable -- see the finding above)' if calibration['metricsTested'] != 66 else ''}"
        f"; {calibration['constantMetricCount']} of these are constant "
        "across the null and so can never reach the |rho| threshold). Correlations are reported descriptively, "
        "without per-metric significance testing; a permutation calibration "
        f"({calibration['permutations']} seeded permutations of the score, applied jointly to every metric family "
        "at once) found that at least one of the tested metrics reaches "
        f"\\|rho\\| >= {thresholds['spearmanRho']} by chance alone in {calibration['chanceHits']} of "
        f"{calibration['permutations']} permutations ({calibration['chanceRate'] * 100:.1f}%) -- this chance rate "
        "applies to any triggered linear-pathway or structural-feature finding below."
    )
    lines.append("")
    lines.append(
        "Throughout this report's metric tables, rows shown in **bold** independently pass both predeclared gates "
        "(outside the null's 2.5-97.5% range and |rho| at or above the threshold above) -- see the Finding section "
        "for the full, mechanically generated list."
    )
    lines.append("")
    return lines


def render_decoder_section(explanation: dict, rewiring_null: dict) -> list[str]:
    thresholds = explanation["thresholds"]
    variants = explanation["variants"]
    authored_bio_score = rewiring_null["biological"]["score"]
    authored_null_mean = rewiring_null["null"]["mean"]
    authored_bio_percentile = rewiring_null["bioPercentile"]
    authored_p_low = rewiring_null["pLow"]
    authored_p_high = rewiring_null["pHigh"]

    lines: list[str] = ["## Decoder-convention check", ""]
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
    return lines


def render_transfer_section(explanation: dict) -> list[str]:
    metrics_by_name = {m["name"]: m for m in explanation["metrics"]}
    transfer_metrics = [m for m in explanation["metrics"] if m["kind"] == "transfer"]
    derived_metrics = [m for m in explanation["metrics"] if m["kind"] == "derived"]

    lines: list[str] = ["## Linear transfer analysis", ""]
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
    return lines


def render_regime_section(explanation: dict) -> list[str]:
    regime = explanation["regime"]
    thresholds = explanation["thresholds"]

    lines: list[str] = ["## Regime check", ""]
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
    stability = regime["stability"]
    lines.append(
        "**Stability** (`T`'s own docstring: both numbers reported side by side, never `stable` alone). "
        f"Biological's continuous-time spectral abscissa is {stability['bio']['spectralAbscissa']:.4f} against a "
        f"leak rate of {stability['bio']['leakRate']:.4f} ({'stable' if stability['bio']['stable'] else 'UNSTABLE'}); "
        f"its per-substep discretized spectral radius is {stability['bio']['discretizedSpectralRadius']:.4f} "
        f"({'stable' if stability['bio']['discretizedStable'] else 'UNSTABLE'} -- must be < 1). Null medians: "
        f"spectral abscissa {stability['nullMedian']['spectralAbscissa']:.4f}, discretized spectral radius "
        f"{stability['nullMedian']['discretizedSpectralRadius']:.4f}. "
        f"{stability['unstableNullCount']} of 500 rewirings are unstable (continuous-time or discretized)."
    )
    lines.append("")
    return lines


def render_features_section(explanation: dict) -> list[str]:
    feature_metrics = [m for m in explanation["metrics"] if m["kind"] == "feature"]

    lines: list[str] = ["## Structural features", ""]
    lines.append(
        "40 predeclared graph features. The plan's stop/go gate 3 (`00-overview.md`: \"The feature list is not "
        "edited after the first run\") was **not met for feature 6**: its definition changed after a full "
        "production run against the original (unrestricted) reading, as disclosed below. The other 39 features "
        "were never edited."
    )
    lines.append("")
    lines.extend(render_feature6_disclosure(explanation))
    lines.append("")
    lines.append(render_metric_stats_table(feature_metrics))
    lines.append("")
    return lines


def render_finding_section(explanation: dict) -> list[str]:
    finding = explanation["finding"]
    lines: list[str] = ["## Finding", ""]
    lines.append(finding["summarySentence"])
    lines.append("")
    # Every metric that independently passes both predeclared gates, not
    # only the single representative each outcome category surfaces -- a
    # thermo-methodology review finding (Important 1): the summary sentence
    # above names only one exemplar per triggered category (via `max(...,
    # key=abs(spearman))`), which previously read as if that were the only
    # qualifying metric. `qualifyingMetricsNote` is generated mechanically
    # from `explanation["metrics"]` (`explain.qualifies`), not hand-picked.
    lines.append(finding["qualifyingMetricsNote"])
    lines.append("")
    if finding["categories"]:
        lines.append(f"Categories that hold, ranked by effect size: {', '.join(finding['ranked'])}.")
    else:
        lines.append("No predeclared category holds.")
    lines.append("")
    return lines


def render_limitations_section(explanation: dict) -> list[str]:
    calibration = explanation["calibration"]
    sources = explanation["sources"]
    producers = sources["producers"]

    lines: list[str] = ["## Limitations", ""]
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
    # Code-identity + graph-identity provenance disclosure -- a thermo-
    # methodology review finding (Important 2): `verify_provenance`
    # previously pinned only graph identity (`sourceGraphSha256`/
    # `rewireSourceSha256`), never code identity, so a `features.json`
    # regenerated from stale code against the *same* graphs would have
    # passed every existing check silently (exactly the failure mode this
    # study's own feature-6 stale-file incident hit, caught only by manual
    # diligence). `transfer.json`/`features.json`/`regime.json` now each
    # record a `producer.sourceSha256` this module's `verify_provenance`
    # recomputes and refuses to combine on mismatch; recorded here (`sources.
    # producers`) so a reader can see exactly what was pinned without
    # re-deriving it. A second thermo-fix-verification review finding
    # showed the *first* version of this code-identity check itself
    # undercounted: a hand-maintained flat filename list omitted files a
    # producer genuinely depends on (`scripts/training/episode.ts`,
    # `src/lib/connectome/model.ts`, `scripts/data/rewire.py`, ...) --
    # replaced below by walking each producer's real import graph from its
    # entry file, so the sentence here describes what that walk actually
    # covers (and, just as importantly, what it deliberately does not).
    transfer_dep_count = len(producers["transfer"]["dependencies"])
    features_dep_count = len(producers["features"]["dependencies"])
    regime_dep_count = len(producers["regime"]["dependencies"])
    lines.append(
        "- **Provenance pins two independent things.** Every input's `sourceGraphSha256`/`rewireSourceSha256` "
        "pins *graph identity* (all five inputs were computed against the same 502 graphs -- biological, "
        "disconnected, 500 rewirings). `transfer.json`/`features.json`/`regime.json` additionally pin *producer "
        "code identity*: each records a `producer.sourceSha256` "
        f"(`transfer.py` {producers['transfer']['sourceSha256'][:12]}..., "
        f"`features.py` {producers['features']['sourceSha256'][:12]}..., "
        f"`regime-check.ts` {producers['regime']['sourceSha256'][:12]}...) -- a sha256 over the *real import-graph "
        "closure* walked from that script's own entry file (repo-relative source files only; third-party "
        f"packages and the standard library are never walked into): {transfer_dep_count} files for `transfer.py` "
        f"(including `scripts/analysis/graph_io.py` and, transitively, `scripts/data/rewire.py`/`binfmt.py`, "
        f"the code that decodes every graph's own bytes), {features_dep_count} for `features.py`, and "
        f"{regime_dep_count} for `regime-check.ts` (including the episode driver, `src/lib/connectome/model.ts`'s "
        "substep execution, and the rest of the simulation code this study characterizes -- every file listed in "
        "`producer.dependencies`) -- the same `filename+NUL+bytes` scheme `scripts/data/compile.py`'s "
        "`compiler_source_sha256()` already uses, generalized to repo-relative paths. `verify_provenance` "
        "recomputes that hash from the current working tree (by re-walking the same import graph, not by "
        "re-reading a stored file list) and refuses to combine a stale input. **What this does not pin:** the "
        "Node/Python runtime version each producer ran under (recorded separately, per input, as `producer.host`) "
        "and every third-party package (numpy, etc.) -- those are pinned by this repo's lockfiles "
        "(`package-lock.json`/`uv.lock`), not hashed into `producer.sourceSha256`. The one deliberate source-file "
        "exception is `features-exploratory-unrestricted.json`: a pinned, stale-code exploratory input (feature "
        "6's pre-adjudication run, disclosed above) -- its *content* sha256 is still pinned and verified, but it "
        "is explicitly exempt from the code-identity check, since re-running it against current code would defeat "
        "its purpose as a historical snapshot of what the unrestricted reading looked like at adjudication time."
    )
    lines.append("- **No biological claim.** See \"This model only\" above.")
    lines.append("")
    return lines


def render_report_markdown(explanation: dict, rewiring_null: dict) -> str:
    """`rewiring_null` (the already-loaded, already-sha-verified
    `rewiring-null-v1.json`, `explanation["sources"]["rewiringNullSha256"]`'s
    own source) supplies the authored-condition baseline numbers quoted in
    the "Question" section and the decoder-convention table's first row --
    read from that file directly rather than hardcoded, so this report can
    never drift from the artifact it is describing."""
    rewiring_null_baseline = rewiring_null
    authored_bio_score = rewiring_null_baseline["biological"]["score"]
    authored_null_mean = rewiring_null_baseline["null"]["mean"]
    authored_null_std = rewiring_null_baseline["null"]["std"]
    authored_bio_percentile = rewiring_null_baseline["bioPercentile"]
    authored_p_low = rewiring_null_baseline["pLow"]

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

    lines.extend(render_method_section(explanation))
    lines.extend(render_decoder_section(explanation, rewiring_null))
    lines.extend(render_transfer_section(explanation))
    lines.extend(render_regime_section(explanation))
    lines.extend(render_features_section(explanation))
    lines.extend(render_finding_section(explanation))
    lines.extend(render_limitations_section(explanation))
    return "\n".join(lines)
