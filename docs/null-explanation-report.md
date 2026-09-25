# Explaining the null result (under this model)

**Question.** `docs/rewiring-null-report.md` found the biological MaleCNS graph scoring below all 500 degree-preserving rewirings under the authored decoder (biological -0.2199, null mean 2.5653, sd 0.9370, 0.0th percentile, `p_low = 0.0020`). This report tests three predeclared, descriptive explanations for that result under this model only -- the authored encoder, this rate-model dynamics, and this arena -- and makes no claim about the real fly.

## Method

Three predeclared analyses (`.agents/plans/null-explanation/00-overview.md`), evaluated only after all three finished, with a fixed feature list not edited after the first run:

1. **Decoder-convention check.** Re-score biological and all 500 rewirings with the authored decoder's thrust and yaw signs both flipped (`authored-flip-both`). Predeclared rule: only run the two single-axis variants if the mirrored run moves biological to at least the 25th percentile.
2. **Linear transfer analysis.** For each graph, the steady-state linear transfer matrix `T = O(lambda I - g A)^-1 B` (3 outputs x 8 input channels), gated by a linear-regime validity check.
3. **Structural feature attribution.** 40 predeclared graph features, each compared against the null and rank-correlated with score.

**Predeclared thresholds:**

| Threshold | Value |
| --- | --- |
| Decoder-convention bio percentile | >= 25% |
| Spearman \|rho\| (linear-pathway / structural-feature) | >= 0.3 |
| Regime gate: rate-clamp fraction | <= 20% |
| Regime gate: steady-state distance | <= 0.5 |
| Regime gate: condition number | <= 1e+08 |

**Multiple comparisons.** 66 metrics are tested (24 transfer entries, 2 derived predictors, 40 structural features). Correlations are reported descriptively, without per-metric significance testing; a permutation calibration (1000 seeded permutations of the score against the fixed metric set) found that at least one of the 66 metrics reaches \|rho\| >= 0.3 by chance alone in 0.0% of permutations -- this chance rate applies to any triggered linear-pathway or structural-feature finding below.

## Decoder-convention check

| Condition | Biological score | Null mean | Bio percentile | p_low | p_high |
| --- | --- | --- | --- | --- | --- |
| authored, opponent parked | -0.2199 | 2.5653 | 0.0% | 0.0020 | 1.0000 |
| authored (thrust and yaw flipped), opponent parked | -0.7234 | 4.2893 | 0.0% | 0.0020 | 1.0000 |

Mirrored biological percentile (0.0%) stayed below the predeclared 25% threshold, so the single-axis variants (`authored-flip-thrust`, `authored-flip-yaw`) were skipped per the predeclared rule (`.agents/plans/null-explanation/00-overview.md`).

## Linear transfer analysis

`T[channel, population]`: steady-state gain from a unit-held input on that channel to that output population, exact within the model's rate/input clamps.

**Biological `T`:**

| channel \ population | thrust | yaw | brake |
| --- | --- | --- | --- |
| foodBearing | 0.010068 | 0.025640 | 0.061185 |
| foodDistance | 0.004898 | 0.015220 | 0.043277 |
| hazardBearing | -0.017035 | -0.000790 | 0.029804 |
| hazardDistance | 0.002051 | 0.005298 | 0.017776 |
| forwardClearance | 0.002152 | 0.006692 | 0.015635 |
| leftClearance | 0.003018 | 0.005662 | 0.015304 |
| rightClearance | 0.003087 | 0.006746 | 0.021815 |
| speed | 0.001547 | 0.005251 | 0.022019 |

**Null median `T`:**

| channel \ population | thrust | yaw | brake |
| --- | --- | --- | --- |
| foodBearing | 0.043485 | 0.035275 | 0.043220 |
| foodDistance | 0.029009 | 0.022984 | 0.028361 |
| hazardBearing | 0.024595 | 0.018956 | 0.023730 |
| hazardDistance | 0.016253 | 0.013375 | 0.016498 |
| forwardClearance | 0.019514 | 0.015892 | 0.019760 |
| leftClearance | 0.016782 | 0.014073 | 0.016984 |
| rightClearance | 0.020400 | 0.016570 | 0.020164 |
| speed | 0.019769 | 0.016675 | 0.019140 |

**Transfer entry statistics (sorted by \|rho\|):**

| metric | biological | null median | null 2.5% | null 97.5% | bio percentile | rho | rho 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T:rightClearance->thrust | 0.0031 | 0.0204 | 0.0055 | 0.0440 | 0.6% | 0.467 | [0.393, 0.539] |
| T:forwardClearance->thrust | 0.0022 | 0.0195 | 0.0079 | 0.0391 | 0.0% | 0.353 | [0.275, 0.429] |
| T:leftClearance->thrust | 0.0030 | 0.0168 | 0.0045 | 0.0337 | 1.0% | 0.298 | [0.215, 0.378] |
| T:hazardBearing->thrust | -0.0170 | 0.0246 | 0.0069 | 0.0522 | 0.0% | 0.249 | [0.166, 0.328] |
| T:foodBearing->thrust | 0.0101 | 0.0435 | 0.0145 | 0.0863 | 0.6% | -0.189 | [-0.271, -0.105] |
| T:hazardDistance->thrust | 0.0021 | 0.0163 | 0.0036 | 0.0356 | 0.8% | 0.188 | [0.102, 0.271] |
| T:foodDistance->thrust | 0.0049 | 0.0290 | 0.0075 | 0.0618 | 2.0% | 0.172 | [0.082, 0.258] |
| T:foodBearing->brake | 0.0612 | 0.0432 | 0.0164 | 0.0815 | 84.4% | 0.143 | [0.052, 0.232] |
| T:foodDistance->brake | 0.0433 | 0.0284 | 0.0098 | 0.0600 | 84.2% | -0.136 | [-0.223, -0.049] |
| T:forwardClearance->brake | 0.0156 | 0.0198 | 0.0068 | 0.0363 | 30.6% | -0.111 | [-0.199, -0.019] |
| T:rightClearance->yaw | 0.0067 | 0.0166 | 0.0050 | 0.0373 | 8.0% | 0.105 | [0.016, 0.190] |
| T:leftClearance->brake | 0.0153 | 0.0170 | 0.0059 | 0.0356 | 40.4% | -0.077 | [-0.165, 0.011] |
| T:hazardBearing->brake | 0.0298 | 0.0237 | 0.0065 | 0.0529 | 68.8% | -0.077 | [-0.162, 0.010] |
| T:foodBearing->yaw | 0.0256 | 0.0353 | 0.0118 | 0.0722 | 26.6% | -0.076 | [-0.163, 0.015] |
| T:rightClearance->brake | 0.0218 | 0.0202 | 0.0055 | 0.0433 | 56.4% | -0.060 | [-0.145, 0.027] |
| T:hazardDistance->yaw | 0.0053 | 0.0134 | 0.0019 | 0.0309 | 9.6% | 0.059 | [-0.025, 0.145] |
| T:hazardDistance->brake | 0.0178 | 0.0165 | 0.0037 | 0.0332 | 55.0% | 0.043 | [-0.046, 0.131] |
| T:speed->thrust | 0.0015 | 0.0198 | 0.0050 | 0.0431 | 1.0% | 0.037 | [-0.051, 0.123] |
| T:speed->brake | 0.0220 | 0.0191 | 0.0043 | 0.0413 | 61.2% | -0.031 | [-0.119, 0.058] |
| T:foodDistance->yaw | 0.0152 | 0.0230 | 0.0059 | 0.0552 | 24.0% | -0.031 | [-0.119, 0.058] |
| T:hazardBearing->yaw | -0.0008 | 0.0190 | 0.0038 | 0.0457 | 0.0% | 0.027 | [-0.056, 0.113] |
| T:speed->yaw | 0.0053 | 0.0167 | 0.0027 | 0.0363 | 8.0% | 0.015 | [-0.071, 0.100] |
| T:forwardClearance->yaw | 0.0067 | 0.0159 | 0.0039 | 0.0346 | 10.0% | 0.011 | [-0.075, 0.099] |
| T:leftClearance->yaw | 0.0057 | 0.0141 | 0.0040 | 0.0288 | 5.0% | 0.001 | [-0.082, 0.084] |

**Derived predictors** (`turnGain = T[yaw,foodBearing] - T[yaw,hazardBearing]`, `approachGain = T[thrust,foodDistance]`):

| metric | biological | null median | null 2.5% | null 97.5% | bio percentile | rho | rho 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approachGain | 0.0049 | 0.0290 | 0.0075 | 0.0618 | 2.0% | 0.172 | [0.083, 0.259] |
| turnGain | 0.0264 | 0.0155 | -0.0169 | 0.0507 | 71.8% | -0.099 | [-0.189, -0.010] |

## Regime check

Authored episodes on 10 held-out seeds (`30001..30010`) for biological, disconnected, and all 500 rewirings, measuring the fraction of neuron-substeps with an active rate clamp and the linear steady-state distance `||r_t - r*(u_t)|| / ||r*(u_t)||`.

| | rate-clamp fraction | steady-state distance |
| --- | --- | --- |
| biological | 0.63% | 0.1371 |
| null (median over 500 rewirings) | 1.04% | 0.1516 |

The aggregate regime gate **passed** (biological and the null median both within threshold, biological's transfer solve not ill-conditioned or singular). 0 of 500 rewirings were individually excluded from the transfer-kind correlations above for failing their own per-graph regime threshold (none).

This licenses treating the linear analysis as applicable to both biological and the null sample under this model (steady-state distances and clamp fractions are all well inside threshold); it does not by itself certify that any single transfer entry explains the score -- that still requires the outside-range-and-\|rho\|-threshold test above.

## Structural features

40 predeclared graph features (fixed before any analysis ran; not edited after the first run). Feature 6 (`weightedInDegree`, mean input-restricted weighted in-degree per output population) was adjudicated during WP2: the plan's "input->output weighted in-degree" wording was read as restricted to edges whose *presynaptic* neuron is input-labeled (channel-mapped), on plan-text grounds (features 1/2's own "input"/"from any input neuron" usage, and the parallel with feature 4's unqualified "edges into output neurons" phrasing) decided **before any result was seen**, not selected because of its outcome (bean `flyarena-r37r`'s log; advocate write-ups under `/tmp/claude-1000/feature6-debate/`). An **unrestricted** variant (counting edges from *any* presynaptic neuron, not only input-labeled ones) was also computed during that adjudication for comparison and is disclosed here as **exploratory, non-predeclared** -- it is not part of the frozen feature list and is not used in the outcome-category evaluation below: biological's unrestricted thrust in-degree is 1476.5 (null mean 1126.4, sd 72.4, ~100th percentile), with rank correlation to score rho <= 0.072 on every output population -- weaker on every population than the predeclared, input-restricted reading, and it would not itself qualify for the structural-feature-associated category (\|rho\| < 0.3).

| metric | biological | null median | null 2.5% | null 97.5% | bio percentile | rho | rho 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weightedInDegree:thrust | 0.0000 | 123.3125 | 94.8125 | 161.9375 | 0.0% | 0.394 | [0.318, 0.468] |
| weightBalance:thrust | 2866.0000 | 2035.5000 | -612.0000 | 4638.0000 | 74.6% | 0.202 | [0.115, 0.286] |
| inhibitoryPathCount:thrust | 191667.0000 | 164767.0000 | 156614.0000 | 172749.0000 | 100.0% | -0.157 | [-0.240, -0.074] |
| weightBalance:brake | 5918.0000 | 2076.0000 | -469.0000 | 4616.0000 | 99.8% | -0.122 | [-0.210, -0.034] |
| excitatoryPathCount:thrust | 190092.0000 | 165096.0000 | 157104.0000 | 172754.0000 | 100.0% | -0.110 | [-0.195, -0.025] |
| feedForwardTriangleCount | 506380.0000 | 221693.0000 | 220079.0000 | 223013.0000 | 100.0% | 0.081 | [-0.011, 0.173] |
| inhibitoryPathCount:yaw | 145738.0000 | 133076.0000 | 126200.0000 | 139421.0000 | 100.0% | -0.069 | [-0.158, 0.020] |
| weightedInDegree:yaw | 0.2500 | 99.4375 | 71.2500 | 135.7500 | 0.0% | 0.053 | [-0.032, 0.140] |
| inhibitoryPathCount:brake | 175309.0000 | 161702.0000 | 154125.0000 | 168636.0000 | 100.0% | -0.050 | [-0.138, 0.038] |
| weightedInDegree:brake | 111.0625 | 122.6875 | 89.2500 | 161.5000 | 25.4% | -0.040 | [-0.130, 0.050] |
| excitatoryPathCount:yaw | 173197.0000 | 133166.5000 | 126397.0000 | 139231.0000 | 100.0% | -0.033 | [-0.121, 0.053] |
| pathLength:speed->yaw | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 99.9% | -0.033 | [-0.061, 0.000] |
| meanPathLength | 1.6250 | 1.0000 | 1.0000 | 1.0000 | 100.0% | -0.033 | [-0.061, 0.000] |
| weightBalance:yaw | 5170.0000 | 1797.0000 | -549.0000 | 4151.0000 | 99.4% | -0.012 | [-0.098, 0.075] |
| excitatoryPathCount:brake | 197073.0000 | 161651.0000 | 153813.0000 | 169198.0000 | 100.0% | 0.010 | [-0.079, 0.099] |
| reciprocity | 0.2526 | 0.0632 | 0.0602 | 0.0659 | 100.0% | 0.006 | [-0.081, 0.093] |
| twoCycleCount | 5849.0000 | 1464.0000 | 1395.0000 | 1527.0000 | 100.0% | 0.006 | [-0.080, 0.095] |
| pathLength:foodBearing->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:foodBearing->yaw | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:foodBearing->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:foodDistance->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:foodDistance->yaw | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:foodDistance->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:hazardBearing->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:hazardBearing->yaw | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:hazardBearing->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:hazardDistance->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:hazardDistance->yaw | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:hazardDistance->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:forwardClearance->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:forwardClearance->yaw | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:forwardClearance->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:leftClearance->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:leftClearance->yaw | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:leftClearance->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:rightClearance->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:rightClearance->yaw | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:rightClearance->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |
| pathLength:speed->thrust | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.000 | [0.000, 0.000] |
| pathLength:speed->brake | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.0% | 0.000 | [0.000, 0.000] |

## Finding

Biological's low score is associated with: the linear transfer entry T:rightClearance->thrust sits outside the null's 2.5-97.5% range (rank correlation with score rho=0.467); the structural feature weightedInDegree:thrust sits outside the null's 2.5-97.5% range (rank correlation with score rho=0.394) -- a descriptive correlation, not a causal claim.

Categories that hold, ranked by effect size: linearPathway, structuralFeature.

## Limitations

- **66 metrics tested.** No per-metric significance testing is performed; correlations are descriptive. The permutation calibration above found a 0.0% chance that at least one metric reaches the |rho| threshold by chance alone, and that rate applies to any triggered linear-pathway or structural-feature finding in this report.
- **Regime-invalid is never reported as a positive finding.** If the aggregate regime gate fails, the linear-pathway analysis is reported as regime-invalid (inconclusive), never as a positive finding, regardless of any individual transfer entry's statistics.
- **This model only.** Every analysis here describes the authored decoder, this rate-model dynamics, and this arena running on the measured biological topology versus 500 degree-preserving rewirings of it. Nothing here is a claim about the real fly's neural function or behavior, and no rewiring's topology is claimed to be causally "worse" or "better" than biological's.
- **The linear analysis is valid only to the measured regime extent.** `T` is the model's exact fixed-point gain when no rate/input clamp is active; the regime check quantifies how close the real, clamped, discretized simulation actually sits to that fixed point, and the linear-pathway category is gated on that check, not assumed.
- **Correlation is not causation.** A rank correlation between a structural or transfer metric and score across the 500 rewirings describes an association within this null model's sample, not a causal mechanism.
- **No biological claim.** See "This model only" above.
