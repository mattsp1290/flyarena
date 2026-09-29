# Subgraph-selection robustness (under this model)

**Question.** Do the headline authored-decoder findings -- biological at the bottom of its degree-preserving null, the clearance->thrust metrics explaining the gap, and pathway P being pathway-supported -- depend on *how* the 1,008-neuron subgraph was selected? This report tests four predeclared alternative selections, all compiled from the same pinned MaleCNS raw files, under the authored decoder and the default arena task only. Every result below is descriptive and bound to this model; none is a claim about the real fly.

## Selections

| id | Change versus `compile.py` defaults | Size |
| --- | --- | --- |
| `larger` | `BRIDGE_TARGET 1600` | about 1,808 neurons |
| `smaller` | `BRIDGE_TARGET 400` | about 608 neurons |
| `random-bridge` | Bridge neurons are sampled uniformly from the eligible candidates with a seeded RNG (seed `20260927`) instead of by degree rank, size 800 | 1,008 |
| `alt-sensory-mapping` | The same neurons, but sensory→channel assignment uses a seeded permutation (seed `20260927`) of the sorted sensory ids instead of contiguous body-id blocks. The descending mapping is unchanged | 1,008 |

## Predeclared outcome rules

Quoted verbatim from `.agents/plans/selection-robustness/00-overview.md`:

- **Null holds:** biological is below that selection's rewired-null 25th percentile.
- **Explanation replicates:** at least one of the two original gate-passing **transfer** entries (`T:rightClearance->thrust`, `T:forwardClearance->thrust`), which are the ones P's search optimizes, passes both predeclared gates in that selection's explanation. `weightedInDegree:thrust` is reported separately as `structuralReplicates` and never decides `replicates`. `selection-report.ts` recomputes both flags **in TypeScript** from the raw `spearman` and `bioPercentile` values and the published thresholds. It does not trust the flags from `explain.py`.
- **Pathway supported:** P for that selection passes the original rule (at or above that selection's null 25th percentile, and above the 95th percentile of both C and M).
- A selection whose null, C, or M is degenerate (the IQR guard) is not categorized.
- **Overall:** reported on two axes, never as "4 independent selections". `larger`, `default`, and `smaller` are nested cuts of one degree ranking, so together they test **bridge-population size**. `random-bridge` tests **selection method**. `alt-sensory-mapping` tests **the authored channel mapping**, not topology. Each verdict is `true`, `false`, or `"indeterminate"` with a reason. "Robust to size" is `true` if all three findings hold for both `larger` and `smaller`, `false` if any finding fails in a categorized one, and `"indeterminate"` if either is uncategorized and none failed. "Robust to method" follows the same rule for `random-bridge`. The mapping result is reported by itself. Degenerate selections are not categorized.
- **Search-budget disclosure:** each selection reports P's swap count `k`, `targetReached`, and the size of its bridge-candidate pool. A P that stopped at the 200-swap cap without reaching its target is reported as "search-limited", not "not supported".
- **Coverage check (`random-bridge`, and recorded for all):** per input channel and output population, the number of selected bridge neurons with at least one edge from that channel's sensory neurons and at least one edge to that population's descending neurons. Any channel or population with zero coverage is flagged, and a selection with a flagged channel is not categorized for the pathway finding.

## Per-selection results

### `larger` -- Larger (bridge target 1600, ~1,808 neurons)

Graph sha `6a6304aa9ecc`, global gain 0.00017556179775280898, 1808 neurons, compiled from git revision `da1375aeedc0` (compiler source sha `c015d9fa5ca0`).

Null holds: biological's score -0.4679 vs. the rewired-null 25th percentile 1.3041 (bioPercentile 0.0% of 500 rewirings, i.e. biological ranks below 0 of 500).

Explanation replicates (structuralReplicates: true): `T:forwardClearance->thrust` (rho=0.345, 0.6% of null); `T:rightClearance->thrust` (rho=0.450, 0.8% of null); `weightedInDegree:thrust` (rho=0.360, 0.0% of null). Mirrored decoder (flip-both): biological moves to the 0.0% percentile. Exploratory feature: omitted (selection-robustness WP2 (--selection-mode): the historical pre-adjudication features-exploratory-unrestricted snapshot has no per-selection counterpart and cannot be regenerated against this selection's own rewired graphs (docs/null-explanation-report.md:201)).

Pathway P: **pathway-supported** -- score 3.6612 vs. null floor 1.3041, C p95 -0.4520, M p95 -0.0876, k=8 of 200 swaps, targetReached=true, bridge-candidate pool 2557. Q channel-specific: true (score 3.0813, k=7).

Coverage: not flagged (every sensory channel and descending population has at least one covered bridge neuron).

Categorized: yes.

### `smaller` -- Smaller (bridge target 400, ~608 neurons)

Graph sha `47992d462e26`, global gain 0.000248015873015873, 608 neurons, compiled from git revision `da1375aeedc0` (compiler source sha `c015d9fa5ca0`).

Null holds: biological's score -0.4223 vs. the rewired-null 25th percentile 2.8645 (bioPercentile 0.0% of 500 rewirings, i.e. biological ranks below 0 of 500).

Explanation replicates (structuralReplicates: true): `T:leftClearance->thrust` (rho=0.300, 0.4% of null); `T:rightClearance->thrust` (rho=0.403, 0.4% of null); `weightedInDegree:thrust` (rho=0.392, 0.0% of null). Mirrored decoder (flip-both): biological moves to the 0.0% percentile. Exploratory feature: omitted (selection-robustness WP2 (--selection-mode): the historical pre-adjudication features-exploratory-unrestricted snapshot has no per-selection counterpart and cannot be regenerated against this selection's own rewired graphs (docs/null-explanation-report.md:201)).

Pathway P: **pathway-supported** -- score 3.4221 vs. null floor 2.8645, C p95 -0.2710, M p95 0.5809, k=8 of 200 swaps, targetReached=true, bridge-candidate pool 2557. Q channel-specific: false (score 2.7147, k=9).

Coverage: not flagged (every sensory channel and descending population has at least one covered bridge neuron).

Categorized: yes.

### `random-bridge` -- Random bridge (seeded-uniform bridge sample, seed 20260927)

Graph sha `340763358b3b`, global gain 0.000532141336739038, 1008 neurons, compiled from git revision `a4a57f2c1d2e` (compiler source sha `56b3b7c6b077`).

Null holds: biological's score 2.9527 vs. the rewired-null 25th percentile 2.9802 (bioPercentile 24.8% of 500 rewirings, i.e. biological ranks below 124 of 500).

Explanation does not replicate (structuralReplicates: true): `weightedInDegree:thrust` (rho=0.395, 0.0% of null). Mirrored decoder (flip-both): biological moves to the 29.4% percentile. The mirrored run cleared the conditional 25% threshold, so the single-axis variants ran: flip-thrust 94.0%, flip-yaw 81.0%. Exploratory feature: omitted (selection-robustness WP2 (--selection-mode): the historical pre-adjudication features-exploratory-unrestricted snapshot has no per-selection counterpart and cannot be regenerated against this selection's own rewired graphs (docs/null-explanation-report.md:201)).

Pathway P/Q: **degenerate**, not categorized. P's search reached its transfer target with 0 swaps, so the size-matched C/M/MQ controls are unperturbed copies of biological and cannot be distinguished from it. P score 2.9527, Q score 2.9527, published null floor 2.9802, k=0, qK=0, bridge-candidate pool 2557.

Coverage: not flagged (every sensory channel and descending population has at least one covered bridge neuron).

Categorized: no -- P's search reached its transfer target with 0 swaps, so the size-matched C/M/MQ controls are unperturbed copies of biological and cannot be distinguished from it (IQR guard on the C/M control arms).

### `alt-sensory-mapping` -- Alt sensory mapping (seeded-permutation channel assignment, seed 20260927)

Graph sha `25d48a35f8cc`, global gain 0.00019837727389950208, 1008 neurons, compiled from git revision `a4a57f2c1d2e` (compiler source sha `56b3b7c6b077`).

Null holds: biological's score -1.3397 vs. the rewired-null 25th percentile 2.6386 (bioPercentile 0.0% of 500 rewirings, i.e. biological ranks below 0 of 500).

Explanation replicates (structuralReplicates: true): `T:forwardClearance->thrust` (rho=0.367, 0.4% of null); `T:leftClearance->thrust` (rho=0.369, 0.0% of null); `T:rightClearance->thrust` (rho=0.455, 0.6% of null); `weightedInDegree:thrust` (rho=0.513, 0.0% of null). Mirrored decoder (flip-both): biological moves to the 0.0% percentile. Exploratory feature: omitted (selection-robustness WP2 (--selection-mode): the historical pre-adjudication features-exploratory-unrestricted snapshot has no per-selection counterpart and cannot be regenerated against this selection's own rewired graphs (docs/null-explanation-report.md:201)).

Pathway P: **not-supported** -- score 1.3933 vs. null floor 2.6386, C p95 -1.3392, M p95 -0.6479, k=2 of 200 swaps, targetReached=true, bridge-candidate pool 2557. Q channel-specific: false (score 1.3851, k=2).

Coverage: not flagged (every sensory channel and descending population has at least one covered bridge neuron).

Categorized: yes.

## Overall

- **Robust to size** (`smaller` + `larger`): **true**.
- **Robust to method** (`random-bridge`): **false** -- random-bridge: explanation does not replicate.
- **Channel mapping** (`alt-sensory-mapping`): **false** -- alt-sensory-mapping: pathway is not-supported.

This is reported on two axes plus the mapping result by itself, never as "N of 4 selections agree" -- `larger`/`smaller` are nested cuts of one degree ranking (a size test), `random-bridge` is a method test, and `alt-sensory-mapping` is a channel-mapping test, not a topology test. See the Predeclared outcome rules above for the exact aggregation rule.

## Provenance

- Default (shipped) graph sha: `f1a0f982ffdf`.
- Raw MaleCNS file shas (pinned, `scripts/data/download.py`): `2177e246113e`, `95c928922066`, `e35da783d1c6`.
- Current compiler source sha (`scripts/data/compile.py` + `scripts/data/selections.py`): `c015d9fa5ca0`.
- Producer: `scripts/selections/selection-report.ts`, source sha `86a061e2ff0d`, 48 dependency files.
- Each selection above records its own `compiledFromGitRevision`/`compilerSourceSha256` -- see "Per-selection results".

## Limitations

- `larger`, `default`, and `smaller` are nested cuts of one degree ranking, testing size, not independent selections.
- The 200-swap cap is not scaled to the bridge-pool size.
- The exploratory unrestricted feature is not recomputed for any selection (`--selection-mode` records `exploratory: null`).
- One seed per seeded variant (`random-bridge`, `alt-sensory-mapping`).
- All selections use the same candidate pools and synapse threshold, so `bridgePoolSize` is a property of that shared candidate set, not of the selection -- it is expected to be identical across all four rows, not a bug.
- `alt-sensory-mapping` changes the authored channel assignment; it is not a topology test.
- Authored decoder and the default arena task only.
- Uncorrected comparisons across 4 selections x 3 findings each (12 comparisons); no multiple-comparisons adjustment is applied.
- This model only -- no biological claim about the real fly is made anywhere in this report.
