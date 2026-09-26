# Clearance -> thrust pathway interventions

## Question

`docs/null-explanation-report.md` (on `origin/main`) associates biological's 0th-percentile score with the
clearance->thrust pathway:

- `T:rightClearance->thrust`: biological 0.0031 versus null median 0.0204, rho 0.467;
- `T:forwardClearance->thrust`: 0.0022 versus 0.0195, rho 0.353;
- input-restricted `weightedInDegree:thrust`: biological **0.0** versus null median 123.3, rho 0.394.

In the biological graph, no input-labeled neuron synapses directly onto a thrust neuron. Degree-preserving
rewiring creates such edges. This experiment edits the biological graph along this pathway with predeclared
interventions, with degree-preserving controls matched in size, and scores it under both the authored decoder
and CEM-retrained readouts.

## Method

**Primary -- targeted degree-preserving swaps (P):** starting from the biological graph, repeatedly apply double-edge
swaps `(a->b, c->d) -> (a->d, c->b)` where `a` is input-labeled, `d` is a thrust-population neuron, and both
collateral endpoints `b` and `c` are bridge neurons (neither input nor output assigned). Choose greedily by the
largest first-order increase of `T:rightClearance->thrust + T:forwardClearance->thrust`. Stop when both entries
reach at least the null's 25th percentile, or at 200 swaps. This preserves in-degree, out-degree, the weight
multiset, and the presynaptic signs (per-neuron, `presynapticSigns`). `k` is the number of accepted swaps.

**Control -- random degree-preserving swaps (C):** 100 graphs, each with exactly `k` uniformly random valid
double-edge swaps anywhere in the graph (seeds 0-99). This tests "any perturbation of this size".

**Control -- class-matched random swaps (M):** 100 graphs, each with exactly `k` valid swaps drawn uniformly (not
greedily) from the same candidate class as P (`a` input-labeled, `d` thrust, `b`/`c` bridge), seeds 1000-1099.

**Secondary -- magnitude-matched removal (R):** remove the attributed edges into thrust neurons from the input
side that carry at least 50% of the first-order transfer. Because the biological graph has none, R is expected
to be empty. If so, it is reported as not applicable.

**Secondary -- clearance-only targeting (Q):** the same as P, but only `rightClearance` and `forwardClearance`
input neurons may be the source `a`. This tests channel specificity rather than generic input->thrust wiring.

### Predeclared outcome categories (authored decoder)

- **Pathway supported**: P's score is at or above the null's 25th percentile and above the 95th percentile of both the C and M distributions.
- **Edge-class effect**: P is at or above the null's 25th percentile and above C's 95th percentile, but at or below M's 95th percentile. Any input->thrust edges of this class help about equally, and the specific optimized edges do not matter.
- **Generic rewiring effect**: P is at or above the null's 25th percentile but at or below C's 95th percentile. Any k swaps help about equally.
- **Not supported**: P stays below the null's 25th percentile.
- **Channel-specific (modifier, authored decoder only)**: Q (clearance-channel sources only) is above the null's
  25th percentile and above the 95th percentile of its own size-matched class control MQ: 100 graphs, each with
  exactly `k_Q = |Q swaps|` uniform valid swaps from Q's candidate class, seeds 2000-2099. Q is never compared
  against the P-sized C or M arms. Q is not evaluated with trained readouts, and this report states this.

Claim language: a positive result means "the net effect of this accepted swap set", not an isolated single-edge
causal effect.

### Trained decoder

The same C/M comparisons apply, but the **governing reference** is the freshly trained controls: 5 C graphs
(C000-C004) and 5 M graphs (M1000-M1004), each at trainer seed 101 -- not the published 500-graph authored null,
and the authored floor's null-percentile prong is not applied on this side (see the disclosure below for why).
The published trained null (`rewiring-null-v1.json` `trained`, 20 full rewirings) is reported for context only
and does not decide the category. With 5 graphs per arm, the trained cutoff is "above the maximum of that arm's
5". The result is robust only if all three P trainer seeds (101/202/303) agree on the category, because the
trained null was trainer-seed-sensitive.

The trained decoder's predeclared C/M-max comparison can rule pathway-supported and edge-class-effect in or out (P above/below the max of the freshly-trained 5-graph C/M arms at trainer seed 101), but 00-overview.md does not say how to report the finer generic-rewiring-effect vs not-supported split when P does not clear the C arm: that split needs a trained-null percentile floor, and this study's only trained null (rewiring-null-v1.json's published n=20 sample) is reported for context only, per 00-overview.md, not as a decisive threshold. 'no-specific-effect' is a reporting convention this study's coordinator adopted after the trained scores were known (methodology review, 2026-09-26), to avoid forcing an unlicensed generic/not-supported label -- it does not change which predeclared comparison P passed or failed, only how the undecidable case is named.

## Graph construction

- P: `k = 6` accepted swaps, target reached: true (target_reached).
- Q: `k_Q = 5` accepted swaps, target reached: true (target_reached).
- R: applicable = false (no input-labeled -> thrust edges in the biological graph).

### Transfer matrix T, before (biological)

| population | foodBearing | foodDistance | hazardBearing | hazardDistance | forwardClearance | leftClearance | rightClearance | speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thrust | 0.0101 | 0.0049 | -0.0170 | 0.0021 | 0.0022 | 0.0030 | 0.0031 | 0.0015 |
| yaw | 0.0256 | 0.0152 | -0.0008 | 0.0053 | 0.0067 | 0.0057 | 0.0067 | 0.0053 |
| brake | 0.0612 | 0.0433 | 0.0298 | 0.0178 | 0.0156 | 0.0153 | 0.0218 | 0.0220 |

### Transfer matrix T, after P

| population | foodBearing | foodDistance | hazardBearing | hazardDistance | forwardClearance | leftClearance | rightClearance | speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thrust | 0.0105 | 0.0048 | -0.0172 | 0.0020 | 0.0190 | 0.0028 | 0.0948 | 0.0012 |
| yaw | 0.0256 | 0.0152 | -0.0008 | 0.0053 | 0.0064 | 0.0057 | 0.0064 | 0.0053 |
| brake | 0.0612 | 0.0433 | 0.0298 | 0.0178 | 0.0153 | 0.0153 | 0.0243 | 0.0220 |

### Transfer matrix T, after Q

| population | foodBearing | foodDistance | hazardBearing | hazardDistance | forwardClearance | leftClearance | rightClearance | speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thrust | 0.0105 | 0.0050 | -0.0171 | 0.0020 | 0.0190 | 0.0027 | 0.0811 | 0.0016 |
| yaw | 0.0256 | 0.0152 | -0.0008 | 0.0053 | 0.0064 | 0.0057 | 0.0063 | 0.0053 |
| brake | 0.0612 | 0.0433 | 0.0298 | 0.0178 | 0.0153 | 0.0153 | 0.0217 | 0.0220 |

Every entry of T may move, including yaw and brake channels unrelated to the targeted pathway -- the tables
above show the full 3x8 matrix precisely so that is checkable; the category below uses only each graph's
`movementScore`, never a T entry directly.

## Authored results

Empirical p is the `(k+1)/(n+1)` rank statistic against a 100-graph control arm (`03-evaluation.md`'s own rank
statistic): the smallest value it can take is 1/101 ≈ 0.0099, which means the intervention exceeded every one of
the 100 control graphs -- not "near the bottom" of the arm.

| graph | score | percentile in published null | empirical p vs C `(k+1)/(n+1)` | empirical p vs M | empirical p vs MQ |
| --- | --- | --- | --- | --- | --- |
| P | 3.6125 | 85.6% | 0.0099 | 0.0099 | -- |
| Q | 3.4014 | 81.6% | -- | -- | 0.0099 |

- P's category: **Pathway supported**: P's score is at or above the null's 25th percentile and above the 95th percentile of both the C and M distributions.
- Q's channel-specific modifier: **holds** (Q above the null's
  25th percentile and above its own MQ control's 95th percentile).
- Control arms (mean `movementScore`, n=100 each): C p50=-0.2198 p95=-0.1833;
  M p50=0.0142 p95=0.3870; MQ (n=100, k_Q=5) p50=0.3818 p95=0.6084.
- Biological reproduction check: computed -0.2199 vs published
  -0.2199 (matches: true).
- Multiple comparisons disclosed: P vs C, P vs M, Q vs MQ, and (below) 3 trainer seeds --
  none of these comparisons is corrected against the others; each is reported and read on its own predeclared
  terms.

## Trained results

| P trainer seed | score | above C max | above M max | category | vs published trained-null p25 (context only) |
| --- | --- | --- | --- | --- | --- |
| 101 | 65.5338 | no | no | no-specific-effect (neither pathway-supported nor edge-class; the generic-vs-not-supported split is undetermined) | below |
| 202 | 70.2713 | no | no | no-specific-effect (neither pathway-supported nor edge-class; the generic-vs-not-supported split is undetermined) | above |
| 303 | 70.6666 | no | no | no-specific-effect (neither pathway-supported nor edge-class; the generic-vs-not-supported split is undetermined) | above |

- Control arms (mean `movementScore`, trainer seed 101, n=5 each): C max=71.5822; M max=75.4873.
- `trainedRobust`: **true** (every P trainer seed above agrees on the category).
- The published trained null's 25th percentile is context only (percentile resolution 5.0%) and
  does not decide the category -- see the disclosure above. Informally, the finer generic-vs-not-supported split
  this context value would suggest is **not seed-robust**: below at trainer seed(s) 101 and above at trainer seed(s) 202/303.
- Q is not evaluated with trained readouts (authored-decoder only, per the predeclared method above).

## Outcome

Under the **authored decoder**, this study's mechanical outcome is **pathway-supported**, with the
channel-specific modifier **holding**. This authored-decoder
verdict is bound to the hand-written decoder; it does not by itself say what a trained readout finds. Under
**trained readouts**, P shows no advantage over either freshly-trained control arm at all 3 trainer seeds tested (**no-specific-effect (neither pathway-supported nor edge-class; the generic-vs-not-supported split is undetermined)**, robust: true).

## Limitations

- This experiment covers this model only: the authored encoder, rate dynamics, arena, and decoders. It makes no
  biological claim, and a real fly's sensory neurons not synapsing directly onto descending neurons is expected
  anatomy, not a defect.
- The authored decoder is hand-written, not trained and not biology.
- Every claim here is descriptive and bound to this model only; no causal claim is made about the real fly.
- A positive result means the net effect of this accepted swap set, not a single-edge causal effect.
- The method is a greedy targeted search against 100 unrestricted (C) and 100 class-matched (M) random controls;
  it does not prove the targeted edges are individually necessary or sufficient.
- The trained arm compares against only 5 controls per arm (a coarse resolution), and its result depends on the
  trainer seed -- reported as robust only when all three P trainer seeds agree on the category.
- The channel-specific test (Q against MQ) is authored-decoder only; Q was not evaluated with trained readouts.
- The predeclared C/M-max comparison decides pathway-supported/edge-class-effect for the trained decoder, but
  00-overview.md does not say how to report the remaining generic-vs-not-supported split when P does not clear
  the C arm (that needs a trained-null percentile floor, and this study's trained null is context-only).
  'no-specific-effect' is a reporting convention this study's coordinator adopted after the trained scores were
  known (see the disclosure above), not itself a predeclared category; this report does not force an unlicensed
  generic/not-supported label.
- P at trainer seeds 202/303 is compared against C/M control arms trained only at seed 101, so those two
  comparisons mix a graph difference with a trainer-seed difference; only the seed-101 comparison is seed-matched.
- Multiple comparisons (P vs C, P vs M, Q vs MQ, and 3 trainer seeds) are disclosed above
  and are not corrected against each other.
