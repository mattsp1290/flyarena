# Single-neuron lesion atlas

For each of the 1008 neurons in the biological MaleCNS graph and in the shipped
rewired-seed-0 control, the paired change in `movementScore` when that one neuron is silenced for the
whole episode, versus the same episode unlesioned, averaged over 100 held-out seeds.

**What this measures, and what it does not.** Every number below is a property of *this model* --
this hand-authored decoder (`src/lib/arena/actions.ts`'s `decodeAction`), this rate-model dynamics, this
arena, with the opponent parked and evaluated only on held-out seeds -- not a claim about the real fly's
neural function. A neuron's effect being near zero does not mean it is biologically unimportant; it means
silencing it does not change this specific decoder's score under this specific evaluation. See "Limitations"
below.

## Method

Every neuron is lesioned alone (the counterfactual workbench's semantics exactly: rates zeroed before the
substep loop and after every substep, `src/lib/counterfactual/engine.ts`'s `stepBranch`, reproduced by
`scripts/training/episode.ts`'s `lesion` option -- see `.agents/plans/lesion-atlas/01-lesion-episodes.md`),
for the whole episode from tick 0 (no warmup fork, unlike the interactive workbench). The left agent runs the
authored decoder; the right agent is parked (always the zero action). Held-out seeds
`30001..30100` (n=100),
`T=1800` ticks, `K=4` neural substeps per tick -- the same condition the
rewiring-null study uses. The lesion effect for neuron `i` is the mean over seeds of
`movementScore(lesioned i) - movementScore(baseline)`, paired by seed, with a 95% bootstrap CI
(`scripts/training/stats.ts`'s `pairedStats`, 10000 resamples, seed
`1279611721`, one independent RNG stream per neuron per graph). The `p (bootstrap)` column in
the top-effects tables below is a separate two-sided percentile-bootstrap p-value (twice the fraction of
resampled mean differences on the opposite side of 0 from the observed effect, capped at 1), drawn from its
own RNG stream independent of the CI's -- a neuron's CI and p-value can therefore disagree near the
boundary (a CI that just excludes 0 while p is just above 0.05, or the reverse). With
10000 resamples the smallest achievable nonzero p-value is
`1/10000` (from a single resample landing exactly on 0, weighted evenly between
the two sides); `p < ...` below means no resample crossed to, or landed on, the other side, not that the
true p-value is exactly zero.

**Multiple comparisons.** With 1008 simultaneous per-neuron CIs per graph, about
50 are expected to exclude 0 by chance alone even if every neuron's true
effect were exactly 0. Benjamini-Hochberg FDR control at `q = 0.05` (per graph, 1008
tests) marks which neurons' effects survive that correction (`fdrSignificant`); only FDR-surviving neurons
should be read as reliable effects, not every neuron whose raw CI happens to exclude 0.

## Parameters

| Parameter | Value |
| --- | --- |
| Condition | authored, opponent parked, single-neuron lesion |
| Held-out seeds | 30001..30100 (n=100) |
| Ticks (T) | 1800 |
| Substeps (K) | 4 |
| Neurons per graph | 1008 |
| Evaluation shards | 18 |
| Wall time | 7177.3s |
| Per-episode time | 35.6 ms |
| Bootstrap resamples | 10000 |
| Bootstrap seed | 1279611721 |
| FDR q | 0.05 |
| Host | arm64, Node v22.22.3 |

## Results

### Biological

| Quantity | Value |
| --- | --- |
| Baseline score | -0.2199 |
| Median effect | 0.0000 |
| 5th percentile effect | -0.0012 |
| 95th percentile effect | 0.0993 |
| CIs excluding 0 (uncorrected) | 332 (expected by chance: 50.4) |
| FDR-surviving neurons (q=0.05) | 280 |

Top 20 neurons by \|effect\|:

| Index | Body ID | Role | Effect | 95% CI | p (bootstrap) | FDR significant |
| --- | --- | --- | --- | --- | --- | --- |
| 54 | 10671 | bridge | -0.4504 | (-0.9904, 0.0374) | 0.0730 | no |
| 3 | 10045 | descending | -0.4136 | (-0.8943, 0.0447) | 0.0740 | no |
| 4 | 10056 | descending | -0.4111 | (-0.9048, 0.0354) | 0.0778 | no |
| 14 | 10177 | descending | -0.3486 | (-0.9216, 0.2071) | 0.2222 | no |
| 15 | 10180 | bridge | -0.3402 | (-0.8190, 0.0991) | 0.1350 | no |
| 49 | 10581 | bridge | 0.3234 | (-0.0907, 0.7834) | 0.1246 | no |
| 813 | 805722 | sensory | -0.2738 | (-0.7585, 0.1805) | 0.2336 | no |
| 534 | 800473 | bridge | 0.2307 | (-0.1443, 0.6305) | 0.2336 | no |
| 852 | 807736 | sensory | 0.2064 | (-0.1513, 0.5873) | 0.2272 | no |
| 821 | 806053 | sensory | 0.2061 | (0.0052, 0.5071) | < 1e-4 | yes |
| 866 | 808225 | sensory | 0.2041 | (0.0031, 0.5050) | < 1e-4 | yes |
| 819 | 805939 | sensory | 0.2016 | (0.0008, 0.5025) | 0.0106 | yes |
| 536 | 800485 | bridge | 0.2003 | (-0.0003, 0.5010) | 0.1172 | no |
| 31 | 10286 | descending | 0.1994 | (0.0003, 0.4790) | 0.0226 | no |
| 814 | 805733 | sensory | -0.1917 | (-0.6037, 0.1931) | 0.3236 | no |
| 29 | 10281 | descending | -0.1862 | (-0.7218, 0.3298) | 0.4884 | no |
| 848 | 807446 | sensory | 0.1819 | (-0.1781, 0.5633) | 0.3064 | no |
| 251 | 27702 | bridge | 0.1809 | (-0.0397, 0.4813) | 0.1262 | no |
| 2 | 10038 | descending | 0.1784 | (-0.1804, 0.5596) | 0.3564 | no |
| 28 | 10274 | descending | -0.1733 | (-0.6294, 0.2300) | 0.4418 | no |

### Rewired seed 0

| Quantity | Value |
| --- | --- |
| Baseline score | 1.0215 |
| Median effect | 0.0001 |
| 5th percentile effect | -0.0118 |
| 95th percentile effect | 0.1187 |
| CIs excluding 0 (uncorrected) | 486 (expected by chance: 50.4) |
| FDR-surviving neurons (q=0.05) | 448 |

Top 20 neurons by \|effect\|:

| Index | Body ID | Role | Effect | 95% CI | p (bootstrap) | FDR significant |
| --- | --- | --- | --- | --- | --- | --- |
| 31 | 10286 | descending | 1.3940 | (0.3296, 2.5881) | 0.0110 | yes |
| 893 | 902565 | sensory | 0.9844 | (0.0343, 1.9870) | 0.0428 | no |
| 105 | 12601 | descending | 0.9466 | (0.1745, 1.7978) | 0.0098 | yes |
| 33 | 10301 | descending | 0.8759 | (0.1232, 1.6749) | 0.0164 | yes |
| 776 | 803739 | sensory | 0.7724 | (-0.0735, 1.6785) | 0.0744 | no |
| 22 | 10234 | descending | 0.7609 | (-0.0314, 1.6459) | 0.0608 | no |
| 34 | 10360 | descending | 0.7588 | (-0.0035, 1.6059) | 0.0556 | no |
| 28 | 10274 | descending | 0.7086 | (-0.0521, 1.5720) | 0.0672 | no |
| 713 | 801756 | sensory | 0.7074 | (-0.0034, 1.4435) | 0.0432 | no |
| 728 | 801960 | sensory | 0.6106 | (-0.0277, 1.2641) | 0.0540 | no |
| 746 | 802409 | sensory | 0.5626 | (-0.0602, 1.2211) | 0.0802 | no |
| 725 | 801906 | sensory | 0.5450 | (-0.1689, 1.3148) | 0.1402 | no |
| 902 | 903119 | sensory | 0.5437 | (-0.2095, 1.4016) | 0.1724 | no |
| 30 | 10283 | descending | 0.5036 | (-0.0734, 1.2035) | 0.0952 | no |
| 37 | 10417 | descending | 0.4587 | (-0.1864, 1.2060) | 0.1788 | no |
| 43 | 10520 | descending | 0.4446 | (-0.1957, 1.1620) | 0.1846 | no |
| 24 | 10247 | descending | 0.4400 | (-0.1940, 1.1586) | 0.1816 | no |
| 19 | 10221 | descending | 0.4120 | (-0.1614, 1.0321) | 0.1698 | no |
| 18 | 10218 | descending | 0.3918 | (-0.3169, 1.1103) | 0.2832 | no |
| 6 | 10090 | descending | 0.3889 | (-0.1703, 1.0014) | 0.1970 | no |

## Limitations

- **This model only.** No claim is made that any lesioned neuron "controls" a behavior in the real fly --
  only that silencing it changes this authored decoder's score under this evaluation.
- **Single lesions only.** Redundant neurons can each show a near-zero effect alone while a joint lesion of
  both would not; this atlas does not probe pairs or groups (a stated non-goal, follow-up work).
- **Full-episode lesions from tick 0**, unlike the interactive counterfactual workbench's warmup fork -- the
  two are not directly comparable measurements.
- **About 50 of the per-neuron 95% CIs are expected to exclude 0 by
  chance per graph** (1008 simultaneous tests at the nominal 5% rate) -- only
  FDR-surviving neurons (`fdrSignificant`) should be treated as reliable effects.
- **The opponent is parked** for every episode, matching the null studies' single-agent condition, not a
  competitive one.
- **No biological claim.** This describes how this specific hand-authored decoder and rate-model dynamics
  interact with the measured topology, not a measurement of the real fly's neural function.
- **Timing correction.** `.agents/plans/lesion-atlas/02-atlas-computation.md` projected "about 6 min at 18
  shards" for this run. The actual measured wall time was 119.6 minutes
  (7177.3s, see Parameters above) -- roughly 20x longer. The plan's
  estimate treated the rewiring-null study's published "32.1 ms/episode" figure as a serial per-episode cost;
  that figure is itself a wall-clock-per-episode average over an already-18-way-parallel run, not a serial cost,
  so the estimate understated the real wall time by roughly the shard count. A pre-run calibration on the real
  Spark (10 lesions x 2 graphs x 100 seeds, single-process) measured the true serial cost at
  approximately 421-424 ms/episode, correctly projecting the multi-hour range this run's actual wall time falls
  within.
