# Rewiring null — authored-decoder evaluation

Where the measured biological MaleCNS topology falls among 500 degree-preserving
rewirings of the same graph, scored under identical dynamics, encoder, decoder, and held-out seeds.

**What "authored decoder" means.** The authored decoder is a fixed, hand-written mapping from
output-neuron rates to motor actions (`src/lib/arena/actions.ts`'s `decodeAction`, invoked via
`scripts/training/episode.ts`'s `aggregateOutputs -> decodeAction` pipeline) — it is not biological,
not trained, and not derived from the connectome beyond reading rates off anatomically-labeled output
neurons (see `docs/model-ledger.md`, which labels this **Authored** and separately states the product
"must not... imply that authored behavior is biological"). This
report describes how this specific hand-authored decoder, this rate-model dynamics, and this arena
interact with the biological graph's topology versus 500 rewirings of it — it is
not a claim about the real fly's neural function or behavior, and it makes no claim that any rewiring's
topology is causally "worse," or that any other rewiring is "better," than the biological one.

## Method

Every graph below (biological, disconnected, and each of the 500 rewired graphs)
is scored by `scripts/training/episode.ts`'s `runEpisode`: the authored decoder drives the left agent,
the right agent is parked (always the zero action), over the same 100 held-out seeds
(`30001..30100`), `T = 1800`
ticks, `K = 4` neural substeps per tick. Degree-preserving rewiring comes only from
`scripts/data/rewire.py`'s `rewire_graph` (rewiring seeds `0..499`, seed 0
being the shipped control arm). A graph's score is the mean `movementScore` over its held-out seeds; the
null set `N` is the 500 rewired graphs' mean scores.

## Parameters

| Parameter | Value |
| --- | --- |
| Condition | authored, opponent parked |
| Held-out seeds | 30001..30100 (n=100) |
| Ticks (T) | 1800 |
| Substeps (K) | 4 |
| Rewired graphs | 500 (seeds 0..499) |
| Evaluation shards | 18 |
| Bootstrap resamples | 10000 |
| Bootstrap seed | 1314212940 |
| Histogram bins | 30 |
| Source graph sha256 | `f1a0f982ffdfba12ecb2206d064c3dc1ceaae7cffa06a22049093d69a79098f7` |
| Rewiring source sha256 | `cacf666d311e00156b389cd273a998f6165d1f58845e2136aaf8a3eb932eed81` |
| Host | arm64, Node v22.22.3 |

## Histogram

Null distribution of graph scores (500 rewired graphs), 30
equal-width bins over `[min(N ∪ {biological, disconnected}), max(N ∪ {biological, disconnected})]`:

| range | count | |
| --- | --- | --- |
| -1.860 to -1.591 | 0 |  |
| -1.591 to -1.322 | 0 |  |
| -1.322 to -1.054 | 0 |  |
| -1.054 to -0.785 | 0 |  |
| -0.785 to -0.516 | 0 |  |
| -0.516 to -0.247 | 0 |  |
| -0.247 to 0.021 | 0 |  |
| 0.021 to 0.290 | 1 | # |
| 0.290 to 0.559 | 1 | # |
| 0.559 to 0.828 | 11 | ###### |
| 0.828 to 1.096 | 14 | ####### |
| 1.096 to 1.365 | 20 | ########### |
| 1.365 to 1.634 | 30 | ################ |
| 1.634 to 1.903 | 53 | ############################ |
| 1.903 to 2.171 | 45 | ######################## |
| 2.171 to 2.440 | 42 | ###################### |
| 2.440 to 2.709 | 75 | ######################################## |
| 2.709 to 2.978 | 60 | ################################ |
| 2.978 to 3.246 | 41 | ###################### |
| 3.246 to 3.515 | 27 | ############## |
| 3.515 to 3.784 | 26 | ############## |
| 3.784 to 4.053 | 19 | ########## |
| 4.053 to 4.321 | 18 | ########## |
| 4.321 to 4.590 | 10 | ##### |
| 4.590 to 4.859 | 2 | # |
| 4.859 to 5.128 | 2 | # |
| 5.128 to 5.396 | 1 | # |
| 5.396 to 5.665 | 0 |  |
| 5.665 to 5.934 | 1 | # |
| 5.934 to 6.203 | 1 | # |

## Results

| Quantity | Value |
| --- | --- |
| Biological score (95% CI) | -0.2199 (-0.8540, 0.4527) |
| Disconnected score (95% CI) | -1.8600 (-2.2200, -1.5200) |
| Rewired-seed-0 score (95% CI) | 1.0215 (0.3048, 1.7657) |
| Null mean | 2.5653 |
| Null median | 2.5378 |
| Null std | 0.9370 |
| Null 2.5–97.5% | 0.8171 .. 4.4144 |
| Null IQR | 1.2463 |
| Biological empirical percentile in null | 0.0% |
| Rank statistic p_low | 0.0020 |
| Rank statistic p_high | 1.0000 |
| Paired biological − rewired-seed-0 (mean diff, 95% CI) | -1.2414 (-2.0176, -0.4735) |

`p_low`/`p_high` are rank-based descriptive statistics `(k_below + k_equal + 1)/(|N| + 1)` and
`(|N| - k_below + 1)/(|N| + 1)` — reported without significance language, per this study's non-goals.

**Reconciling with [`docs/seed-sweep.md`](seed-sweep.md).** That study reports a 20-seed, two-agent, self-paired
sweep (`T=2700`) where rewired-seed-0 outscored biological (mean 0.70 vs. −2.04). This report's paired
biological − rewired-seed-0 comparison above, run under this study's different condition (single-agent,
opponent parked, 100 seeds, `T=1800`), agrees in direction — biological scores lower here too
(mean diff -1.2414, 95% CI -2.0176 to -0.4735) —
but the two studies differ in agent/opponent condition, tick count, and seed count (and seed set), so this
is corroborating evidence under a related-but-distinct condition, not a replication of the same measurement.

## Trained-readout sample

Where the biological MaleCNS topology's *trained* score — a readout CEM-trained specifically for that
topology, not the fixed hand-authored mapping the sections above use — falls among 20
rewired topologies, each given its own readout trained with the identical production CEM configuration
`flyarena-bigq` used for its own biological replicas.

**Condition and config.** trained, opponent parked. `T = 1800`, `K = 4`,
`D = 48`, held-out seeds `30001..30100`
(n=100) — the same held-out seeds the authored null above and
[the trained-readout report](trained-readout-report.md) both use. Every rewired readout is trained at the
single trainer seed `replicaSeed = 101` (isolating topology from trainer-seed
variance, per this study's key decisions), with the exact CEM config copied from the merged
`flyarena-bigq` manifest (commit `69b610d4a9da11b12a7ac180997e702cf9fd2a4f`): population 128, elites 32, generations 150, alpha 0.7, stdFloor 0.02, initStd 0.5, E=16. Rescored by evaluator git rev
`23c324da56dd7220972dc747447496b98de12e08` — the same TS `runEpisode` authoritative path the authored null
above uses, so every number in this section shares one evaluator revision with every other number in this
section, never a PyTorch-side validation fitness.

### Rewired trained scores (n=20)

| seed | trained score (95% CI) |
| --- | --- |
| 0 | 77.5669 (73.6120, 81.5604) |
| 1 | 122.8570 (117.7549, 127.7131) |
| 2 | 63.7140 (59.8406, 67.7208) |
| 3 | 74.4400 (69.8391, 79.1383) |
| 4 | 71.1723 (67.1730, 75.2335) |
| 5 | 85.0891 (80.9429, 89.0783) |
| 6 | 67.9147 (64.1074, 71.8472) |
| 7 | 75.5309 (70.9921, 80.1535) |
| 8 | 72.9251 (68.8233, 77.1131) |
| 9 | 71.5533 (67.6914, 75.5193) |
| 10 | 73.6187 (69.1414, 78.0363) |
| 11 | 75.6174 (71.1762, 80.0592) |
| 12 | 80.2122 (76.1720, 84.2526) |
| 13 | 75.4194 (70.8880, 79.9064) |
| 14 | 68.2911 (64.6110, 71.8715) |
| 15 | 67.6136 (63.5480, 71.6205) |
| 16 | 69.3956 (65.2115, 73.7358) |
| 17 | 69.7696 (65.9172, 73.5896) |
| 18 | 73.8657 (69.5527, 78.1792) |
| 19 | 75.3015 (70.3144, 80.4141) |

### Biological trained scores (per trainer seed)

| trainer seed | trained score (95% CI) |
| --- | --- |
| 101 | 62.9627 (59.0038, 67.3063) |
| 202 | 72.2304 (67.6313, 76.9528) |
| 303 | 62.7420 (58.3866, 67.4470) |

### Results

| Quantity | Value |
| --- | --- |
| Null (rewired trained) mean | 75.5934 |
| Null median | 73.7422 |
| Null std | 11.8110 |
| Null 2.5–97.5% | 63.7140 .. 122.8570 |
| Null IQR | 5.7613 |
| Biological (trainer seed 101) percentile among the 20 rewired trained scores | 0.0% |
| Percentile resolution (1/n) | 5.0% |
| Rank statistic p_low | 0.0476 |
| Rank statistic p_high | 1.0000 |
| Wall time | 124.5s |
| Per-episode time | 54.2 ms |

With only 20 rewired replicas, the percentile above has a resolution of only
5.0%: one more or fewer rewired replica scoring below biological shifts it
by a full 5.0% step. This is a much coarser distribution than the authored
null's 500-replica, 0.2%-resolution percentile
above, and percentile differences finer than 5.0% are not meaningfully
distinguishable at this sample size.

### Trainer-seed variance context

Biological trained scores across the 3 `flyarena-bigq` replicas (trainer seeds
101/202/303) span
`62.7420` to `72.2304`
(range `9.4884`) — **trainer-noise variance at fixed (biological) topology -- NOT comparable to the null's topology variance at a fixed trainer seed (bioPercentile, above); no overlap-based conclusion may be drawn from comparing the two.** — at the
*same* biological topology. For context on how large this kind of noise alone can be: the merged
[`trained-readout-v1.manifest.json`](../public/data/trained-readout-v1.manifest.json)'s recorded CUDA
rerun of the shipped biological replica moved TS held-out `trained` fitness by
`6.3540` (rerun minus original, that report's sign convention) —
0.67x the trainer-seed spread recorded above (see that report's own Limitations for the CI comparison) — a magnitude comparison offered only as context for how large trainer-seed/run-to-run noise can be, not
a claim that the two numbers should match. **This spread is not comparable to the 0.0%
percentile above**: the spread measures trainer-seed/run-to-run noise at *fixed* topology; the percentile
measures where one topology (biological, at trainer seed 101) falls among
20 different topologies, each at the *same* one trainer seed. Whether these two
numbers happen to overlap, and neither's size relative to the other, supports any conclusion about topology
"mattering more or less" than trainer-seed noise.

## Limitations

- Scores come from a **single-agent condition with the opponent parked**, on the same held-out seeds and
  tick count as [the trained-readout report](trained-readout-report.md) — but that report evaluates a
  different condition: it scores trained readouts (and a `silenced`-readout control) alongside the
  authored decoder, restricted to the biological/rewired-seed-0/disconnected arms, not a null
  distribution over 500 degree-preserving rewirings. Two-agent competitive
  dynamics are not evaluated in either report.
- **One null model** is used: degree-preserving double-edge swaps (`scripts/data/rewire.py`). Other null
  models (weight shuffles within degree, Erdős–Rényi with matched density) are deferred follow-ups.
- **No causal or superiority claim is made.** The percentile and rank statistics above are descriptive: they
  say where the biological graph's score falls among this null model's rewirings under this exact evaluation
  setup, not that biological topology causes or predicts any particular score.
- The trained section above (n=20 rewired replicas) reports the same kind of descriptive percentile/rank statistics as the authored null, at a much coarser 5.0% resolution, and makes no causal or superiority claim either. Its trainer-seed variance context (`bioTrainerSeedSpread`) is trainer-noise variance at fixed topology, explicitly not comparable to its own topology-variance percentile — see that section's own caveats.
