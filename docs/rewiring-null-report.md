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
| Wall time | 1610.3s |
| Per-episode time | 32.1 ms |
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
