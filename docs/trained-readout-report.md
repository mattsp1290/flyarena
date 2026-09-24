# Trained-Readout Evaluation Report

## Method

Each evaluated arm (biological, rewired, disconnected) is scored on 100 held-out seeds (seeds 30001–30100), running 1800 ticks at 4 neural substeps per tick, with the opponent parked (zero action) unless stated otherwise in the Side-by-side section.

Three decoder conditions are reported per arm/replica:

- **trained**: the readout loaded from that replica’s `theta_final`.
- **authored**: the current `aggregateOutputs → decodeAction` path (no trained readout).
- **silenced**: the trained readout with its input vector forced to zero every tick — a circuit-silenced control.

For every condition: mean, median, population standard deviation, and a 95% bootstrap confidence interval of the mean (10,000 seeded resamples, bootstrap seed 1163280716). For every arm pair and every trained-vs-authored/trained-vs-silenced comparison: a paired difference on the same seeds, with its own 95% bootstrap CI. No significance or superiority language is used beyond these confidence intervals.

Graph source: `artifact` (`public/data/malecns-arena-v1.bin.gz`), sha256 `ce247668a63df9a84d8e9969227de7b85d5eba679c45821644812ce0740dfb90`.

## Parameter accounting

D (input size, output-neuron count) is gated equal across arms at export time (`export-arms.ts`’s node-set gate). H (hidden size) and parameter count are taken from each arm’s replicas, which share one training config except `arm`/`replica-seed` and are therefore expected equal across arms as well.

| Arm | D | H | Parameter count |
| --- | --- | --- | --- |
| biological | 48 | 16 | 835 |
| rewired | 48 | 16 | 835 |
| disconnected | 48 | 16 | 835 |

## Results

Per arm, per replica: trained, authored, and silenced condition statistics.

### biological

| Condition | Replica | n | Mean | Median | Std | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| authored | — | 100 | -0.2199 | -1.4084 | 3.2801 | [-0.8256, 0.4459] |
| trained | 101 | 100 | 62.9627 | 61.2587 | 21.1832 | [58.9467, 67.4333] |
| silenced | 101 | 100 | 17.2999 | 16.3599 | 5.4659 | [16.2399, 18.3799] |
| trained | 202 | 100 | 72.2304 | 68.8925 | 23.8183 | [67.7213, 76.9312] |
| silenced | 202 | 100 | 34.8270 | 31.3270 | 9.8199 | [32.9270, 36.7670] |
| trained | 303 | 100 | 62.7420 | 58.0923 | 23.1341 | [58.3390, 67.3297] |
| silenced | 303 | 100 | 18.4677 | 17.7477 | 5.2423 | [17.4477, 19.4877] |

### rewired

| Condition | Replica | n | Mean | Median | Std | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| authored | — | 100 | 1.0215 | 0.6155 | 3.7262 | [0.3124, 1.7728] |
| trained | 101 | 100 | 68.7238 | 67.9244 | 21.5373 | [64.4804, 73.0299] |
| silenced | 101 | 100 | 37.7097 | 34.0497 | 11.7203 | [35.5097, 40.0097] |
| trained | 202 | 100 | 74.0673 | 70.4811 | 21.6350 | [69.8136, 78.3595] |
| silenced | 202 | 100 | 21.3649 | 20.4649 | 6.6895 | [20.1049, 22.7049] |
| trained | 303 | 100 | 68.0224 | 66.0900 | 23.5143 | [63.4660, 72.6126] |
| silenced | 303 | 100 | 35.4800 | 33.0200 | 8.8673 | [33.7600, 37.2000] |

### disconnected

| Condition | Replica | n | Mean | Median | Std | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| authored | — | 100 | -1.8600 | -2.0000 | 1.7945 | [-2.2200, -1.5200] |
| trained | 101 | 100 | 35.5588 | 33.3788 | 9.6170 | [33.7588, 37.4788] |
| silenced | 101 | 100 | 35.5588 | 33.3788 | 9.6170 | [33.7188, 37.4588] |
| trained | 202 | 100 | 35.7812 | 33.4212 | 10.7680 | [33.8012, 38.0212] |
| silenced | 202 | 100 | 35.7812 | 33.4212 | 10.7680 | [33.7612, 37.9612] |
| trained | 303 | 100 | 36.1877 | 33.4077 | 11.2566 | [34.0477, 38.4877] |
| silenced | 303 | 100 | 36.1877 | 33.4077 | 11.2566 | [34.0077, 38.4677] |

## Paired differences

Within-arm: trained vs. authored, and trained vs. silenced, on the same held-out seeds.

| Arm | Replica | Comparison | n | Mean difference | 95% CI |
| --- | --- | --- | --- | --- | --- |
| biological | 101 | trained − authored | 100 | 63.1826 | [59.1578, 67.5040] |
| biological | 101 | trained − silenced | 100 | 45.6629 | [41.4790, 50.0223] |
| biological | 202 | trained − authored | 100 | 72.4503 | [67.7470, 77.3807] |
| biological | 202 | trained − silenced | 100 | 37.4034 | [32.9896, 42.1107] |
| biological | 303 | trained − authored | 100 | 62.9620 | [58.2396, 67.7059] |
| biological | 303 | trained − silenced | 100 | 44.2743 | [39.7669, 48.8133] |
| rewired | 101 | trained − authored | 100 | 67.7024 | [63.4122, 71.9970] |
| rewired | 101 | trained − silenced | 100 | 31.0141 | [26.0191, 35.8287] |
| rewired | 202 | trained − authored | 100 | 73.0459 | [68.6420, 77.5289] |
| rewired | 202 | trained − silenced | 100 | 52.7025 | [48.6440, 56.9648] |
| rewired | 303 | trained − authored | 100 | 67.0009 | [62.5143, 71.6497] |
| rewired | 303 | trained − silenced | 100 | 32.5424 | [27.7834, 37.4682] |
| disconnected | 101 | trained − authored | 100 | 37.4188 | [35.5788, 39.3588] |
| disconnected | 101 | trained − silenced | 100 | 0.0000 | [0.0000, 0.0000] |
| disconnected | 202 | trained − authored | 100 | 37.6412 | [35.5412, 39.8412] |
| disconnected | 202 | trained − silenced | 100 | 0.0000 | [0.0000, 0.0000] |
| disconnected | 303 | trained − authored | 100 | 38.0477 | [35.9277, 40.3277] |
| disconnected | 303 | trained − silenced | 100 | 0.0000 | [0.0000, 0.0000] |

Across arms: paired difference on the same held-out seeds, per reported condition.

| Condition | Replica | Arm A | Arm B | n | Mean difference | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| authored | — | biological | rewired | 100 | -1.2414 | [-2.0220, -0.4853] |
| trained | 101 | biological | rewired | 100 | -5.7611 | [-11.6228, 0.1524] |
| silenced | 101 | biological | rewired | 100 | -20.4098 | [-23.1298, -17.8498] |
| trained | 202 | biological | rewired | 100 | -1.8369 | [-7.8594, 4.1201] |
| silenced | 202 | biological | rewired | 100 | 13.4621 | [11.3421, 15.6421] |
| trained | 303 | biological | rewired | 100 | -5.2804 | [-11.5326, 0.8932] |
| silenced | 303 | biological | rewired | 100 | -17.0123 | [-19.2123, -14.8923] |
| authored | — | biological | disconnected | 100 | 1.6401 | [1.0185, 2.3151] |
| trained | 101 | biological | disconnected | 100 | 27.4039 | [23.3508, 31.6471] |
| silenced | 101 | biological | disconnected | 100 | -18.2590 | [-20.5590, -16.0390] |
| trained | 202 | biological | disconnected | 100 | 36.4492 | [31.8549, 41.1284] |
| silenced | 202 | biological | disconnected | 100 | -0.9542 | [-3.4542, 1.4258] |
| trained | 303 | biological | disconnected | 100 | 26.5543 | [21.6577, 31.7152] |
| silenced | 303 | biological | disconnected | 100 | -17.7199 | [-20.2799, -15.2199] |
| authored | — | rewired | disconnected | 100 | 2.8815 | [2.1140, 3.6970] |
| trained | 101 | rewired | disconnected | 100 | 33.1650 | [28.6179, 37.7135] |
| silenced | 101 | rewired | disconnected | 100 | 2.1509 | [-0.5291, 4.9309] |
| trained | 202 | rewired | disconnected | 100 | 38.2861 | [33.7270, 42.8545] |
| silenced | 202 | rewired | disconnected | 100 | -14.4164 | [-16.9964, -11.9164] |
| trained | 303 | rewired | disconnected | 100 | 31.8347 | [26.8233, 36.8986] |
| silenced | 303 | rewired | disconnected | 100 | -0.7077 | [-3.6077, 1.9723] |

## Side-by-side

The shipped default view: both agents driven, same seeds, biological vs. rewired.

### trained-side-by-side

| Replica | Left arm | Left n | Left mean | Left 95% CI | Right arm | Right n | Right mean | Right 95% CI | Left − right mean diff | Left − right 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | biological (left) | 100 | 63.0766 | [59.1474, 67.1033] | rewired (right) | 100 | 67.5310 | [63.0363, 72.0985] | -4.4544 | [-9.8712, 0.8033] |
| 202 | biological (left) | 100 | 73.0765 | [69.0152, 77.2770] | rewired (right) | 100 | 81.0749 | [76.4916, 85.7314] | -7.9983 | [-13.8734, -2.1428] |
| 303 | biological (left) | 100 | 58.7777 | [54.5361, 63.2463] | rewired (right) | 100 | 70.5033 | [65.8953, 75.2032] | -11.7256 | [-18.3076, -5.3876] |

### authored-side-by-side

| Replica | Left arm | Left n | Left mean | Left 95% CI | Right arm | Right n | Right mean | Right 95% CI | Left − right mean diff | Left − right 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | biological (left) | 100 | -0.2202 | [-0.8600, 0.4661] | rewired (right) | 100 | 0.8054 | [-0.0869, 1.7993] | -1.0255 | [-2.2998, 0.1978] |

## Limitations

- Headline per-arm statistics (Results, Paired differences) are measured single-agent, opponent parked — see "What this does not show" below.
- Statistics are descriptive (mean/median/std) plus bootstrap confidence intervals; no significance test or superiority claim is made or implied.
- The `silenced` control forces the trained readout’s input vector to zero every tick; it does not silence the recurrent connectome dynamics themselves.
- Held-out seeds (30001–30100) are the plan's default range, disjoint from its training (1–10000) and validation (20001–20064) seed ranges, but are a fixed, finite sample (not the full seed space).

## Finding: readout input was structurally zero

For disconnected, the graph itself guarantees the readout's input was exactly zero on every tick of every episode: the graph has no edges, and none of its output-assigned neurons is itself directly wired to an input channel, so their rates can never leave their zero starting value, independent of weights or seeds. Every replica’s `trained` and `silenced` scores were also numerically identical on every held-out seed (paired difference exactly 0, 95% CI exactly [0, 0]), consistent with that guarantee. For that arm, `trained` and `silenced` computed the identical function; whatever score the readout achieved came entirely from its learned bias terms — a fixed, input-independent action — never from sensory information. This finding is descriptive only: it does not rank or compare arms against each other.

## What this does not show

The headline per-arm numbers above were measured single-agent with the opponent parked, which differs from the shipped two-agent side-by-side default; see the Side-by-side section of this report for the shipped two-agent condition.
