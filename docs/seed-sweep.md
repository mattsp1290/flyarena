# Seed sweep: descriptive multi-seed results

This is a **descriptive** summary of a fixed-seed sweep over the real,
checked-in MaleCNS-derived artifact (`public/data/malecns-arena-v1.*`) for
every experimental arm the plan defines: biological, the degree-preserving
rewired control, and the disconnected negative control.

**No superiority claim is made here.** Per the plan's product boundary
("Results are descriptive for the POC; no superiority claim is made without
multiple seeds and uncertainty"), the numbers below describe the score
distributions each arm produced under this specific arena/dynamics/encoder/
decoder contract — they are not evidence that one topology performs better
than another. The sample sizes are small, the arena is intentionally simple,
and no statistical test for a difference between modes has been run.

## Methodology

- Script: `scripts/experiments/seed-sweep.ts` (run via `npm run
  experiment:seed-sweep`), driven through the same headless
  `ExperimentRunner` + `createOracleAgentBinding` oracle path the project's
  own unit tests and `scripts/training/export-traces.ts` use — no browser,
  no Worker — and through the exact same `buildGraphBufferForMode` the
  product's own `ExperimentController` calls, not a separately hand-copied
  mode-switch implementation.
- **Integrity-verified inputs**: both artifacts are sha256-verified against
  the committed manifest before use, the same integrity bar the browser
  enforces before a real run is allowed to start (see "Provenance" below for
  the exact hashes this run used).
- **Self-paired runs**: each run pairs an arm against *itself* (e.g.
  biological-vs-biological), not the product UI's own default
  biological-vs-rewired pairing. `world.ts#processContacts` awards a
  contested food pickup to whichever agent already overlaps it first, so
  pairing two *different* topologies in the same run would let each mode's
  measured score be biased by which topology it happened to be racing that
  seed. Self-pairing removes that cross-topology confound while every run
  still uses the exact same seed, world config, encoder, dynamics, and
  decoder the product does — only the topology differs between the three
  groups of runs.
- **20 fixed seeds** (`1000`–`1019`, documented and reproducible — not
  `Math.random()`), each run for the full 90 simulated seconds (2,700 ticks
  at the fixed 30 Hz step, 4 neural substeps per tick, matching the
  product's own `NEURAL_SUBSTEPS_PER_TICK`).
- **Sampling unit: the run (seed), not the arm.** Each run's two arms share
  one world (same seed, same food/hazard positions and motion, and they
  compete for the same contested food pickups) — they are correlated, not
  independent draws. The tables below treat each seed as one independent
  sample, averaging its two arms together first (`n = 20` per mode). A
  secondary, clearly-labeled per-arm breakdown (`n = 40`, correlated pairs)
  is also included for transparency, but is not the primary statistic. Its
  `n` should not be used to judge how precisely a mean is known (e.g. via a
  standard error of `sd / sqrt(n)`) — that would overstate precision, since
  the true number of independent observations is 20, not 40. It is *not*
  the case that the per-arm spread is simply "smaller than it should be":
  averaging two values before computing a statistic reduces dispersion
  regardless of correlation, so the per-arm sample standard deviations
  below are, if anything, larger than the per-run ones.
- Reported statistics: mean, median, and **sample** standard deviation
  (`n − 1` denominator).
- The rewired arm reflects exactly **one** rewiring realization
  (`rewiredArms.seed0`); rewiring-to-rewiring variation across different
  rewiring seeds is not sampled by this study.

## Provenance

This run used:

- Biological artifact sha256 (decompressed): `f1a0f982ffdfba12ecb2206d064c3dc1ceaae7cffa06a22049093d69a79098f7`
- Rewired (seed0) artifact sha256 (decompressed): `7321adeb7fc7a86ad72525007985d846327f3f525d19cd69724c87aaa1935986`
- Arena config fingerprint: `arena-config-v1|fixedDeltaSeconds=0.03333333333333333|halfWidth=12|halfDepth=8|agentRadius=0.35|foodRadius=0.25|hazardRadius=0.6|foodCount=4|hazardCount=2|spawnInset=1|maxSpeed=6|acceleration=9|turnRate=3.141592653589793|rollingDrag=0.7|brakeDrag=8|movementScorePerUnit=0.1|foodScore=10|hazardPenalty=2|sensorRange=24`

These are also recorded in the (gitignored) raw output JSON for every run.

## Results (movement score, the arena's composite score) — per-run, n = 20

| Mode | n (runs) | mean | median | sample sd |
| --- | ---: | ---: | ---: | ---: |
| Biological (measured graph) | 20 | −2.04 | −2.50 | 2.87 |
| Rewired (degree-preserving control) | 20 | 0.70 | 0.94 | 3.69 |
| Disconnected (negative control) | 20 | −3.25 | −3.00 | 1.68 |

### Secondary per-arm breakdown (n = 40, correlated pairs — see Methodology)

| Mode | n (arms) | mean | median | sample sd |
| --- | ---: | ---: | ---: | ---: |
| Biological | 40 | −2.04 | −3.26 | 4.81 |
| Rewired | 40 | 0.70 | −1.02 | 5.58 |
| Disconnected | 40 | −3.25 | −2.00 | 2.89 |

## Results by raw component (per-run, n = 20)

| Mode | food pickups (mean) | hazard contacts (mean) | distance travelled (mean) |
| --- | ---: | ---: | ---: |
| Biological | 0.075 | 1.70 | 6.08 |
| Rewired | 0.25 | 1.73 | 16.48 |
| Disconnected | 0.00 | 1.63 | 0.00 |

The disconnected arm's food pickups and distance travelled are **exactly**
zero for every sample — not merely small. This matches the model contract
directly: with `edgeCount = 0`, no recurrent path exists from any
input-mapped neuron to any output-mapped neuron (see
`tests/unit/experiment-runner.test.ts`'s "disconnected control never
produces a non-zero decoded action" test), so the decoded thrust/yaw/brake
action is always exactly zero and the agent never accelerates away from its
mirrored start position. Its hazard-contact count is still nonzero because
the arena's hazards move independently and can pass through a stationary
agent. This is the same "disconnecting the graph materially changes the
declared neural features" property `tests/e2e/arena.spec.ts` checks directly
in the browser.

## Reproducing this study

```bash
npm run experiment:seed-sweep
```

Writes per-seed raw samples and the full summary (including both the
per-run and per-arm statistics, and the artifact hashes/config fingerprint
above) to `scripts/experiments/out/seed-sweep-results.json` (gitignored —
regenerate on demand rather than diffing a large committed data file). The
script refuses fewer than 20 seeds for its default methodology; pass
`--seeds N` (`N >= 20`) and/or `--out <dir>` to explore a different sweep
without touching this committed summary.

**Measured runtime**: 20 seeds × 3 modes (60 full 90-simulated-second runs)
completed in **74.6 seconds** of wall-clock time on the machine this
document was generated on — well within a reasonable local/CI budget.
