# Counterfactual workbench contract

This browser experiment uses the same pinned MaleCNS-derived graph, 30 Hz world,
sensors, authored decoder and neural substep constant as the arena. It tests a
model intervention, not a biological animal. The existing DGX sandbox remains a
separate authored 64-unit model with different physics and score units.

## Matched past and controlled futures

For each seed, the left agent runs the selected graph with the authored decoder;
the right agent receives zero action but still participates in world physics.
After the requested warmup, the experiment copies every world field (including
RNG state, contacts, accumulated scores and previous transforms) and all neural
rates into three independent branches. This is a seeded restart, not a capture
of the live two-agent arena or its trained decoder.

Baseline and sham follow the identical numerical path. The lesion branch zeros
the selected rates before its first recurrent scatter and after every neural
substep, before motor aggregation. Every branch retains the same graph weights,
sensors, decoder and world rules. Targets are eight authored input-channel groups,
bridge neurons with neither input nor output assignment, or descending output
neurons. Group membership and decimal body IDs are exported; these are not
anatomical-region labels. Empty groups cannot run.

All score components are increments after the fork. Pair each silenced score
with baseline on that seed, then report their mean difference and descriptive
normal interval `mean ± 1.96 × sampleSD / sqrt(n)`. These intervals are rough with
small samples and carry no significance or biological inference claim. The sham
must agree exactly on every outcome and replay frame or the engine rejects the
experiment. Zero and negative effects are valid.

World RNG is equal at the fork. Different contacts and pickups can later cause
different food placements and random-number consumption; separate replay panels
show each branch's own environment. Each saved frame contains a real world snapshot
and the score components at that exact tick. The timeline displays the selected
frame's score minus the fork score, never a reconstructed or final score.

## Bounds and lifetime

Settings: unsigned 32-bit base seed, 4–16 consecutive seeds with unsigned wrap,
0–300 warmup ticks, and 30–300 future ticks. Replay includes fork and endpoint and
at most 61 frames per branch/seed. No training or seed selection occurs. A
single request owns a dedicated Worker, with a 30-second preparation deadline,
120-second no-progress timeout and five-minute overall ceiling. Cancel or leaving
the workbench terminates that Worker; a new run starts fresh. Incomplete results
are never published as completed.

The graph loader verifies compressed/uncompressed hashes. Evidence records the
actual binary hash used, including a separately computed hash for the runtime-derived
disconnected graph. Source artifact identity, body IDs, model version, config
fingerprint, K, seed list and decoder/opponent conventions accompany outcomes.

## Reproduction

```bash
npm run experiment:counterfactual -- --seed 17 --seeds 8 --warmup 120 --horizon 180 --target bridge --topology biological --output /tmp/probe.json
npm run experiment:counterfactual -- --verify /tmp/probe.json
npm run experiment:counterfactual -- --compare-numerical /tmp/browser-probe.json
```

The CLI accepts exports up to 16 MiB and verifies graph, model and target identity
before simulation. Default verification rejects any mismatch in deterministic
evidence; runtime provenance/timing is deliberately excluded. Same-runtime repeats
must be exact. Browser architecture may be unknown and is not inferred from its
user-agent. Node and Chromium can differ even on the same machine: a local probe
observed hazard-coordinate differences up to `1.11e-16`, with identical outcomes.

`--compare-numerical` is an explicit diagnostic, not exact verification. Identity,
structure and discrete values remain exact. Only continuous result fields permit
absolute error ≤1e-6 or relative error ≤1e-6, with at most eight numeric leaves above
both 1e-13 absolute and relative noise floors. It reports maximum errors and the
inexact-leaf count, rejects over-budget or discrete differences, and labels a
passing inexact result “numerically close; not exact reproduction.” No automatic
fallback or rounding hides an exact-verification failure. These bounds do not
guarantee reproduction of contact boundaries across platforms.
