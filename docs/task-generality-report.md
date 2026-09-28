# Task generality of the null and pathway findings

## Question

`docs/rewiring-null-report.md` and `docs/pathway-interventions-report.md` (both on the default foraging task,
`ARENA_CONFIG`) report that biological scores at the bottom of the 500-rewiring null, and that the
clearance->thrust pathway intervention P is pathway-supported. This experiment tests whether those two findings
hold beyond the default task, using four predeclared `ArenaConfig` variants.

## Task variants

| id | Changes | Intent |
| --- | --- | --- |
| `hazard-heavy` | `hazardCount 4`, `hazardPenalty 6` | Scoring dominated by avoidance |
| `sparse-food` | `foodCount 1`, `halfWidth 18`, `halfDepth 12` | Long-range search |
| `no-movement` | `movementScorePerUnit 0` | Score only from food and hazards, which removes the distance term thrust feeds |
| `crowded` | `halfWidth 8`, `halfDepth 5.5` | Walls are close and clearance signals dominate |

Each variant changes only the listed `ArenaConfig` fields; every other field (including `sensorRange`) is
unchanged from `ARENA_CONFIG`, and each passes `retainArenaConfig`'s validation, including the disk-packing
capacity check.

## Measured sensor saturation

`00-overview.md`'s own "Consequences" paragraph predicted, from the `sensorRange`/arena-size formula alone
(before WP2 measured anything), that wall clearance would saturate near its reachable maximum in `sparse-food`
and be compressed in `crowded`, and that `foodDistance` would saturate "far more often" in `sparse-food`'s
enlarged arena. WP2 (`scripts/null/task-clearance.ts`, 10 seeds x 1800 ticks per task) measured the real
distributions instead of relying on that formula. The table below is that measurement, quoted directly
(`00-overview.md:52`: "the report quotes the measured values, not a formula"):

| task | channel | p5 | p50 | p95 | max | fraction saturated |
| --- | --- | --- | --- | --- | --- | --- |
| `hazard-heavy` | foodDistance | 0.062 | 0.161 | 0.250 | 0.267 | 0.00% |
| `hazard-heavy` | forwardClearance | 0.242 | 0.497 | 0.655 | 0.729 | 0.00% |
| `hazard-heavy` | leftClearance | 0.253 | 0.361 | 0.619 | 0.678 | 0.00% |
| `hazard-heavy` | rightClearance | 0.319 | 0.389 | 0.500 | 0.594 | 0.00% |
| `sparse-food` | foodDistance | 0.518 | 0.612 | 0.671 | 0.677 | 0.00% |
| `sparse-food` | forwardClearance | 0.462 | 0.592 | 0.936 | 1.000 | 1.16% |
| `sparse-food` | leftClearance | 0.417 | 0.539 | 0.956 | 1.000 | 1.09% |
| `sparse-food` | rightClearance | 0.489 | 0.608 | 0.858 | 1.000 | 0.65% |
| `no-movement` | foodDistance | 0.087 | 0.171 | 0.248 | 0.264 | 0.00% |
| `no-movement` | forwardClearance | 0.219 | 0.534 | 0.664 | 0.704 | 0.00% |
| `no-movement` | leftClearance | 0.260 | 0.353 | 0.616 | 0.684 | 0.00% |
| `no-movement` | rightClearance | 0.319 | 0.378 | 0.492 | 0.621 | 0.00% |
| `crowded` | foodDistance | 0.068 | 0.104 | 0.183 | 0.195 | 0.00% |
| `crowded` | forwardClearance | 0.147 | 0.304 | 0.428 | 0.489 | 0.00% |
| `crowded` | leftClearance | 0.152 | 0.240 | 0.400 | 0.450 | 0.00% |
| `crowded` | rightClearance | 0.215 | 0.279 | 0.378 | 0.461 | 0.00% |

## Predeclared outcome rules

- **Null holds:** biological is below the task's rewired-null 25th percentile (the authored null, 500 rewirings, seeds `30001-30100`, T 1800).
- **Pathway generalizes:** that task's P score passes the same "pathway-supported" rule as the original (at or above the task's null 25th percentile and above the 95th percentile of both C and M), using the existing P/C/M graphs. Those graphs were built from the task-independent transfer matrix, so they are reused unchanged.
- **Trained (predeclared taxonomy, made mechanical here):** per P trainer seed, with `cMax` and `mMax` the maxima of the 5 C and 5 M trained scores: `pathway-supported` if P > `cMax` and P > `mMax`; `edge-class-effect` if P > `cMax` and P <= `mMax`; `no-specific-effect` if P <= `cMax`. The published trained null is context only and never decides the category. `trainedRobust` holds only if all 3 seeds give the same category.
- **Degenerate guard:** a task's authored result is **not categorized** (reported as `degenerate`) if the null, the C arm, or the M arm has IQR below the existing `DEGENERATE_IQR_THRESHOLD`. The same guard applies to the trained side: if the 5-run trained C or M arm has IQR below the threshold (or all its scores are equal), that task's trained result is `degenerate` and gets no category.
- **Overall:** "general" if the null holds and the pathway generalizes in all non-degenerate tasks, with at least 3 of 4 tasks non-degenerate; "task-dependent" otherwise, listing per task. The authored and trained results are reported separately.

**Trained-side "generalizes" (this study's own operational rule, not verbatim from `00-overview.md`):** the
"Overall" bullet above only defines "null holds" and "pathway generalizes," both authored-side concepts, and never
states a trained-side predicate in those terms. This report applies the same 3-of-4-non-degenerate threshold to
the trained side, reading a non-degenerate task as "generalizing" only if it is `trainedRobust` (all 3 P trainer
seeds agree) **and** that agreed category is `pathway-supported` -- never a single seed's hit alone. This rule
has not been ratified in `00-overview.md` itself; see `scripts/null/task-generality-report.ts`'s own comment on
`computeOverall`'s trained call site for the identical disclosure in code.

## Per-task results

### `hazard-heavy` (hazardCount=4, hazardPenalty=6)

- Null: biological -10.86 (1.8%) vs 25th percentile -9.27 (median -8.19) -- holds.
- Pathway: **pathway-supported** (generalizes). P=-5.64, C p95=-10.86, M p95=-9.26. Q-vs-MQ channel-specific (Q=-5.90): holds.
- Trained: **no-specific-effect** (not robust (1 of 3 seeds reach pathway-supported -- a single-seed hit, not a robust finding)). Per seed: seed 101: no-specific-effect, seed 202: pathway-supported, seed 303: no-specific-effect.

### `sparse-food` (halfWidth=18, halfDepth=12, foodCount=1)

- Null: biological -0.42 (0.0%) vs 25th percentile 0.94 (median 1.30) -- holds.
- Pathway: **pathway-supported** (generalizes). P=1.11, C p95=-0.42, M p95=-0.06. Q-vs-MQ channel-specific (Q=1.15): holds.
- Trained: **no-specific-effect** (robust). Per seed: seed 101: no-specific-effect, seed 202: no-specific-effect, seed 303: no-specific-effect.

### `no-movement` (movementScorePerUnit=0)

- Null: biological -0.62 (0.0%) vs 25th percentile 0.60 (median 1.10) -- holds.
- Pathway: **degenerate** (C arm IQR below threshold) -- not categorized. P=1.82, C p95=-0.58, M p95=-0.08. Q-vs-MQ channel-specific (Q=1.70): holds.
- Trained: **no-specific-effect** (robust). Per seed: seed 101: no-specific-effect, seed 202: no-specific-effect, seed 303: no-specific-effect.

### `crowded` (halfWidth=8, halfDepth=5.5)

- Null: biological -1.88 (0.2%) vs 25th percentile 0.17 (median 0.88) -- holds.
- Pathway: **pathway-supported** (generalizes). P=1.59, C p95=-1.82, M p95=-1.26. Q-vs-MQ channel-specific (Q=1.14): holds.
- Trained: **no-specific-effect** (not robust (1 of 3 seeds reach pathway-supported -- a single-seed hit, not a robust finding)). Per seed: seed 101: no-specific-effect, seed 202: no-specific-effect, seed 303: pathway-supported.

## Overall

- **Authored:** **general** (3 of 4 tasks non-degenerate).
- **Trained:** **task-dependent** (4 of 4 tasks non-degenerate).

The default task's own separately-published results are context only, not a fifth study task: authored category
**pathway-supported**; trained category **no-specific-effect** (robust across trainer seeds). A non-robust trained hit on a task variant above is never framed as generalizing this default-task result.

## Limitations

- Measured `foodDistance` never saturates in any task (fractionSaturated 0 in all 4 tasks; the largest measured value is 0.677 in `sparse-food`, still well under the sensor's 1.0 ceiling).
- Measured wall clearance saturates only in: `sparse-food` (forwardClearance 1.16%, leftClearance 1.09%, rightClearance 0.65%); every other task shows 0% wall-clearance saturation on every channel.
- `no-movement`'s authored pathway category is `degenerate` (C arm IQR below the predeclared threshold), while its Q-vs-MQ channel-specific result (holds) is independently valid and is not conflated with the degenerate P/C/M category.
- The trained decoder's predeclared C/M-max comparison can rule pathway-supported and edge-class-effect in or out (P above/below the max of the freshly-trained 5-graph C/M arms at trainer seed 101), but 00-overview.md does not say how to report the finer generic-rewiring-effect vs not-supported split when P does not clear the C arm: that split needs a trained-null percentile floor, and this study's only trained null (rewiring-null-v1.json's published n=20 sample) is reported for context only, per 00-overview.md, not as a decisive threshold. 'no-specific-effect' is a reporting convention this study's coordinator adopted after the trained scores were known (methodology review, 2026-09-26), to avoid forcing an unlicensed generic/not-supported label -- it does not change which predeclared comparison P passed or failed, only how the undecidable case is named.
- This experiment covers this model only: config variants of the existing arena, no new physics, sensors, or reward code, and no claim about fly behavior.
- `sensorRange` is unchanged (24) in every task, so observation scaling is unchanged; the measured clearance and `foodDistance` distributions still differ per task -- see "Measured sensor saturation" above for the quoted per-channel percentiles and `fractionSaturated` values, not a formula.
- `no-movement` severs the direct channel through which thrust earned score (distance x `movementScorePerUnit`, set to 0); any pathway result there reflects only thrust's indirect effect on food and hazard outcomes.
- P/C/M/Q/MQ graphs were selected once on the default task's task-independent transfer matrix and reused unchanged across all four tasks (checked against the default study's own `sources.indexSha` above) -- this study never re-selects them per task.
- Degenerate tasks (authored or trained) are not categorized at all, and are excluded from the "at least 3 of 4 non-degenerate" overall verdict's eligible set.
- Trained results use only 5 freshly-trained controls per arm (a coarse resolution) and are reported robust only when all 3 P trainer seeds agree; a category reached at fewer than all 3 seeds is disclosed as a split next to that category, never presented as if it were the robust finding, and is never treated as generalizing the default task's own separately-published trained result.
- The null and pathway comparisons are repeated across 4 tasks x 2 decoders (8 combinations) without correction for multiple comparisons; each is reported and read on its own predeclared terms.
- Everything here is descriptive and bound to this model only; no causal claim is made about the real fly.
