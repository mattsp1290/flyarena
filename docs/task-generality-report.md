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

## Predeclared outcome rules

- **Null holds:** biological is below the task's rewired-null 25th percentile (the authored null, 500 rewirings, seeds `30001-30100`, T 1800).
- **Pathway generalizes:** that task's P score passes the same "pathway-supported" rule as the original (at or above the task's null 25th percentile and above the 95th percentile of both C and M), using the existing P/C/M graphs. Those graphs were built from the task-independent transfer matrix, so they are reused unchanged.
- **Trained (predeclared taxonomy, made mechanical here):** per P trainer seed, with `cMax` and `mMax` the maxima of the 5 C and 5 M trained scores: `pathway-supported` if P > `cMax` and P > `mMax`; `edge-class-effect` if P > `cMax` and P <= `mMax`; `no-specific-effect` if P <= `cMax`. The published trained null is context only and never decides the category. `trainedRobust` holds only if all 3 seeds give the same category.
- **Degenerate guard:** a task's authored result is **not categorized** (reported as `degenerate`) if the null, the C arm, or the M arm has IQR below the existing `DEGENERATE_IQR_THRESHOLD`. The same guard applies to the trained side: if the 5-run trained C or M arm has IQR below the threshold (or all its scores are equal), that task's trained result is `degenerate` and gets no category.
- **Overall:** "general" if the null holds and the pathway generalizes in all non-degenerate tasks, with at least 3 of 4 tasks non-degenerate; "task-dependent" otherwise, listing per task. The authored and trained results are reported separately.

## Per-task results

### `hazard-heavy` (hazardCount=4, hazardPenalty=6)

- Null: biological -10.86 (1.8%) vs 25th percentile -9.27 (median -8.19) -- holds.
- Pathway: **pathway-supported** (generalizes). P=-5.64, C p95=-10.86, M p95=-9.26.
- Trained: **no-specific-effect** (not robust (1 of 3 seeds reach pathway-supported -- this is a single-seed hit, not a robust finding)). Per seed: seed 101: no-specific-effect, seed 202: pathway-supported, seed 303: no-specific-effect.

### `sparse-food` (halfWidth=18, halfDepth=12, foodCount=1)

- Null: biological -0.42 (0.0%) vs 25th percentile 0.94 (median 1.30) -- holds.
- Pathway: **pathway-supported** (generalizes). P=1.11, C p95=-0.42, M p95=-0.06.
- Trained: **no-specific-effect** (robust). Per seed: seed 101: no-specific-effect, seed 202: no-specific-effect, seed 303: no-specific-effect.

### `no-movement` (movementScorePerUnit=0)

- Null: biological -0.62 (0.0%) vs 25th percentile 0.60 (median 1.10) -- holds.
- Pathway: **degenerate** (C arm degenerate: true, M arm degenerate: false) -- not categorized. P=1.82, C p95=-0.58, M p95=-0.08.
- Trained: **no-specific-effect** (robust). Per seed: seed 101: no-specific-effect, seed 202: no-specific-effect, seed 303: no-specific-effect.

### `crowded` (halfWidth=8, halfDepth=5.5)

- Null: biological -1.88 (0.2%) vs 25th percentile 0.17 (median 0.88) -- holds.
- Pathway: **pathway-supported** (generalizes). P=1.59, C p95=-1.82, M p95=-1.26.
- Trained: **no-specific-effect** (not robust (1 of 3 seeds reach pathway-supported -- this is a single-seed hit, not a robust finding)). Per seed: seed 101: no-specific-effect, seed 202: no-specific-effect, seed 303: pathway-supported.

## Overall

- **Authored:** **general** (3 of 4 tasks non-degenerate).
- **Trained:** **task-dependent** (4 of 4 tasks non-degenerate).

The default task's own separately-published results are context only, not a fifth study task: authored category
**pathway-supported**; trained category **no-specific-effect** (robust across trainer seeds). A single-seed pathway-supported hit on a task variant above is never framed as generalizing this default-task result.

## Limitations

- This experiment covers this model only: config variants of the existing arena, no new physics, sensors, or reward code, and no claim about fly behavior.
- `sensorRange` is unchanged (24) in every task, so observation scaling is unchanged, but the measured clearance and `foodDistance` distributions differ per task -- reported values, not a formula. Wall clearance saturates near its reachable maximum in `sparse-food` and is compressed in `crowded`; `foodDistance` saturates at 1 (uninformative) far more often in `sparse-food`'s enlarged arena.
- `no-movement` severs the direct channel through which thrust earned score (distance x `movementScorePerUnit`, set to 0); any pathway result there reflects only thrust's indirect effect on food and hazard outcomes. Its authored pathway category is `degenerate` (the C control arm's IQR is 0, the predeclared guard), while its Q-vs-MQ channel-specific result is independently valid and is not conflated with the degenerate P/C/M category above.
- P/C/M/Q/MQ graphs were selected once on the default task's task-independent transfer matrix and reused unchanged across all four tasks -- this study never re-selects them per task.
- Degenerate tasks (authored or trained) are not categorized at all, and are excluded from the "at least 3 of 4 non-degenerate" overall verdict's eligible set.
- Trained results use only 5 freshly-trained controls per arm (a coarse resolution) and are reported robust only when all 3 P trainer seeds agree; a category reached at only 1 of 3 seeds is disclosed as a single-seed hit next to that category, never presented as if it were the robust finding, and is never treated as generalizing the default task's own separately-published trained result.
- `no-specific-effect` (trained side) is a reporting convention adopted after the trained scores were known, not a predeclared category (see `docs/pathway-interventions-report.md`).
- The null and pathway comparisons are repeated across 4 tasks x 2 decoders (8 combinations) without correction for multiple comparisons; each is reported and read on its own predeclared terms.
- Everything here is descriptive and bound to this model only; no causal claim is made about the real fly.
