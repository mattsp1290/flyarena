# Review Overview

- **Branch:** `feat/t8dj-null-evaluation` (base `main`, head `9d91bbef1e2d94d36cdf5a577037348258077299`)
- **Date:** 2026-09-24
- **Reviewer:** Maintainability Scout (`maintainability-scout`)
- **Role:** Independent second reviewer focused on subtle edge cases, implicit assumptions, and long-term maintainability.

## Summary

This branch adds WP2 of the rewiring-null plan. `scripts/null/null-evaluate.ts` checks `rewire_batch.py`'s `index.json` and every rewired `.bin.gz` against their sha256 values. It then builds one task per graph: biological, disconnected, and each rewired seed. The tasks are shared across `child_process.fork`ed copies of `scripts/null/null-worker.ts`. Each worker verifies the decompressed sha256 again, builds the graph through the browser's `buildGraphBufferForMode`, and runs `runEpisode` (authored decoder vs. parked opponent) on every held-out seed. The parent writes a raw per-seed `authored.json` sorted by graph id, so the output does not depend on the shard count. `scripts/null/null-report.ts` then uses the pure helpers in `scripts/null/null-stats.ts` (per-graph bootstrap CIs through `conditionStats`/`conditionRng`, null summary, rank statistics, histogram) plus a paired bio-vs-rewired-seed-0 comparison. It publishes `public/data/rewiring-null-v1.json`, adds a `rewiringNull` key to the manifest, and writes `docs/rewiring-null-report.md`.

The overall design is sound. Determinism comes from the canonical task order, not from completion order. Per-label bootstrap RNGs reuse `scripts/training/stats.ts`. Sha256 is checked both up front and in each worker. The problems are at the edges:
- A NaN score is silently turned into `0` when the JSON is written and read back.
- A worker killed by the OOM killer or another external signal is treated as a clean exit.
- `index.json` seeds can be duplicated and nothing rejects it.
- The published `shards` field comes from an unchecked operator flag.
- `--trained` is a dead flag.
- `null-report.ts` has no tests, even though the plan requires the report to be byte-identical across reruns.
- `null-report.ts` writes over the shipped `public/data` artifact and manifest by default. `evaluate.ts` has a guard against this; `null-report.ts` does not.

## Verdict

**REQUEST_CHANGES**. None of these findings breaks the published numbers as committed: I spot-checked the artifact and found 500 contiguous seeds 0..499, a non-degenerate null, and finite values. Still, several Important gaps would let a future rerun publish corrupted statistics or wrong provenance without any error. They should be fixed before this becomes the standing pipeline.

## Stats

`git diff main...HEAD --stat`: 10 files changed, 1635 insertions(+), 0 deletions(-)

| File | + |
| --- | --- |
| docs/rewiring-null-report.md | 101 |
| package.json | 2 |
| public/data/malecns-arena-v1.manifest.json | 4 |
| public/data/rewiring-null-v1.json | 1 (single-line, ~130KB) |
| scripts/null/null-evaluate.ts | 447 |
| scripts/null/null-report.ts | 424 |
| scripts/null/null-stats.ts | 150 |
| scripts/null/null-worker.ts | 136 |
| tests/unit/null-evaluate.test.ts | 218 |
| tests/unit/null-stats.test.ts | 152 |

Commits (`git log main..HEAD --oneline`):

- `9d91bbe` data: publish the authored-decoder rewiring null (500 graphs, arm64 Spark)
- `72b10ed` feat: score the authored-decoder rewiring null with sharded TS evaluation
