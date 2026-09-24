# Code Review Overview

- **Branch:** `feat/t8dj-null-evaluation` (base: `main`, head `9d91bbef1e2d94d36cdf5a577037348258077299`)
- **Date:** 2026-09-24
- **Reviewer:** Correctness Sentinel (`correctness-sentinel`)
- **Role:** Hunts for logic errors, race conditions, and determinism/correctness bugs in the sharded evaluation and statistics pipeline.

## Summary

This branch implements WP2 of the rewiring-null plan. `scripts/null/null-evaluate.ts` builds a canonical task list: biological, disconnected, and one task per rewired seed from `rewire_batch.py`'s `index.json`. Before any shard is forked, it checks each rewired file's gzip size and sha256. It then spreads the tasks across `fork`ed `null-worker.ts` children. Each child re-checks the decompressed sha256, builds the graph through the browser's `buildGraphBufferForMode`, and runs `runEpisode` (authored decoder vs parked) on each held-out seed. The driver rebuilds `authored.json` by walking the seed-sorted index, so completion order does not matter. `null-report.ts` then runs the pure statistics in `null-stats.ts` over that raw file: per-graph bootstrap CIs seeded per label, the null summary, rank statistics, and a histogram. It publishes `rewiring-null-v1.json`, adds a `rewiringNull` key to the manifest, and renders `docs/rewiring-null-report.md`.

The core determinism design holds. Output order never depends on the Map or on which task finished first. The sha256 checks compare the right fields. The quantile and rank formulas match the plan, and the test expectations are correct when worked by hand. `null-stats` and `null-evaluate` tests pass locally (22/22).

I found four problems:
- **Error handling in the fork pool.** If a worker is killed by a signal, the pool treats that as a clean exit (I reproduced this). If a task fails, the rest of the queue still runs to completion before the run fails.
- **The published histogram.** Its `counts` include the biological and disconnected scores (they sum to 502), but the report labels it as the 500-graph null distribution, and WP4 will draw these counts as the null bars.
- **The `shards` field and timing.** The published `shards` value comes from an unchecked CLI flag. The plan requires the measured wall and per-episode time to appear in the report, and that timing is not recorded anywhere.
- **Missing input checks at the publish boundary.** Seeds are not checked for uniqueness. The pairing of held-out seeds and the source-graph/manifest match are not asserted before publishing.

## Verdict

**REQUEST_CHANGES**

The histogram issue affects the published artifact and its downstream consumer. The fork-pool error handling is the main runtime risk in a multi-hour sharded run. Everything else is small or hardening.

## Stats

`git diff main...HEAD --stat`:

```
 docs/rewiring-null-report.md               | 101 +++++++
 package.json                               |   2 +
 public/data/malecns-arena-v1.manifest.json |   4 +
 public/data/rewiring-null-v1.json          |   1 +
 scripts/null/null-evaluate.ts              | 447 +++++++++++++++++++++++++++++
 scripts/null/null-report.ts                | 424 +++++++++++++++++++++++++++
 scripts/null/null-stats.ts                 | 150 ++++++++++
 scripts/null/null-worker.ts                | 136 +++++++++
 tests/unit/null-evaluate.test.ts           | 218 ++++++++++++++
 tests/unit/null-stats.test.ts              | 152 ++++++++++
 10 files changed, 1635 insertions(+)
```

- Files changed: 10
- Lines added / removed: +1635 / -0

Commits (`git log main..HEAD --oneline`):

```
9d91bbe data: publish the authored-decoder rewiring null (500 graphs, arm64 Spark)
72b10ed feat: score the authored-decoder rewiring null with sharded TS evaluation
```

## Verification performed by this reviewer

- `npx vitest run tests/unit/null-stats.test.ts tests/unit/null-evaluate.test.ts`: 2 files, 22 tests, all pass.
- Probed `runShardedEvaluation` with a throwaway stub worker (since removed):
  - A worker that SIGKILLs itself mid-task makes the function **resolve** with only the tasks finished before the kill (`keys=['a']`, tasks `sigkill` and `c` missing).
  - A worker that errors on task 1 of 7 (2 shards, 300 ms tasks) rejects only after **2026 ms**, once the surviving worker has drained the whole queue by itself.
- Spot-checked `public/data/rewiring-null-v1.json`:
  - Its sha256 matches the manifest.
  - `rewired` holds seeds 0..499, in order.
  - `pLow = 1/501` with `kBelow = 0` is consistent.
  - `sourceGraphSha256` equals the manifest's `binarySha256`.
  - Rewired seed 0's `gzipSha256` (`d143d6...`) equals the manifest's `rewiredArms.seed0.gzipSha256`.
  - `bins.counts` sum to **502**, not 500.
