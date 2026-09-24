# Positive Notes

- **Output order does not depend on completion order.** `assembleRaw` (`scripts/null/null-evaluate.ts:378-386`) builds `rewired` by sorting `index.seeds` numerically and looking each result up by `graphId`. It never iterates the `results` Map. Biological and disconnected are pulled by name (`:395-397`). Which shard ran which task, and in what order, cannot affect the bytes written. This is the right structure for the `--shards 1` == `--shards 3` guarantee.

- **No race on `nextTaskIndex`.** `assignNext` (`:276-284`) reads and increments the counter synchronously inside one event-loop callback. Every worker's `message` handler runs on the parent's single thread, so a task cannot be assigned twice or skipped.

- **The two sha256 checks compare the right fields.**
  - `verifyRewiredFiles` checks raw file bytes against `gzipBytes`/`gzipSha256` (`:119-126`).
  - The worker's `expectedSha256` is set to `entry.binarySha256` (`:235`), and `loadVerifiedGraphBinary` checks it against the **decompressed** bytes (`null-worker.ts:70-80`).
  - Biological and disconnected are checked against `index.sourceSha256` in both places. I confirmed in `scripts/data/rewire_batch.py:128` that `sourceSha256` is `sha256_hex(source_binary)` of the decompressed bytes, so both sides mean the same thing.
  - The worker re-checks immediately before `parseGraphBinary`, which closes the gap between the up-front check and the actual load.

- **One code path shared with the browser.** `graphFromTask` (`null-worker.ts:96-101`) goes through `buildGraphBufferForMode` exactly as the browser worker does, including the parse, encode, and re-parse round trip for `disconnected`.

- **The quantile convention is symmetric and matches `bootstrapCI`.** `quantileIndex` (`null-stats.ts:56-57`) uses the same floor / ceil-minus-one rule as `bootstrapCI` (`scripts/training/stats.ts:56-57`). For n=500 it gives indices 12/487 for 2.5/97.5% and 125/374 for the quartiles, the same distance from each end.

- **The rank statistics match the plan exactly,** and the tests' hand-computed values are correct. I re-derived:
  - `[1..5]` with bio=3: `pLow = pHigh = 4/6`;
  - bio below all of `[10,20,30]`: `pLow = 1/4`, `pHigh = 1`;
  - bio above all: `pLow = 1`, `pHigh = 1/4`;
  - `nullSummary(1..10)`: q1 index 2 gives 3, q3 index 7 gives 8, IQR 5;
  - the ε test: n=8, indices 2 and 5, IQR = ε.

- **Bootstrap streams seeded per label, with no collisions.** The labels `biological`, `disconnected`, `rewired-${seed}`, and `paired|biological-vs-rewired-seed0` are distinct, and `conditionRng` hashes `${seed}|${label}`. Each CI depends only on (seed, label, data), so adding or removing a rewired graph does not move any other graph's CI.

- **No clock or environment in the deterministic outputs.** `host` is captured once, at evaluation time, and copied through by `null-report.ts` (line 275). Timing (`performance.now`) goes only to stdout. `JSON.stringify` on insertion-ordered object literals is byte-stable.

- **Early checks keep a long run from failing late.** The rewired files and the biological source are all checked before any `fork` (`runNullEvaluate`, `:408-412`), so a stale or corrupt batch fails in seconds.

- **The published artifact is internally consistent.** `kBelow=0` gives `pLow=1/501=0.001996` and `pHigh=1`. The manifest sha matches the file. `sourceGraphSha256` equals the manifest's `binarySha256`. Rewired seed 0's gzip sha equals the shipped `rewiredArms.seed0`.

- **The shard test runs the real mechanism.** `null-evaluate.test.ts:130-184` launches the actual CLI with `--import tsx`, which exercises `fork` with inherited `execArgv`. It uses 7 tasks over 3 shards, enough for sharding to matter, rather than a mock.
