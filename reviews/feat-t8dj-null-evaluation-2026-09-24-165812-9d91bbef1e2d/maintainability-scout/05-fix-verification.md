# Fix Verification — Maintainability Scout (follow-up pass)

Commits checked: f482a64 (code), 57b8a33 (data). I read the full current versions of `scripts/null/null-{evaluate,worker,report,stats}.ts`, `tests/unit/null-{report,evaluate}.test.ts`, and `tests/fixtures/null-stub-worker.mjs`, plus the regenerated `public/data/rewiring-null-v1.json`, the manifest, and `docs/rewiring-null-report.md`. I also confirmed three edge cases with a scratch probe (`probe.ts` in the session scratchpad). The probe is not committed.

VERDICT: REQUEST_CHANGES

Every original Important item from both reviewers is addressed. The fix also brought in two new Important issues. Both are small (a few lines each), but they involve exactly the partial-publish and data-loss risks the original findings were about, so I am not signing off yet.

## My original I1–I9

- **I1 NaN/Infinity: Fixed correctly.** `scripts/null/null-worker.ts:114-127` throws inside the worker, naming the graphId and seed. This covers all three score fields. There is no report-side `assertFiniteScores` guard for hand-edited files. That was optional, so I accept its absence.
- **I2 signal exit treated as clean: Fixed correctly.** `scripts/null/null-evaluate.ts:370-380`: `cleanExit = code === 0 || (code === null && aborted)`. An external SIGKILL or OOM kill while not aborted is now an error, and the message names the in-flight graph. The stub test is at `tests/unit/null-evaluate.test.ts:358`. For the residual `code 0` gap, see N3.
- **I3 fail-slow / orphaned children: Fixed correctly.**
  - `abortAll` (`:303-306`) sets `aborted` and kills every tracked child.
  - `assignNext` (`:323`) stops dispatching once `aborted` is set.
  - The `finally` block (`:386-394`) sweeps any children still alive.
  - The completeness check (`:399-402`) runs after collection.
  - I traced the bookkeeping. Node delivers all callbacks on one thread, so `aborted`, `nextTaskIndex`, and `children` cannot race. `finish` is idempotent through `settled`. The `error`-then-`exit` double fire resolves once. A late legitimate result after an abort is stored and then `disconnect()`s, which is harmless because `errors` is already non-empty.
  - A legitimate result is never wrongly rejected. Each child has at most one task outstanding. The worker replies exactly once per task, echoing `task.graphId` and `task.heldOutSeeds.map(...)`, so the checks at `:341` and `:350-352` always pass for real results. The only way to get a second message for one task is `process.send` throwing inside the worker's `try`. That case is correctly treated as a protocol failure.
  - Fail-fast is covered by a timing test (`<900ms`, test `:348`).
- **I4 duplicate/non-integer seeds: Fixed correctly.** `null-evaluate.ts:88-113` rejects:
  - non-integer seeds;
  - negative seeds;
  - a missing `gzipBytes`;
  - duplicates, detected with a `Set`.

  The duplicate and `1.5` cases have tests (`null-evaluate.test.ts:274-325`). A `null` entry would throw a TypeError on `entry.seed` rather than the friendly message, but it still fails closed.
- **I5 unchecked `shards`: Fixed correctly, with one new hazard (N2).**
  - `resolveRunMeta` (`null-report.ts:177-193`) reads `<authored>.run.json`.
  - It throws if there is no sidecar and no `--shards`. There is no more silent 18.
  - The regenerated artifact has `shards: 18` and `timing: {elapsedMs: 1610300, perEpisodeMs: 32.08}`.
  - `sum(bins.counts) = 500`.
  - The manifest sha256 `592446bb…` matches the file's actual sha256. I recomputed it.

  Two leftovers:
  - The `NullEvaluationRaw` doc comment at `null-evaluate.ts:431-433` still says that `null-report.ts` "takes its own `--shards` flag … rather than reading it from here". That is now stale: the report reads the sidecar by default.
  - The sidecar has no link to the `authored.json` it describes, such as a sha. A stale sidecar next to a replaced `authored.json` would be trusted. This is minor.
- **I6 overwrite guard / write ordering: Partially fixed. See N1.**
  - `guardShippedDefault` (`null-report.ts:559-568`) and the manifest graph-sha check (`:580-605`) both run before any write (`:625-628`).
  - The markdown is rendered before any write (`:635`).
  - The atomic temp-file-plus-rename writes are correct (`:61-74`).
  - **However**, the new manifest round-trip validation sits inside `updateManifestWithRewiringNull` (`:287-294`), which runs *after* `atomicWriteFileSync(args.out, …)` (`:638`). See N1.
  - The guard itself is a count floor: `>=500` rewired graphs. A 500-graph run with a different `--held-out-count` or `--ticks` would still overwrite the shipped files. That is acceptable as a heuristic, but it is weaker than the doc comment implies.
- **I7 no `null-report` tests: Fixed, with gaps.** `tests/unit/null-report.test.ts` is substantive:
  - the byte-identity rerun (`:221-232`) compares raw bytes of all three files;
  - the additive manifest diff (`:234-241`);
  - the guard, graph-sha, `--trained`, version, held-out-seed mismatch, and missing-seed-0 paths;
  - the histogram sum;
  - both branches of the round trip.

  Gaps:
  - (a) The "additive manifest" test compares *parsed* objects on a 3-key JS-written fixture. It never checks that the bytes of other fields are preserved, and never runs against a copy of the real Python-written manifest. The round-trip guard covers most of this.
  - (b) No test checks that a failed manifest round trip leaves the artifact untouched. That test would have caught N1.
  - (c) The guard test (`:243-247`) passes the *real* `DEFAULT_OUT`/`DEFAULT_REPORT_MD`/`DEFAULT_MANIFEST`. If the guard ever regresses, running the tests overwrites the committed `public/data` files.
  - (d) Nothing tests the seed-0 `rewiredArms` mismatch branch, the graphId/seed-shape mismatch in `runShardedEvaluation`, or `readRewireIndex` with a missing `gzipBytes` or a negative seed.
- **I8 `--trained`: Fixed correctly.** `null-report.ts:614-619`:
  - A missing file, including the default `training/runs/null/trained.json` (absent in this checkout), is a silent no-op.
  - An existing file throws with a WP3 message.

  Two nits:
  - A typo'd `--trained` path is silently ignored, which is harmless because there is no data.
  - When the default path triggers the throw, the error still says "Remove --trained", even though the operator never passed the flag.
- **I9 seed-range wording: Fixed correctly.**
  - `rewiredSeedRangeText` (`null-report.ts:435-440`) derives the range from the data.
  - The regenerated doc correctly reads `0..499`.
  - The trailing blank line is gone: `trimEnd()` at `:545`, and the file ends with a single `\n` (checked with `od`).
  - Nit: a contiguous set that does not start at 0 (for example `100..599`) is labelled "not necessarily contiguous", which is vaguer than it needs to be.

## Correctness Sentinel I1–I6 (cross-check)

- **CS-I1 signal exit, CS-I2 fail-fast:** Fixed. These are the same as my I2 and I3 above. The completeness check they suggested is present.
- **CS-I3 histogram counts: Fixed correctly.**
  - `buildHistogram` now takes a separate `range` (`null-stats.ts:135-159`).
  - `null-report.ts:386-387` counts `nullValues` only.
  - In the data, the edges span -1.86 (disconnected) to 6.2027 (max of N). Bins 0 and 6 are now 0, and the sum is 500.
  - The regression test is at `null-report.test.ts:164`.
- **CS-I4 shards/timing provenance:** Fixed (same as my I5). Timing is rendered in the Parameters table. Label nit: "Per-episode time 32.1 ms" is *wall-clock time divided by episodes across 18 shards*, which is throughput. The actual per-episode compute is about 18× that (~577 ms). The plan asked for the measured per-episode time, so the published label understates it. Relabel it as "Wall time per episode (amortized over 18 shards)" or publish both figures.
- **CS-I5 duplicate seeds:** Fixed (same as my I4). The parent uses `if (!results.has(...))` (`null-evaluate.ts:358`) and silently keeps the first result rather than throwing. That is unreachable now, but it is weaker than the suggested defense-in-depth `throw`.
- **CS-I6 report input checks: Fixed correctly.**
  - `raw.version` is checked (`:320`).
  - `heldOutSeeds` is checked for disconnected and for every rewired graph (`:338-348`).
  - `manifest.binarySha256` is checked (`:589`).
  - The seed-0 `rewiredArms` check is best-effort (`:596-604`).

## Newly introduced issues

**N1 (Important): a failed manifest round-trip check still leaves a partial publish.** `runNullReport` writes the artifact (`null-report.ts:638`) *before* `updateManifestWithRewiringNull` performs its round-trip validation (`:287-294`). If that validation fails, the new `rewiring-null-v1.json` is already on disk, and the manifest still carries the *old* sha. That is exactly the inconsistent state the I6 reordering was meant to prevent. The probe reproduced it with a manifest containing `1.0`: `ROUNDTRIP ERR … | artifact written anyway: true | md written: false`.

Fix: split the manifest step into a pure `buildUpdatedManifestText(manifestPath, entry)`, which does the round-trip check and returns the text. Call it next to `renderReportMarkdown` at `:635`, before any write, then write all three files. Add a test asserting that `out` does not exist after a failed round trip.

**N2 (Important): `null-evaluate --out` without a `.json` suffix overwrites the results with the sidecar.** `runMetaPathFor = outPath.replace(/\.json$/, '.run.json')` (`null-evaluate.ts:496`) returns the *same path* when `--out` does not end in lowercase `.json`, for example `--out runs/authored`, `authored.JSON`, or `authored.out`. The probe confirmed that `p.replace(...) === p` is `true`. Line `:523` then overwrites the multi-hour `authored.json` with the 80-byte `{shards, elapsedMs, perEpisodeMs}`. `null-report.ts:166` has the same derivation. There it would parse `authored.json` as the sidecar, and because `shards` is not a number, it throws a confusing error.

Fix: use `` `${out.replace(/\.json$/i, '')}.run.json` ``, or assert `runMetaOut !== args.out`, and share one helper between the two files.

**N3 (Minor): a worker that exits with code 0 mid-task is not fail-fast and loses the cause.** `cleanExit` treats `code === 0` as clean even while `inFlight` is set (`null-evaluate.ts:371`). The other shards keep draining the queue. The run fails only at the completeness check, with `no result for 1 task(s): exit0` and no mention of the worker exit. The probe took about 1.5 s against about 0.1 s for the abort path. Fix: `const cleanExit = !inFlight && (code === 0 || (code === null && aborted))`, or at least treat `inFlight && !aborted` as an error.

**N4 (Minor):**
- The stale `NullEvaluationRaw` doc comment (`null-evaluate.ts:424-434`, see I5).
- The per-episode timing label (see CS-I4).
- `resolveRunMeta` does not type-check the sidecar's `elapsedMs`/`perEpisodeMs`. A string value would crash `toFixed` at render time. That happens before any write, so it fails closed.

**Round-trip check (item 9), assessed on its own terms:** it is meaningful. It catches:
- float reformatting (`1.0`→`1`);
- non-ASCII (`ensure_ascii`);
- integers above 2^53 (precision loss changes the digits);
- key-order drift;
- CRLF line endings;
- a missing trailing newline.

It cannot produce a false negative that corrupts *other* fields. If the round trip is byte-identical, adding one sorted key with ASCII string values changes only those lines. The only residual risk is ordering (N1), not the check itself.

## Test / build output (actually run)

`export PATH=~/.nvm/versions/node/v22.22.3/bin:$PATH`, then `npm run test:unit` and then `npm run build`, as separate commands:

```
 Test Files  48 passed (48)
      Tests  484 passed | 1 skipped (485)
   Duration  4.88s
```
(The jsdom "HTMLCanvasElement getContext not implemented" warnings are pre-existing noise.)

```
dist/assets/ArenaScene-B8sM_vvc.js  580.22 kB │ gzip: 145.20 kB
(!) Some chunks are larger than 500 kB after minification.   [pre-existing warning]
✓ built in 315ms
```

Not run by me: `npm run check` (lint/type-check) and the Python tests. The commit message claims both pass. I did not re-verify them.
