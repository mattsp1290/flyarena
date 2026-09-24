# Fix Verification: f482a64 + 57b8a33 (Correctness Sentinel follow-up)

VERDICT: REQUEST_CHANGES

This is a narrow request. All six of my original Important items are fixed correctly, and I checked each one against the current source and by running it. The fix commit adds one new Important bug: a one-line sidecar-path collision (N1). It also leaves one stale doc comment (N2). Everything else below is optional.

## Original Correctness Sentinel items

- **I1 (signal exit = success): Fixed correctly.** `scripts/null/null-evaluate.ts:370-380`. An exit is clean only when `code === 0`, or when `code === null && aborted`. Any other exit pushes an error that names the in-flight task. A completeness check was also added at `:399-402`. I reproduced the original case (1 shard, `[a, kill, c]`). It now rejects in 223 ms with `worker exited unexpectedly (code null, signal SIGKILL) while running "kill"`. Before the fix, it resolved with `keys=['a']`.
- **I2 (fail fast + child cleanup): Fixed correctly.** `null-evaluate.ts:299-306, 322-326, 386-394`.
  - The `children` Set, `abortAll()`, the `aborted` gate in `assignNext`, and the `finally` sweep are all present.
  - Every failure branch (error message, wrong graphId, bad shape, `'error'` event, abnormal exit) calls `abortAll()`.
  - `Promise.all` settles only after every child's `exit` (or `'error'`) event. So when the function resolves, every child has already exited.
  - Probe results with 2 shards and 300 ms sibling tasks: non-zero exit rejected in 219 ms, error-then-`exit(1)` in 218 ms, wrong-graphId in 190 ms, bad-shape in 214 ms. After every case, `pgrep -P` showed no worker children left.
  - Error-then-`exit(1)` produced exactly one error line, not two. Because `aborted` is set before the exit arrives, the exit counts as clean.
- **I3 (histogram counts): Fixed correctly.**
  - `scripts/null/null-stats.ts:135-160`: a `range` parameter was added, and a descending range throws.
  - `scripts/null/null-report.ts:386-387`: the call site takes the edges from `N ∪ {bio, disc}` and the counts from `nullValues` only.
  - Published artifact: `sum(bins.counts) = 500`, `edges[0] = -1.86` (disconnected, the union min), `edges[30] = 6.2027` (the rewired max). The counts are 0 in bins 0 and 6, where disc and bio used to be counted.
- **I4 (shards/timing provenance): Fixed correctly, with a caveat.**
  - `null-evaluate.ts:519-523` writes the `<out>.run.json` sidecar.
  - `null-report.ts:177-193` reads it, and there is no silent default of 18 any more.
  - `null-report.ts:415-417` and `:457-459` publish the timing and render it.
  - Caveats: see N1 (sidecar path collision), N2 (stale doc), and N6 (the published timing was reconstructed).
- **I5 (duplicate/malformed seeds): Fixed correctly.** `null-evaluate.ts:88-113` rejects non-integer, negative, and duplicate seeds and a missing `gzipBytes`. As extra protection, `results.set` at `:358` is guarded by `!results.has`, and the per-task graphId/shape check at `:341-357` catches it too.
- **I6 (report input checks): Fixed correctly.**
  - `null-report.ts:320-322` checks the version.
  - `:338-348` checks that held-out seeds match across disconnected and every rewired graph.
  - `:589-595` checks `binarySha256` against the manifest.
  - `:596-604` cross-checks seed 0 against the manifest (best-effort).

## Maintainability Scout items (at a glance)

- **I1 (NaN/Infinity):** mostly fixed. The worker throws at `scripts/null/null-worker.ts:122-127`. The suggested report-side `assertFiniteScores` guard was not added, so a hand-edited `authored.json` with `null` scores is still summed as 0 (N3).
- **I2, I3, I4:** fixed (they are the same items as my I1, I2, and I5).
- **I5:** fixed via the sidecar. The doc comment the item asked to update is still stale (N2).
- **I6:** fixed. `guardShippedDefault` is at `null-report.ts:559-568`, and the markdown is rendered before any write (`:633-635`). The manifest round-trip check still runs after the artifact write (N4).
- **I7:** fixed (the new `tests/unit/null-report.test.ts`).
- **I8:** fixed. `--trained` fails loudly if the file exists (`null-report.ts:614-619`).
- **I9:** fixed (`rewiredSeedRangeText`, `null-report.ts:435-440`).

## Newly introduced issues

- **N1 (Important, one-line fix): the sidecar path can collide with `--out` and overwrite the raw results.** At `null-evaluate.ts:496`, `runMetaPathFor = (p) => p.replace(/\.json$/, '.run.json')` changes nothing when `--out` does not end in lowercase `.json`. I confirmed `foo/authored → foo/authored` and `foo/authored.JSON → foo/authored.JSON`.
  - What happens: `runNullEvaluate` writes `authored` at `:517`, then overwrites the same file with `{shards, elapsedMs, perEpisodeMs}` at `:523`. The CLI still prints "wrote X and X". A multi-hour run's raw per-seed data is lost without any error.
  - The same regex in `null-report.ts:166` would read `authored` itself as the sidecar. It fails safely there (no numeric `shards`), but only after the data is already gone.
  - Fix: build the path as `${out.replace(/\.json$/i, '')}.run.json`, or throw if `runMetaOut === args.out`.
- **N2 (Minor): stale doc comment.** `null-evaluate.ts:431-433` still says "`null-report.ts` takes its own `--shards` flag ... rather than reading it from here". It now reads the sidecar by default, and `--shards` is only an override. Maintainability Scout's I5 explicitly asked for this comment to be updated.
- **N3 (Minor):** no finite-number check in `buildArtifact` (see Maintainability Scout I1 above).
- **N4 (Minor): the manifest check can still leave a partial publish.** `null-report.ts:638-640` runs the manifest round-trip check inside `updateManifestWithRewiringNull`, after `atomicWriteFileSync(args.out, ...)`. If the check fails, the new artifact is on disk but the manifest sha is the old one. That is the partial publish the render-first comment at `:630-632` says it prevents. Fix: run the round-trip check before the first write.
- **N5 (Minor): out-of-range values are clamped silently.** `null-stats.ts:155-156` puts values outside an explicit `range` into the first or last bin instead of throwing. The only caller passes a superset range, so this is not reachable today.
- **N6 (Minor, provenance): the published timing is reconstructed, not measured.** The commit message says `authored.run.json` was rebuilt from log output (`elapsedMs: 1610300` exactly, rounded to 0.1 s). Neither the artifact nor the report says so.
- **N7 (Minor): no sidecar validation.** `resolveRunMeta` does not check that the sidecar's `shards` is a positive integer. `--shards` overrides the shard count but keeps the sidecar's timing, which may belong to a different run.

**No race, double-count, or false-positive error found in the `expectedTask`/`inFlight` logic.**
- The parent sends exactly one task at a time per child, and `null-worker.ts:132-145` sends exactly one message per task. So a valid reply always matches `inFlight`.
- A late result from a child that has already been aborted and killed is still stored, and at most once because of `!results.has`. `assignNext` then stops dispatching because `aborted` is set.
- Two edge cases resolve correctly: more shards than tasks (8 shards, 2 tasks) and zero tasks.

## Tests: do they exercise the fixed behavior?

- **Stub-worker SIGKILL test:** meaningful. The pre-fix code resolves on this input, and the test asserts the `/exited unexpectedly/` message.
- **Fail-fast test:** meaningful. Pre-fix it takes about 2400 ms; the assertion is `< 900` ms. Because `Promise.all` waits for every child's exit, passing also shows the sibling was killed. It is a timing threshold, so it could be flaky on a heavily loaded CI host.
- **Reverse-order test:** weak. Results are collected into a Map, so this test passed before the fix as well.
- **Histogram regression (`null-report.test.ts:164-172`):** meaningful. The pre-fix code gives sum 7, not 5.
- **Other `null-report.test.ts` tests:** the byte-identity, manifest-additive-only, shipped-path guard, source-sha mismatch, and round-trip refusal tests are all meaningful.
- **Paths with no test (my probe covered them manually):**
  - non-zero exit code;
  - wrong graphId;
  - bad result shape (the stub always sends `heldOutSeeds: []`, so the shape check passes trivially);
  - the rewired branch of the held-out-seed check;
  - the seed-0 `rewiredArms` cross-check.

## Commands run (from the worktree, node v22.22.3)

```
$ npm run check
> svelte-check --tsconfig ./tsconfig.json
COMPLETED 844 FILES 0 ERRORS 0 WARNINGS 0 FILES_WITH_PROBLEMS

$ npx vitest run tests/unit/null-evaluate.test.ts tests/unit/null-stats.test.ts tests/unit/null-report.test.ts
 Test Files  3 passed (3)
      Tests  53 passed (53)
   Duration  2.30s
```

Ad-hoc probe of `runShardedEvaluation` with its own stub worker (scratchpad, not committed):

```
ok 10 tasks/3 shards: RESOLVED (all 10 keys) 342ms
sigkill mid-queue, 1 shard: REJECTED 223ms :: worker exited unexpectedly (code null, signal SIGKILL) while running "kill"
exit code 3, 2 shards: REJECTED 219ms :: worker exited unexpectedly (code 3, signal null) while running "exit3"
error then exit(1): REJECTED 218ms :: errexit: boom   (single error line)
wrong graphId: REJECTED 190ms :: received a message for "other" but no matching task was in flight
bad result shape: REJECTED 214ms :: result shape does not match the task's held-out seeds
more shards than tasks: RESOLVED keys=b,a
zero tasks: RESOLVED keys=(none)
(no worker children alive after any case)
```

## Published data spot-check (public/data/rewiring-null-v1.json @ 57b8a33)

- `sum(bins.counts) = 500`, with 30 bins. `shards = 18`, and there are 500 rewired entries.
- `timing = {elapsedMs: 1610300, perEpisodeMs: 32.0777}`.
- The manifest `rewiringNull.sha256` is `592446bb...083d1`. That equals the actual sha256 of the file.
- `training/runs/null/authored.run.json` is gitignored (`.gitignore:38`). This is expected, but it means the sidecar that the published provenance depends on is not versioned.
