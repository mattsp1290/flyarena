## Action Items

### Critical
- (none)

### Important
- [ ] [scripts/null/null-worker.ts:107-120 + scripts/null/null-report.ts:387] Reject non-finite `movementScore`/`foodPickups`/`hazardContacts`. `JSON.stringify` turns NaN/Infinity into `null`, and `conditionStats` then counts it as `0`, which silently corrupts means, CIs, and the null rank. Throw in the worker with the graphId and seed. Also add an `assertFiniteScores` guard in `buildArtifact`.
- [ ] [scripts/null/null-evaluate.ts:296-304] In the `child.on('exit')` handler, stop treating `code === null` (killed by a signal, e.g. OOM `SIGKILL`) as success. Track `inFlight` and a `killedByParent` flag, and reject with `worker exited (code, signal) while running <graphId>` when a task was lost or the parent did not ask for the kill.
- [ ] [scripts/null/null-evaluate.ts:286-315,407-426] Make failure handling explicit. Either fail fast (on the first error, set `nextTaskIndex = tasks.length` and kill all children) or persist partial results to `${out}.partial.json` before throwing. In either case, wrap `Promise.all` in try/finally so every forked child is killed, and `process.exit(1)` never orphans workers mid-task.
- [ ] [scripts/null/null-evaluate.ts:74-98] In `readRewireIndex`, reject duplicate `seed` values, non-integer or negative seeds, null entries, and a missing/non-number `gzipBytes`. Duplicates currently produce duplicate `rewired` entries that are double-counted in the null set.
- [ ] [scripts/null/null-report.ts:46-48,64-72,258 + scripts/null/null-evaluate.ts:336-358] The published `shards` field comes from `null-report --shards`, which defaults to 18 regardless of the actual run. Remove it from the artifact, read it from a non-compared sidecar written by `null-evaluate`, or make the flag required. Update both doc comments to match.
- [ ] [scripts/null/null-report.ts:37-41,386-402] Add a guard against overwriting the shipped `public/data/rewiring-null-v1.json`, the manifest, and `docs/rewiring-null-report.md` from a fixture or partial `authored.json`, as `scripts/training/evaluate.ts:268-320` does (for example, require an explicit `--publish` or a real-run shape). Render the markdown before any write, so a render failure cannot leave the manifest pointing at a new sha.
- [ ] [tests/unit/ (new null-report.test.ts)] Add tests for `null-report.ts`: rerun byte-identity of the artifact, markdown, and manifest (plan requirement) in a tmpdir with a copied manifest; a manifest diff that touches only `rewiringNull`; the missing-seed-0 and missing-biological errors; and `parseNullReportArgs` unknown-flag rejection.
- [ ] [scripts/null/null-report.ts:38,57,76,91-93,120] Remove the unused `--trained` flag and `DEFAULT_TRAINED`, or make it throw "not implemented". Today it is parsed and then silently ignored.
- [ ] [scripts/null/null-report.ts:321-323,333] Derive the "rewiring seeds" range in the markdown from `artifact.rewired` seeds, not `0..length-1`, and flag non-contiguous sets. The current text is false for any batch that isn't exactly 0..n-1.

### Suggestions
- [ ] [scripts/null/null-evaluate.ts:265-307] Add a per-task watchdog timeout (reset in `assignNext`, cleared on exit) so a hung worker cannot stall `Promise.all` forever.
- [ ] [scripts/null/null-evaluate.ts:287-289] Assert that each result message's `graphId` equals the in-flight task's id and that `results.length === heldOutSeeds.length`.
- [ ] [scripts/null/null-report.ts:237-246,387] Before `pairedStats`, assert `raw.biological.heldOutSeeds` equals `seed0.heldOutSeeds`. Validate `raw.version === 1` instead of relying on the bare `as NullEvaluationRaw` cast.
- [ ] [scripts/null/null-report.ts:238-240] Make the missing-seed-0 error actionable: name the file, the seed range found, and the fix (rerun with a seed range that includes 0).
- [ ] [scripts/null/null-evaluate.ts:177-184,411] Resolve `--graph`/`--rewired-index`/`--graphs-dir` against cwd at parse time, as `--out` already is. Reject or document `--graph` passed without `--biological`.
- [ ] [scripts/null/null-evaluate.ts:267] Filter `--inspect*` out of the `execArgv` passed to `fork`, so debugging the parent doesn't crash 18 children with EADDRINUSE.
- [ ] [scripts/null/null-report.ts:378-379,286-296] Remove the extra trailing blank line in the generated markdown (it currently ends with `\n\n`). Use `toPrecision`/exponent formatting for histogram edges when the span is tiny, so degenerate histograms don't render identical ranges.
- [ ] [scripts/null/null-report.ts:177-203] Document that `updateManifestWithRewiringNull`'s byte-compat with Python `json.dumps(sort_keys=True, indent=2)` holds only while the manifest has no floats or non-ASCII characters. Alternatively, switch to a text-level insert of the `rewiringNull` block.
- [ ] [tests/unit/null-evaluate.test.ts:30-72,158] Add negative arg-parse cases (unknown flag, value-less flag, `--shards 0`), direct `readRewireIndex`/`buildTasks` ordering tests, and an explicit vitest timeout on the spawn-based determinism `it`.
- [ ] [tests/unit/null-stats.test.ts:18-35] Pin `graphStats` std (for example `std([1..5]) === sqrt(2)`) and check that the CI contains the mean, so the delegation to `conditionStats` is covered for correctness and not only determinism.
- [ ] [scripts/null/null-worker.ts:58, null-evaluate.ts:41, null-report.ts:49] Align the three `sha256Hex` copies on `evaluate.ts`'s `(data: Uint8Array | string)` signature. Per-file duplication is an accepted repo convention, so this is optional.
- [ ] [scripts/null/null-stats.ts:116-150] Soften the `Histogram.edges` doc claim of exact half-open bins, or note that floating-point rounding at interior edges may assign an on-edge value to the lower bin.
