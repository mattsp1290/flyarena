import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { readNpyFloat32Array } from '../training/npy';
import { readRunDir, type LoadedRun } from '../training/run-dir';
import type { ArmName } from '../training/arms';
import type { GraphListIndex } from '../null/graph-list-index';
import { resolveBigqGraphIdentity, resolveInterventionGraphIdentity, type GraphIdentity } from './graph-identity';
import {
  assertArmBundlesMatchRawScores,
  copyRawInterventionScores,
  readRawInterventionScores
} from './raw-intervention-scores';
import { copyInterventionIndex, readAndValidateInterventionIndex } from './intervention-index';
import { buildInterventionSwapsArchive, extractInterventionSwaps } from './intervention-swaps';
import { copyVerbatim } from './copy-verbatim';

/**
 * `.agents/plans/readout-attribution/01-archive-and-types.md`'s WP1: archive
 * every default-task trained readout this repository's analyses depend on
 * (the `flyarena-bigq` production readouts and the `pathway-interventions`
 * trained readouts) into one committed, sha-verified JSON, so the analysis
 * (WP2/WP3) is reproducible from the repository alone -- the source run
 * directories are gitignored (`training/runs/`) and live only in agent
 * worktrees, which may be pruned.
 *
 * A run directory's own `config.json` (`run-dir.ts`'s `RunConfig`) cannot by
 * itself identify which graph a readout was trained against: every
 * intervention run's `arm` is `"rewired"` (never `"P"`/`"C000"`/etc -- WP1's
 * intervention graphs are exported through `export-arms.ts --rewired`, which
 * only ever knows the three-arm vocabulary), and `graphArtifactSha256` is the
 * *base* biological-dataset hash, shared by every run this study scores
 * (bigq and every intervention/control graph alike -- they are all rewirings
 * of the same underlying MaleCNS graph). Also note this archive's own
 * `graphId` field is unrelated to `config.json`'s own (differently-scoped)
 * `graphId` field -- see `ArchivedReadout.graphId`'s doc comment. So
 * `graphId` (identifying which topology -- `"biological"`, `"rewired-seed0"`,
 * `"disconnected"`, `"P"`, `"C000"`, `"M1000"`, ...) is supplied by this
 * script's caller (the `--source` label), not inferred from the run
 * directory. `armBundleSha256` (the run's own `config.json` field, already
 * checked against the arm bundle's self-certifying hash at training time --
 * `null-trained-worker.ts`'s `runTask`) proves the run trained against SOME
 * bundle matching that hash, but says nothing on its own about which
 * topology label that bundle carries -- `assertArmMatchesGraphId` below is
 * this script's own cheap, local check that a `--source` label is at least
 * arm-consistent (catches the "swapped two --source paths" operator-error
 * class; it cannot catch e.g. `--source C000=<C001's dir>`, since both are
 * `arm: "rewired"`). WP2's analyses resolve a readout's graph through
 * `graphId` + `armBundleSha256` together, and this archive refuses to record
 * an entry whose `armBundleSha256` is missing.
 *
 * Caveat for WP2 (a dual-review finding, verified against
 * `export-arms.ts`'s `computeArmBundleSha256`, and confirmed empirically by
 * a later thermo-provenance review): `armBundleSha256` hashes the *entire*
 * serialized arm bundle, including `provenance.artifactPath` -- the literal
 * CLI argument string `export-arms.ts --rewired`/`--graph` was invoked with,
 * not a canonicalized path. Whether it reproduces from a fresh checkout
 * therefore depends entirely on what string was passed at export time, NOT
 * on anything this script controls:
 *
 * - **bigq** (`biological`/`rewired-seed0`/`disconnected`): reproducible.
 *   The original export used the repo-relative
 *   `public/data/malecns-arena-v1*.bin.gz` paths, confirmed by re-running
 *   `export-arms-cli.ts` against those same relative paths from this
 *   worktree and getting byte-identical `armBundleSha256` values.
 * - **interventions** (`P`/`C000`-`C004`/`M1000`-`M1004`): NOT reproducible.
 *   `train-sample.sh`/`lookup-intervention-graph.ts`'s
 *   `resolveVerifiedInterventionGraphPath` always resolves an ABSOLUTE,
 *   checkout-specific path before calling `export-arms.ts --rewired` with
 *   it (`node:path`'s `resolve` never returns a relative path) -- so a
 *   graph regenerated from ANY checkout other than the exact one that
 *   trained the original run will have byte-identical CSR arrays but a
 *   DIFFERENT `armBundleSha256`.
 *
 * So `armBundleSha256` is kept as this specific archived run's own
 * training-time provenance (and the field WP2 must still record if it ever
 * needs to reason about which exact bundle a run trained against), but it is
 * NOT usable as WP2's graph-REGENERATION check. `graphGzipSha256`/
 * `graphBinarySha256` (below) are the path-independent identity WP2 should
 * check a regenerated graph against instead: the graph artifact's own
 * content hashes, verified here against the actual artifact bytes (not
 * merely copied from `intervention-index-v1.json`/the manifest) --
 * `graph-identity.ts`'s `resolveBigqGraphIdentity`/`resolveInterventionGraphIdentity`.
 */

const here = dirname(fileURLToPath(import.meta.url));
export const repoRoot = resolve(here, '../..');

export const DEFAULT_OUT = resolve(repoRoot, 'training/archive/trained-readouts-v1.json');
export const DEFAULT_RAW_INTERVENTION_SCORES_OUT = resolve(repoRoot, 'training/archive/intervention-trained-raw-v1.json');
export const DEFAULT_INTERVENTION_INDEX_OUT = resolve(repoRoot, 'training/archive/intervention-index-v1.json');
export const DEFAULT_INTERVENTION_SWAPS_OUT = resolve(repoRoot, 'training/archive/intervention-swaps-v1.json');

/**
 * WP1b: `<dir>/task-intervention-trained-raw-<arenaTask>-v1.json` -- one file
 * per arena task (never merged into one file -- see
 * `raw-intervention-scores.ts`'s doc comment), so each stays a
 * byte-for-byte copy of that task's own `flyarena-s8z8` `trained.json`, each
 * independently checkable against its own recorded sha256. `resolveArenaTask`
 * validates `arenaTask` (throws on an unrecognized id) before it is
 * interpolated into a filesystem path. `dir` is a parameter (like every
 * other output's `*Out`/`*OutDir` field on `ArchiveReadoutsArgs`, overridable
 * via `--task-intervention-raw-scores-out-dir`), never a `repoRoot`-rooted
 * constant baked directly into this function: `runArchiveReadouts` is
 * exercised directly (bypassing CLI parsing entirely) by its own unit tests,
 * and a hard-coded `repoRoot` path here would make those tests write into
 * this actual checkout's committed archive instead of an isolated tmpdir (a
 * real bug caught while adding this module's own test coverage).
 */
export const DEFAULT_TASK_INTERVENTION_RAW_SCORES_OUT_DIR = resolve(repoRoot, 'training/archive');

export const taskInterventionRawScoresOutPath = (dir: string, arenaTask: string): string =>
  resolve(dir, `task-intervention-trained-raw-${resolveArenaTask(arenaTask).id}-v1.json`);

// ---------------------------------------------------------------------------
// Archive schema
// ---------------------------------------------------------------------------

export type ArchivedReadoutKind = 'bigq' | 'intervention' | 'task-intervention';

export interface ArchivedReadout {
  /** `"<graphId>-seed<trainerSeed>[-<idSuffix>][-<arenaTask>]"` (the `arenaTask` suffix is automatic -- WP1b -- and omitted for `arenaTask: "default"`) -- unique within the archive (see `mergeReadouts`). Not guaranteed injective against an adversarially-chosen `graphId`/`idSuffix`/`arenaTask` containing `"-seed<digits>"` itself (string concatenation, not a structured key) -- not a live risk for this archive's own fixed `graphId`/`arenaTask` vocabularies (which do include the literal `"rewired-seed0"`, so a naive "reject `-seed\d` in labels" guard would incorrectly reject real data), but `mergeReadouts`' identity check is the actual safety net against a collision, not this format. */
  readonly id: string;
  readonly arm: ArmName;
  readonly kind: ArchivedReadoutKind;
  /**
   * The topology label this script's `--source` caller supplied -- e.g.
   * `"biological"`, `"rewired-seed0"`, `"P"`, `"C000"`. NOT the same field as
   * the run's own `config.json` `graphId` (always `"malecns-arena-v1"`, the
   * *base* dataset id `flyarena-train` recorded regardless of which arm/
   * topology it trained -- see `run-dir.ts`'s `RunConfig`). A reader joining
   * this archive against a run directory's raw `config.json` by `graphId`
   * will get silent non-matches if it doesn't know the two fields mean
   * different things.
   */
  readonly graphId: string;
  readonly trainerSeed: number;
  /** `run.config.arenaTask`, or `'default'` when the run directory predates arena tasks. */
  readonly arenaTask: string;
  /**
   * WP1b: `resolveArenaTask(arenaTask).fingerprint` (`src/lib/arena/tasks.ts`)
   * -- `undefined` for `arenaTask: "default"` (every WP1 bigq/intervention
   * entry; the default task's fingerprint is a fixed, always-reproducible
   * constant, and leaving it `undefined` here keeps those 23 already-committed
   * entries' bytes reproducible bit-for-bit on a hypothetical future re-run
   * of their own `--source` args -- see `IDENTITY_FIELDS`), always present
   * for a `kind: "task-intervention"` entry. Recorded directly on the entry
   * (not merely re-derivable by re-resolving `arenaTask`) so a reader/test
   * can check it against `tests/fixtures/golden/tasks.json` without importing
   * `resolveArenaTask` itself.
   *
   * DO NOT read this field directly to answer "what is this entry's arena-
   * task fingerprint" -- `undefined` here does NOT mean "unknown" or "any
   * task" (unlike most other optional fields in this codebase, e.g. a
   * legacy run's absent `arenaTaskFingerprint` in `null-trained-worker.ts`'s
   * `assertRunMatchesExpectedIdentity`); it means "the default task's,
   * omitted only to keep these bytes stable." Use `arenaTaskFingerprintOf`
   * below, the sanctioned read path, instead (a thermo-maintainability
   * review finding).
   */
  readonly arenaTaskFingerprint?: string;
  readonly graphArtifactSha256: string;
  /**
   * This run's arm bundle's own self-certifying hash. Training-time
   * provenance only -- NOT a durable, path-independent graph identity: it
   * hashes `provenance.artifactPath` (a literal export-time CLI argument
   * string) along with the graph bytes, so it reproduces from a fresh
   * checkout only when that original export used a repo-relative path
   * (true for bigq, false for every intervention entry -- see the module
   * doc comment's "Caveat for WP2" section for the empirical check). Use
   * `graphGzipSha256`/`graphBinarySha256` below for graph-regeneration
   * verification instead (a thermo-provenance review finding).
   */
  readonly armBundleSha256: string;
  /**
   * Path-independent graph identity, thermo-provenance review: the graph
   * artifact's own content hashes (gzip bytes / decompressed CSR binary
   * bytes), recomputed and verified against the actual artifact -- for bigq
   * entries against `public/data/malecns-arena-v1.manifest.json` (and the
   * committed `public/data/malecns-arena-v1*.bin.gz` files themselves); for
   * intervention entries against `training/archive/intervention-index-v1.json`'s
   * entry for this `graphId` (and the graph binary that index entry's `path`
   * names, at archive-build time). `null` only for `graphId: "disconnected"`,
   * which has no separate artifact at all (derived at runtime from
   * `biological` with `edgeCount: 0` -- see `export-arms.ts`'s
   * `toDisconnectedGraph`/`disconnected-runtime-zero-edge` provenance kind).
   * Unlike `armBundleSha256`, these ARE expected to reproduce from a graph
   * regenerated in any checkout -- this is what WP2's `resolve-graph.ts`
   * should check a regenerated graph against.
   */
  readonly graphGzipSha256: string | null;
  /** The decompressed CSR binary's sha256 -- see `graphGzipSha256`'s doc comment. */
  readonly graphBinarySha256: string | null;
  readonly D: number;
  readonly H: number;
  readonly parameterCount: number;
  /** sha256 of `theta_final.npy`'s raw file bytes (header + data) -- distinct from `weightsSha256` below. NOT recomputable from this archive's own `theta` field (the decoded payload has no `.npy` header), so it is provenance of the shipped file only, not a self-check a downstream consumer can repeat. */
  readonly thetaSha256: string;
  /**
   * sha256 of the DECODED flat float32 buffer -- equals `run-dir.ts`'s
   * `weightsSha256`, and, for every bigq entry, the published
   * `trained-readout-v1.report.json`'s per-replica `weightsSha256` (verified
   * for all 9 non-rerun bigq entries when this archive was built). Unlike
   * `thetaSha256`, this IS recomputable from the archive alone:
   * `sha256(base64decode(theta)) === weightsSha256`. Not part of
   * `01-archive-and-types.md`'s literal field list -- added because both of
   * this diff's independent reviewers flagged `thetaSha256` as the archive's
   * only integrity field despite being unverifiable from the archive's own
   * contents.
   */
  readonly weightsSha256: string;
  /** The flat `theta_final.npy` payload, base64-encoded little-endian float32 (`[w1, b1, w2, b2]`, `run-dir.ts`'s documented layout). */
  readonly theta: string;
  /**
   * Absolute run-directory path at archive time -- historical provenance
   * only, never re-resolved from the archive, and NOT expected to still
   * exist (every real WP1 source is inside another agent worktree, which may
   * since have been pruned, and the path is meaningful only relative to
   * whichever checkout originally ran this CLI). `mergeReadouts` keeps the
   * PRIOR entry's `sourcePath` on an identity-matching re-run, so this value
   * does not change merely because a later invocation ran from a different
   * checkout.
   */
  readonly sourcePath: string;
}

/**
 * The sanctioned way to read an `ArchivedReadout`'s arena-task fingerprint
 * (a thermo-maintainability review finding on `arenaTaskFingerprint`'s own
 * doc comment above): always resolvable from `arenaTask` -- which is never
 * optional -- even when the stored field itself is omitted for byte-identity
 * reasons on a default-task entry. Never returns "any"/"unknown": a default
 * entry resolves to the DEFAULT task's own fingerprint, exactly what
 * `resolveArenaTask('default').fingerprint` (equivalently
 * `resolveArenaTask(undefined).fingerprint`) already is. WP2 code should
 * call this rather than branch on `entry.arenaTaskFingerprint === undefined`
 * itself.
 */
export const arenaTaskFingerprintOf = (entry: Readonly<ArchivedReadout>): string =>
  entry.arenaTaskFingerprint ?? resolveArenaTask(entry.arenaTask).fingerprint;

export interface TrainedReadoutArchive {
  readonly version: 1;
  /** Sorted by `id` ascending -- see `mergeReadouts`. */
  readonly readouts: readonly ArchivedReadout[];
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface SourceSpec {
  readonly graphId: string;
  readonly idSuffix: string | null;
  readonly runDir: string;
}

/** `graphId`/`idSuffix` feed directly into the archived `id` string (a plain concatenation, not a structured key) -- restricted to a safe, unambiguous charset (letters, digits, underscore, hyphen) so a stray space or `=`/`:` in a label can't produce a malformed or silently-different id. Every real WP1 label (`biological`, `rewired-seed0`, `P`, `C000`, `M1000`, `gpurerun`) already fits this. */
const SAFE_LABEL = /^[A-Za-z0-9_-]+$/;

/**
 * `--source <graphId>[:<idSuffix>]=<run-dir>`. `idSuffix` disambiguates two
 * runs that would otherwise share one `(graphId, trainerSeed)` id -- the one
 * case this arises for WP1 is the bigq biological/101 GPU rerun
 * (`training/runs/production/biological-101-gpurerun`, same `graphId`
 * `"biological"` and `trainerSeed` 101 as the original CPU run), archived
 * "labeled as such" per `01-archive-and-types.md`'s acceptance criterion via
 * `--source biological:gpurerun=...`.
 */
export const parseSourceArg = (flag: string, value: string): SourceSpec => {
  const eq = value.indexOf('=');
  if (eq <= 0) throw new Error(`${flag} must be "<graphId>[:<idSuffix>]=<run-dir>", got "${value}"`);
  const spec = value.slice(0, eq);
  const runDir = value.slice(eq + 1);
  if (!runDir) throw new Error(`${flag} must be "<graphId>[:<idSuffix>]=<run-dir>", got "${value}"`);
  const colon = spec.indexOf(':');
  const graphId = colon === -1 ? spec : spec.slice(0, colon);
  if (!graphId) throw new Error(`${flag}: missing graphId in "${value}"`);
  if (!SAFE_LABEL.test(graphId)) {
    throw new Error(`${flag}: graphId "${graphId}" must match ${SAFE_LABEL} (got "${value}")`);
  }
  const idSuffixRaw = colon === -1 ? '' : spec.slice(colon + 1);
  // A `:` with nothing after it (`--source biological:=dir`) is almost
  // certainly a typo -- the suffix exists specifically to disambiguate an
  // id that would otherwise collide (the GPU-rerun case), so silently
  // treating it as "no suffix" would defeat that disambiguation exactly
  // when the caller most clearly intended to use it.
  if (colon !== -1 && idSuffixRaw.length === 0) {
    throw new Error(`${flag}: empty idSuffix after ":" in "${value}"`);
  }
  if (idSuffixRaw.length > 0 && !SAFE_LABEL.test(idSuffixRaw)) {
    throw new Error(`${flag}: idSuffix "${idSuffixRaw}" must match ${SAFE_LABEL} (got "${value}")`);
  }
  return { graphId, idSuffix: idSuffixRaw.length > 0 ? idSuffixRaw : null, runDir: resolve(process.cwd(), runDir) };
};

export interface TaskInterventionRawScoreSpec {
  readonly arenaTask: string;
  readonly path: string;
}

/**
 * `--task-intervention-raw-scores <arenaTaskId>=<path>` (WP1b, repeatable --
 * one per task): `<path>` is a task's own `flyarena-s8z8`
 * `null-trained-evaluate.ts --graph-list` rescore output (`trained.json`),
 * copied verbatim into `taskInterventionRawScoresOutPath(outDir, arenaTaskId)`
 * (`outDir` defaults to `DEFAULT_TASK_INTERVENTION_RAW_SCORES_OUT_DIR`,
 * overridable with `--task-intervention-raw-scores-out-dir`). No `idSuffix`
 * (unlike `--source`): one raw-scores file per arena task is the whole shape
 * (`raw-intervention-scores.ts`'s doc comment).
 */
export const parseTaskInterventionRawScoresArg = (flag: string, value: string): TaskInterventionRawScoreSpec => {
  const eq = value.indexOf('=');
  if (eq <= 0) throw new Error(`${flag} must be "<arenaTaskId>=<path>", got "${value}"`);
  const arenaTask = value.slice(0, eq);
  const path = value.slice(eq + 1);
  if (!path) throw new Error(`${flag} must be "<arenaTaskId>=<path>", got "${value}"`);
  resolveArenaTask(arenaTask); // throws on an unrecognized arena-task id
  return { arenaTask, path: resolve(process.cwd(), path) };
};

/**
 * Rejects two `--task-intervention-raw-scores` specs for the SAME arena
 * task in one invocation (a dual-review finding): both would validate
 * independently and the second would silently overwrite the first's output
 * file (both resolve to the same `taskInterventionRawScoresOutPath`) --
 * unlike `mergeReadouts`, which always refuses to silently overwrite a
 * conflicting entry. Extracted into its own named function (a
 * thermo-provenance review suggestion) for stylistic consistency with this
 * file's other `parseArgs`-adjacent validators (`parseSourceArg`,
 * `parseTaskInterventionRawScoresArg`), rather than a bare block inline in
 * `parseArgs`.
 */
export const assertNoDuplicateTaskInterventionRawScores = (specs: readonly TaskInterventionRawScoreSpec[]): void => {
  const seenArenaTasks = new Set<string>();
  for (const spec of specs) {
    if (seenArenaTasks.has(spec.arenaTask)) {
      throw new Error(`archive-readouts: --task-intervention-raw-scores "${spec.arenaTask}" was given more than once`);
    }
    seenArenaTasks.add(spec.arenaTask);
  }
};

export interface ArchiveReadoutsArgs {
  readonly sources: readonly SourceSpec[];
  readonly out: string;
  readonly rawInterventionScoresPath: string | null;
  readonly rawInterventionScoresOut: string;
  readonly interventionIndexPath: string | null;
  readonly interventionIndexOut: string;
  readonly interventionAttributionPath: string | null;
  readonly interventionSwapsOut: string;
  readonly taskInterventionRawScores: readonly TaskInterventionRawScoreSpec[];
  readonly taskInterventionRawScoresOutDir: string;
}

export const parseArgs = (argv: readonly string[]): ArchiveReadoutsArgs => {
  const sources: SourceSpec[] = [];
  let out = DEFAULT_OUT;
  let rawInterventionScoresPath: string | null = null;
  let rawInterventionScoresOut = DEFAULT_RAW_INTERVENTION_SCORES_OUT;
  let interventionIndexPath: string | null = null;
  let interventionIndexOut = DEFAULT_INTERVENTION_INDEX_OUT;
  let interventionAttributionPath: string | null = null;
  let interventionSwapsOut = DEFAULT_INTERVENTION_SWAPS_OUT;
  const taskInterventionRawScores: TaskInterventionRawScoreSpec[] = [];
  let taskInterventionRawScoresOutDir = DEFAULT_TASK_INTERVENTION_RAW_SCORES_OUT_DIR;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--source') {
      sources.push(parseSourceArg(flag, requireValue(flag, argv[index + 1])));
      index += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--raw-intervention-scores') {
      rawInterventionScoresPath = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--raw-intervention-scores-out') {
      rawInterventionScoresOut = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--intervention-index') {
      interventionIndexPath = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--intervention-index-out') {
      interventionIndexOut = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--intervention-attribution') {
      interventionAttributionPath = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--intervention-swaps-out') {
      interventionSwapsOut = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--task-intervention-raw-scores') {
      taskInterventionRawScores.push(parseTaskInterventionRawScoresArg(flag, requireValue(flag, argv[index + 1])));
      index += 2;
    } else if (flag === '--task-intervention-raw-scores-out-dir') {
      taskInterventionRawScoresOutDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (
    sources.length === 0 &&
    rawInterventionScoresPath === null &&
    interventionIndexPath === null &&
    interventionAttributionPath === null &&
    taskInterventionRawScores.length === 0
  ) {
    throw new Error(
      'archive-readouts: at least one of --source/--raw-intervention-scores/--intervention-index/' +
        '--intervention-attribution/--task-intervention-raw-scores is required'
    );
  }
  // Checked here (all of `argv` parsed, one pass) rather than in
  // `runArchiveReadouts`, so a CLI-level operator mistake is caught before
  // any file is even read -- see `assertNoDuplicateTaskInterventionRawScores`'s
  // own doc comment for why this check exists at all.
  assertNoDuplicateTaskInterventionRawScores(taskInterventionRawScores);
  for (const [flagName, path] of [
    ['--out', out],
    ['--raw-intervention-scores-out', rawInterventionScoresOut],
    ['--intervention-index-out', interventionIndexOut],
    ['--intervention-swaps-out', interventionSwapsOut]
  ] as const) {
    if (!path.endsWith('.json')) throw new Error(`${flagName} must end with ".json" (got "${path}")`);
  }

  return {
    sources,
    out,
    rawInterventionScoresPath,
    rawInterventionScoresOut,
    interventionIndexPath,
    interventionIndexOut,
    interventionAttributionPath,
    interventionSwapsOut,
    taskInterventionRawScores,
    taskInterventionRawScoresOutDir
  };
};

// ---------------------------------------------------------------------------
// training/archive/trained-readouts-v1.json
// ---------------------------------------------------------------------------

/** `graphId`s that belong to the bigq production study. Everything else at the default task is an intervention/control graph (`kind: "intervention"`); a non-default-task run is `"task-intervention"` (WP1b, not built by this file). */
const BIGQ_GRAPH_IDS: ReadonlySet<string> = new Set(['biological', 'rewired-seed0', 'disconnected']);

export const kindForEntry = (graphId: string, arenaTaskId: string): ArchivedReadoutKind => {
  if (arenaTaskId !== 'default') return 'task-intervention';
  return BIGQ_GRAPH_IDS.has(graphId) ? 'bigq' : 'intervention';
};

/** The `arm` every bigq `graphId` was actually exported/trained as (`export-arms.ts`'s three-arm vocabulary) -- every non-bigq (intervention/control) graph is exported through `export-arms.ts --rewired`, so its run's `arm` is always `"rewired"`, regardless of its `graphId` label. */
const EXPECTED_ARM_FOR_BIGQ_GRAPH_ID: Readonly<Partial<Record<string, ArmName>>> = {
  biological: 'biological',
  'rewired-seed0': 'rewired',
  disconnected: 'disconnected'
};

/**
 * A cheap, local guard against the most likely `--source` operator error --
 * swapping two run-directory paths, or mistyping a `graphId` label -- for
 * the one thing this script CAN check without reading the arm bundle file
 * itself (which it is never given a path to): whether the run's own
 * `config.arm` is even consistent with the claimed `graphId`. This is
 * necessary but not sufficient (`--source C000=<C001's run dir>` passes,
 * since both are `arm: "rewired"`) -- see the module doc comment's
 * `armBundleSha256` caveat and `runArchiveReadouts`' cross-check against
 * `--raw-intervention-scores` for the stronger check available when both are
 * supplied together.
 */
export const assertArmMatchesGraphId = (graphId: string, arm: ArmName): void => {
  const expected = EXPECTED_ARM_FOR_BIGQ_GRAPH_ID[graphId] ?? 'rewired';
  if (arm !== expected) {
    throw new Error(
      `archive-readouts: --source ${graphId} points at a run whose config.arm is "${arm}", expected "${expected}" ` +
        '-- wrong run directory for this graphId, or a mistyped --source label?'
    );
  }
};

/** `sourcePath` relative to the repository root when possible (every real WP1 source is an absolute path to another worktree, well outside `repoRoot` -- `relative` still produces a valid, if `../`-heavy, relative path in that case, which is fine: it is historical provenance, never re-resolved from the archive itself -- see `ArchivedReadout.sourcePath`'s doc comment). */
const sourcePathFor = (runDir: string): string => relative(repoRoot, runDir);

export const buildArchivedReadout = (spec: Readonly<SourceSpec>): ArchivedReadout => {
  const run: LoadedRun = readRunDir(spec.runDir);
  const { config } = run;

  if (!config.armBundleSha256) {
    throw new Error(
      `archive-readouts: ${spec.runDir}/config.json has no armBundleSha256 -- refusing to archive an entry with no verified graph identity`
    );
  }

  // `graphArtifactSha256` is written by every real `flyarena-train` run
  // (`RunConfig`'s doc comment notes it predates the TS-side field only for
  // tiny/older test fixtures), but is not part of `RunConfig`'s declared
  // shape -- read directly off the parsed config the same way
  // `null-trained-evaluate.ts`'s `CONFIG_RECONCILE_FIELDS` reads `ticks`.
  const rawConfig = config as unknown as Record<string, unknown>;
  const graphArtifactSha256 = rawConfig.graphArtifactSha256;
  if (typeof graphArtifactSha256 !== 'string' || graphArtifactSha256.length === 0) {
    throw new Error(`archive-readouts: ${spec.runDir}/config.json has no graphArtifactSha256`);
  }

  // Throws on an unrecognized arena-task id (`resolveArenaTask`'s own
  // contract) -- a run directory recording a bogus `arenaTask` is not
  // silently archived under a fabricated identity.
  const resolvedArenaTask = resolveArenaTask(config.arenaTask);
  const arenaTaskId = resolvedArenaTask.id;

  assertArmMatchesGraphId(spec.graphId, config.arm);

  const thetaPath = resolve(spec.runDir, 'theta_final.npy');
  // `thetaSha256` hashes the raw .npy FILE bytes (header + data), distinct
  // from `weightsSha256` (the decoded float buffer). `readRunDir` already
  // read and decoded this same file once (into `run.weights`/
  // `run.weightsSha256`); `npy.ts`'s `readNpyFloat32Array` takes a path (a
  // shared, load-bearing module -- widening its signature to also accept a
  // Buffer is out of scope here), so this necessarily opens the file again
  // rather than truly reading it once. The `weightsSha256 !== run.weightsSha256`
  // check below is this function's actual TOCTOU guard: if theta_final.npy
  // changed between `readRunDir`'s read and this one, the decoded buffers
  // (and their shas) will disagree and this throws, rather than silently
  // archiving a `theta` payload that came from a different file than the one
  // `readRunDir` validated (a dual-review finding).
  const thetaSha256 = sha256Hex(readFileSync(thetaPath));
  const theta = readNpyFloat32Array(thetaPath);
  const thetaBase64 = Buffer.from(theta.buffer, theta.byteOffset, theta.byteLength).toString('base64');
  const weightsSha256 = sha256Hex(Buffer.from(theta.buffer, theta.byteOffset, theta.byteLength));
  if (weightsSha256 !== run.weightsSha256) {
    throw new Error(`archive-readouts: ${thetaPath} changed while archiving (weightsSha256 mismatch)`);
  }

  // WP1b (dual-review finding): a non-default `arenaTaskId` is baked into
  // the id AUTOMATICALLY -- never left to depend on the caller also
  // remembering to pass a matching `--source graphId:<task>=...` idSuffix.
  // Before this, id-uniqueness across arena tasks for the same
  // `(graphId, trainerSeed)` pair (e.g. "P" at trainer seed 101 under both
  // "hazard-heavy" and "sparse-food") relied entirely on operator
  // convention; a missed idSuffix would not have corrupted data (`mergeReadouts`'s
  // `IDENTITY_FIELDS` -- which includes `arenaTask` -- still throws on a
  // resulting id collision, since the two entries disagree on `arenaTask`),
  // but it would have failed loudly and late rather than being structurally
  // impossible. `idSuffix` remains available for the orthogonal
  // disambiguation case (default-task GPU rerun) and stacks with the
  // task suffix, but is never REQUIRED for task uniqueness.
  const id = `${spec.graphId}-seed${config.trainerSeed}${spec.idSuffix ? `-${spec.idSuffix}` : ''}${
    arenaTaskId === 'default' ? '' : `-${arenaTaskId}`
  }`;

  return {
    id,
    arm: config.arm,
    kind: kindForEntry(spec.graphId, arenaTaskId),
    graphId: spec.graphId,
    trainerSeed: config.trainerSeed,
    arenaTask: arenaTaskId,
    arenaTaskFingerprint: arenaTaskId === 'default' ? undefined : resolvedArenaTask.fingerprint,
    graphArtifactSha256,
    armBundleSha256: config.armBundleSha256,
    // Filled in by `attachGraphIdentity` in `runArchiveReadouts` (needs
    // cross-file context -- the manifest or the intervention index --
    // `buildArchivedReadout` only ever sees one run directory at a time).
    graphGzipSha256: null,
    graphBinarySha256: null,
    D: config.D,
    H: config.H,
    parameterCount: config.parameterCount,
    thetaSha256,
    weightsSha256,
    theta: thetaBase64,
    sourcePath: sourcePathFor(spec.runDir)
  };
};

export const loadExistingArchive = (path: string): TrainedReadoutArchive | null => {
  if (!existsSync(path)) return null;
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<TrainedReadoutArchive>;
  if (parsed.version !== 1) {
    throw new Error(`archive-readouts: ${path} has unsupported version ${JSON.stringify(parsed.version)} (expected 1)`);
  }
  if (!Array.isArray(parsed.readouts)) {
    throw new Error(`archive-readouts: ${path} has no "readouts" array`);
  }
  return { version: 1, readouts: parsed.readouts };
};

/** Fields that identify what a `--source` run actually IS, not merely where this invocation happened to run from (`sourcePath` is deliberately excluded -- see `mergeReadouts`). A duplicate `id` is only ever accepted when every one of these agrees with what is already archived. */
const IDENTITY_FIELDS = [
  'thetaSha256',
  'weightsSha256',
  'arm',
  'kind',
  'graphId',
  'trainerSeed',
  'arenaTask',
  'arenaTaskFingerprint',
  'armBundleSha256',
  'graphArtifactSha256',
  'graphGzipSha256',
  'graphBinarySha256',
  'D',
  'H',
  'parameterCount'
] as const satisfies readonly (keyof ArchivedReadout)[];

/**
 * Merge `additions` into `existing`, sorted by `id` ascending (a
 * deterministic archive, independent of `--source` argument order or which
 * invocation added which entry -- matches this codebase's existing
 * "canonical task order" convention, e.g. `null-evaluate.ts`). A duplicate
 * `id` is only ever accepted when every `IDENTITY_FIELDS` entry is
 * byte-identical to the entry already on file: re-running this script
 * against the same run directories (idempotent) must succeed, but archiving
 * two DIFFERENT runs under one id (an operator error, or a stale/rebuilt run
 * directory) must not silently overwrite the earlier entry.
 *
 * On an identity match, the PRIOR entry is kept as-is (not replaced by the
 * new one) -- every `IDENTITY_FIELDS` value already agrees by construction,
 * so the only field that could otherwise differ is `sourcePath`, and
 * overwriting it would make the archive's own committed bytes depend on
 * which checkout happened to re-run this CLI (undercutting the "output is
 * deterministic" requirement -- a dual-review finding).
 *
 * Separately, the SAME `weightsSha256` is never accepted under two DIFFERENT
 * ids: `assertArmMatchesGraphId`/`assertArmBundlesMatchRawScores` catch a
 * mislabeled `--source`, but neither catches the same run directory
 * archived twice under two labels that were meant to be different runs (the
 * GPU-rerun case's whole reason to exist: `--source biological:gpurerun=<dir>`
 * pointed at the SAME run dir as `--source biological=<dir>` by operator
 * error, e.g. the GPU rerun wasn't actually done yet). Unlike the
 * `IDENTITY_FIELDS` check above (which only ever compares an entry against a
 * PRIOR entry sharing its own id), this is a cross-id check -- a round-2
 * dual-review finding.
 */
export const mergeReadouts = (
  existing: readonly ArchivedReadout[],
  additions: readonly ArchivedReadout[]
): readonly ArchivedReadout[] => {
  const byId = new Map<string, ArchivedReadout>();
  for (const entry of existing) byId.set(entry.id, entry);
  const idByWeightsSha256 = new Map<string, string>();
  for (const entry of byId.values()) idByWeightsSha256.set(entry.weightsSha256, entry.id);

  for (const entry of additions) {
    const prior = byId.get(entry.id);
    if (prior) {
      const mismatched = IDENTITY_FIELDS.filter((field) => prior[field] !== entry[field]);
      if (mismatched.length > 0) {
        throw new Error(
          `archive-readouts: id "${entry.id}" already exists in the archive with different ${mismatched.join(', ')} ` +
            `-- refusing to silently overwrite it`
        );
      }
      continue; // identical in every way that matters -- keep the prior entry (and its sourcePath) unchanged
    }
    const existingIdForWeights = idByWeightsSha256.get(entry.weightsSha256);
    if (existingIdForWeights !== undefined && existingIdForWeights !== entry.id) {
      throw new Error(
        `archive-readouts: id "${entry.id}" has the same weightsSha256 as already-archived id ` +
          `"${existingIdForWeights}" -- the same theta_final.npy was archived under two different labels ` +
          `(a copy-paste --source error, or a rerun that never actually happened)?`
      );
    }
    byId.set(entry.id, entry);
    idByWeightsSha256.set(entry.weightsSha256, entry.id);
  }
  return [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
};

export const writeArchive = (path: string, readouts: readonly ArchivedReadout[]): void => {
  const archive: TrainedReadoutArchive = { version: 1, readouts };
  mkdirSync(dirname(path), { recursive: true });
  atomicWriteFileSync(path, JSON.stringify(archive));
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/**
 * `public/data/malecns-arena-v1.manifest.json` -- a fixed, already-committed
 * repo file (never an agent-worktree path), so bigq graph identity is always
 * resolvable with no extra CLI flag, unlike the intervention index (which
 * only exists in gitignored worktrees until `--intervention-index` copies
 * it in).
 */
export const DEFAULT_MANIFEST_PATH = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');

/**
 * Resolve every addition's path-independent `graphGzipSha256`/
 * `graphBinarySha256` (thermo-provenance review finding -- see the module
 * doc comment's "Caveat for WP2"), returning a NEW array (additions are
 * immutable `readonly` records) rather than mutating in place.
 *
 * bigq additions are always resolved, from the fixed `manifestPath` -- no
 * `--intervention-index` needed. Intervention/task-intervention additions
 * REQUIRE `interventionIndex` (throws if any such addition is present but
 * `--intervention-index` was not supplied in this invocation): unlike bigq,
 * there is no fixed, always-available repo file to resolve them from.
 */
export const attachGraphIdentity = (
  additions: readonly ArchivedReadout[],
  ctx: {
    readonly manifestPath: string;
    readonly interventionIndex: { readonly index: GraphListIndex; readonly indexDir: string } | null;
  }
): readonly ArchivedReadout[] =>
  additions.map((entry) => {
    if (entry.kind === 'bigq') {
      const identity = resolveBigqGraphIdentity(entry.graphId, ctx.manifestPath);
      return applyGraphIdentity(entry, identity);
    }
    if (!ctx.interventionIndex) {
      throw new Error(
        `archive-readouts: --source ${entry.graphId} (kind "${entry.kind}") needs --intervention-index to resolve ` +
          'a path-independent graph identity, but none was supplied in this invocation'
      );
    }
    const identity = resolveInterventionGraphIdentity(entry.graphId, ctx.interventionIndex.index, ctx.interventionIndex.indexDir);
    return applyGraphIdentity(entry, identity);
  });

const applyGraphIdentity = (entry: ArchivedReadout, identity: GraphIdentity | null): ArchivedReadout => ({
  ...entry,
  graphGzipSha256: identity?.graphGzipSha256 ?? null,
  graphBinarySha256: identity?.graphBinarySha256 ?? null
});

/**
 * WP1b (`01-archive-and-types.md`): "The per-task runs use the same
 * intervention graphs as the default task, and archive-readouts must verify
 * this rather than assume it." `attachGraphIdentity` above already RESOLVES
 * each `task-intervention` addition's `graphGzipSha256`/`graphBinarySha256`
 * independently -- re-verified against the actual graph bytes on disk for
 * *this* invocation's `--intervention-index` (`graph-identity.ts`). This is
 * the separate cross-check that the resolved identity actually lands on the
 * SAME graph the already-archived default-task `"intervention"` entry for
 * this `graphId` has on file -- not merely that both happened to be resolved
 * from an `--intervention-index` file that claims the same id (a caller
 * could point `--intervention-index` at a different, stale copy of the graph
 * list for a per-task invocation; this catches that rather than silently
 * trusting it, because it compares against the archive's own already-verified
 * record, not against the input the current invocation is trying to verify).
 * A no-op for any other `kind`.
 */
export const assertTaskGraphMatchesDefaultGraph = (
  addition: Readonly<ArchivedReadout>,
  existingReadouts: readonly ArchivedReadout[]
): void => {
  if (addition.kind !== 'task-intervention') return;
  const defaultEntry = existingReadouts.find((entry) => entry.kind === 'intervention' && entry.graphId === addition.graphId);
  if (!defaultEntry) {
    throw new Error(
      `archive-readouts: task-intervention entry "${addition.id}" (graphId "${addition.graphId}") has no already-` +
        'archived default-task "intervention" entry for that graphId to verify its graph against -- archive the ' +
        'default-task WP1 entries first'
    );
  }
  if (
    addition.graphGzipSha256 !== defaultEntry.graphGzipSha256 ||
    addition.graphBinarySha256 !== defaultEntry.graphBinarySha256
  ) {
    throw new Error(
      `archive-readouts: task-intervention entry "${addition.id}"'s graph (gzip ${addition.graphGzipSha256}, ` +
        `binary ${addition.graphBinarySha256}) does not match the default-task "${defaultEntry.id}" entry's graph ` +
        `(gzip ${defaultEntry.graphGzipSha256}, binary ${defaultEntry.graphBinarySha256}) for graphId ` +
        `"${addition.graphId}" -- per-task readouts must use the same intervention graphs as the default task`
    );
  }
};

export interface RunArchiveReadoutsResult {
  readonly out: string;
  readonly readoutCount: number;
  /** Paths this invocation actually wrote (in write order) -- never the `DEFAULT_*` constants, so the CLI summary can't name a file this invocation didn't touch (a dual-review finding). */
  readonly wrote: readonly string[];
}

/**
 * Builds and validates every requested addition/copy/extraction BEFORE any
 * file is written, so a late failure (e.g. a malformed `--intervention-index`
 * file, or an intervention `--source` whose `armBundleSha256` disagrees with
 * `--raw-intervention-scores`) cannot leave an earlier write (e.g. the
 * readouts archive) applied while a later one is skipped -- a dual-review
 * finding: the previous version wrote (or, for the index/swaps files,
 * validated-while-writing) each output in sequence, so a later failure left
 * the earlier writes on disk.
 */
export const runArchiveReadouts = (args: Readonly<ArchiveReadoutsArgs>): RunArchiveReadoutsResult => {
  // -- Phase 1: read, parse, and validate every input; write nothing yet. --
  const rawScores = args.rawInterventionScoresPath ? readRawInterventionScores(args.rawInterventionScoresPath) : null;
  const interventionIndexResult = args.interventionIndexPath
    ? readAndValidateInterventionIndex(args.interventionIndexPath)
    : null;
  const swapsArchive = args.interventionAttributionPath
    ? buildInterventionSwapsArchive(args.interventionAttributionPath)
    : null;

  // WP1b: `--task-intervention-raw-scores` requires `--intervention-index` in
  // the SAME invocation -- there is no fixed, always-available repo file to
  // check `graphListSha256` against (mirrors `attachGraphIdentity`'s own
  // requirement for `task-intervention` `--source` additions).
  if (args.taskInterventionRawScores.length > 0 && !interventionIndexResult) {
    throw new Error(
      'archive-readouts: --task-intervention-raw-scores needs --intervention-index in the same invocation, to ' +
        "verify each task's graphListSha256 against the same intervention graph list the per-task readouts resolve against"
    );
  }
  const taskRawFiles = args.taskInterventionRawScores.map((spec) => {
    const read = readRawInterventionScores(spec.path, { expectedArenaTask: spec.arenaTask });
    // "The per-task runs use the same intervention graphs as the default
    // task, and archive-readouts must verify this rather than assume it"
    // (01-archive-and-types.md) -- checked here at the raw-scores level too,
    // not only per readout (`assertTaskGraphMatchesDefaultGraph` below):
    // this task's own rescore was run against the exact byte-verified
    // `--intervention-index` this invocation is using, not merely a
    // same-shaped file.
    const indexSha256 = sha256Hex(interventionIndexResult!.rawBytes);
    if (read.parsed.graphListSha256 !== indexSha256) {
      throw new Error(
        `archive-readouts: ${spec.path}'s graphListSha256 ${read.parsed.graphListSha256} does not match this ` +
          `invocation's --intervention-index sha256 ${indexSha256} -- the per-task rescore was not run against ` +
          'the same intervention graph list the default task uses'
      );
    }
    return { arenaTask: spec.arenaTask, path: spec.path, ...read };
  });

  let additions: readonly ArchivedReadout[] = args.sources.map(buildArchivedReadout);
  additions = attachGraphIdentity(additions, {
    manifestPath: DEFAULT_MANIFEST_PATH,
    interventionIndex: interventionIndexResult
      ? { index: interventionIndexResult.index, indexDir: interventionIndexResult.indexDir }
      : null
  });
  if (rawScores) assertArmBundlesMatchRawScores(additions, rawScores.runs);
  for (const taskRawFile of taskRawFiles) {
    assertArmBundlesMatchRawScores(additions, taskRawFile.runs, { kind: 'task-intervention', arenaTask: taskRawFile.arenaTask });
  }

  const existing = args.sources.length > 0 ? loadExistingArchive(args.out) : null;
  const existingReadouts = existing?.readouts ?? [];
  for (const addition of additions) assertTaskGraphMatchesDefaultGraph(addition, existingReadouts);
  const mergedReadouts = args.sources.length > 0 ? mergeReadouts(existingReadouts, additions) : null;

  // -- Phase 2: every input validated -- now write. --
  const wrote: string[] = [];
  if (mergedReadouts) {
    writeArchive(args.out, mergedReadouts);
    wrote.push(args.out);
  }
  if (rawScores && args.rawInterventionScoresPath) {
    copyVerbatim(args.rawInterventionScoresPath, args.rawInterventionScoresOut, rawScores.rawBytes);
    wrote.push(args.rawInterventionScoresOut);
  }
  if (interventionIndexResult && args.interventionIndexPath) {
    copyVerbatim(args.interventionIndexPath, args.interventionIndexOut, interventionIndexResult.rawBytes);
    wrote.push(args.interventionIndexOut);
  }
  if (swapsArchive) {
    mkdirSync(dirname(args.interventionSwapsOut), { recursive: true });
    atomicWriteFileSync(args.interventionSwapsOut, JSON.stringify(swapsArchive));
    wrote.push(args.interventionSwapsOut);
  }
  for (const taskRawFile of taskRawFiles) {
    const outPath = taskInterventionRawScoresOutPath(args.taskInterventionRawScoresOutDir, taskRawFile.arenaTask);
    copyVerbatim(taskRawFile.path, outPath, taskRawFile.rawBytes);
    wrote.push(outPath);
  }

  const readoutCount = mergedReadouts?.length ?? loadExistingArchive(args.out)?.readouts.length ?? 0;

  return { out: args.out, readoutCount, wrote };
};

const main = (): void => {
  try {
    const result = runArchiveReadouts(parseArgs(process.argv.slice(2)));
    // eslint-disable-next-line no-console -- CLI tool: user-facing summary.
    console.log(
      `archive-readouts: ${result.readoutCount} readout(s) in ${result.out}` +
        result.wrote
          .filter((path) => path !== result.out)
          .map((path) => `; wrote ${path}`)
          .join('')
    );
  } catch (error) {
    // eslint-disable-next-line no-console -- CLI tool: user-facing error.
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
