import { randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { basename, dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { readNpyFloat32Array } from '../training/npy';
import { readRunDir, type LoadedRun } from '../training/run-dir';
import type { ArmName } from '../training/arms';

/**
 * Same atomic-write contract as `../training/fsio.ts`'s `atomicWriteFileSync`
 * (same-directory temp file + `renameSync`), but over a `Buffer` rather than
 * a `string` -- kept as a LOCAL copy, not a change to `fsio.ts`'s own
 * exported signature, because `fsio.ts` is a dependency (via
 * `collectRepoRelativeDependencies`) of several already-published artifacts'
 * own `producer.sourceSha256` (e.g. `repertoire-report.ts`'s
 * `repertoireNullProducer`); editing it -- even a purely additive signature
 * widening -- changes those artifacts' recomputed source identity and
 * spuriously fails `tests/unit/repertoire-report.test.ts`'s "the committed
 * artifact was produced by the producer at HEAD" check, which has nothing to
 * do with this WP (verified empirically while developing this diff).
 */
const atomicWriteBufferSync = (path: string, contents: Buffer): void => {
  const tmpPath = resolve(dirname(path), `.${basename(path)}.${randomBytes(6).toString('hex')}.tmp`);
  try {
    writeFileSync(tmpPath, contents);
    renameSync(tmpPath, path);
  } catch (error) {
    try {
      unlinkSync(tmpPath);
    } catch {
      // tmpPath was never created, or was already cleaned up -- nothing more to do.
    }
    throw error;
  }
};

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
 * `export-arms.ts`'s `computeArmBundleSha256`): `armBundleSha256` hashes the
 * *entire* serialized arm bundle, including `provenance.artifactPath` -- the
 * literal, worktree-specific filesystem path `export-arms.ts --rewired` was
 * invoked with. So `armBundleSha256` is not reproducible from a graph
 * regenerated at a different path (e.g. after the source worktree is pruned
 * and `interventions.py` is rerun elsewhere) even when the regenerated
 * graph's bytes are identical. The path-INDEPENDENT identity is each
 * intervention graph's `gzipSha256`/`binarySha256`
 * (`training/archive/intervention-index-v1.json`, keyed by `graphId`) --
 * WP2 should verify a regenerated graph against those, not against
 * `armBundleSha256`, and treat `armBundleSha256` as this specific archived
 * run's own training-time provenance rather than a durable graph identity.
 */

const here = dirname(fileURLToPath(import.meta.url));
export const repoRoot = resolve(here, '../..');

export const DEFAULT_OUT = resolve(repoRoot, 'training/archive/trained-readouts-v1.json');
export const DEFAULT_RAW_INTERVENTION_SCORES_OUT = resolve(repoRoot, 'training/archive/intervention-trained-raw-v1.json');
export const DEFAULT_INTERVENTION_INDEX_OUT = resolve(repoRoot, 'training/archive/intervention-index-v1.json');
export const DEFAULT_INTERVENTION_SWAPS_OUT = resolve(repoRoot, 'training/archive/intervention-swaps-v1.json');

// ---------------------------------------------------------------------------
// Archive schema
// ---------------------------------------------------------------------------

export type ArchivedReadoutKind = 'bigq' | 'intervention' | 'task-intervention';

export interface ArchivedReadout {
  /** `"<graphId>-seed<trainerSeed>[-<idSuffix>]"` -- unique within the archive (see `mergeReadouts`). Not guaranteed injective against an adversarially-chosen `graphId`/`idSuffix` containing `"-seed<digits>"` itself (string concatenation, not a structured key) -- not a live risk for this archive's own fixed `graphId` vocabulary (which does include the literal `"rewired-seed0"`, so a naive "reject `-seed\d` in labels" guard would incorrectly reject real data), but `mergeReadouts`' identity check is the actual safety net against a collision, not this format. */
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
  readonly graphArtifactSha256: string;
  readonly armBundleSha256: string;
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

export interface ArchiveReadoutsArgs {
  readonly sources: readonly SourceSpec[];
  readonly out: string;
  readonly rawInterventionScoresPath: string | null;
  readonly rawInterventionScoresOut: string;
  readonly interventionIndexPath: string | null;
  readonly interventionIndexOut: string;
  readonly interventionAttributionPath: string | null;
  readonly interventionSwapsOut: string;
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
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (
    sources.length === 0 &&
    rawInterventionScoresPath === null &&
    interventionIndexPath === null &&
    interventionAttributionPath === null
  ) {
    throw new Error(
      'archive-readouts: at least one of --source/--raw-intervention-scores/--intervention-index/' +
        '--intervention-attribution is required'
    );
  }
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
    interventionSwapsOut
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
  const arenaTaskId = resolveArenaTask(config.arenaTask).id;

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

  const id = `${spec.graphId}-seed${config.trainerSeed}${spec.idSuffix ? `-${spec.idSuffix}` : ''}`;

  return {
    id,
    arm: config.arm,
    kind: kindForEntry(spec.graphId, arenaTaskId),
    graphId: spec.graphId,
    trainerSeed: config.trainerSeed,
    arenaTask: arenaTaskId,
    graphArtifactSha256,
    armBundleSha256: config.armBundleSha256,
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
  'armBundleSha256',
  'graphArtifactSha256',
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
// training/archive/intervention-trained-raw-v1.json
// ---------------------------------------------------------------------------

/**
 * `null-trained-evaluate-graph-list.ts`'s `NullTrainedInterventionEvaluationRaw`
 * (id-keyed `runs`), the exact raw output of the hbru worktree's
 * `--graph-list`/`--runs` rescore -- the only place the published
 * `pathway-interventions-v1.json` artifact's sorted, unlabeled C/M score
 * arrays can be traced back to a specific `(id, trainerSeed)` pair. Copied
 * (not re-derived) so that id mapping survives the hbru worktree being
 * pruned. Only lightly validated here (has a non-empty `runs` array of
 * well-shaped entries) -- this is a copy of an already-produced, already
 * cross-checked artifact, not a second independent computation.
 */
interface RawInterventionRun {
  readonly id: string;
  readonly trainerSeed: number;
  readonly armBundleSha256?: string;
  readonly movementScore: readonly number[];
}

/**
 * Write `bytes` to `outPath` unmodified, then re-read and re-hash the
 * written file to confirm the write itself didn't alter anything -- a
 * byte-for-byte copy of an externally-produced artifact (this file's whole
 * purpose is preserving its exact bytes against worktree pruning, so a
 * `string` decode/re-encode round trip, or any other silent reformatting,
 * would defeat it -- a dual-review finding).
 */
const copyVerbatim = (sourcePath: string, outPath: string, bytes: Buffer): void => {
  mkdirSync(dirname(outPath), { recursive: true });
  atomicWriteBufferSync(outPath, bytes);
  const expected = sha256Hex(bytes);
  const actual = sha256Hex(readFileSync(outPath));
  if (actual !== expected) {
    throw new Error(`archive-readouts: ${outPath} sha256 ${actual} does not match ${sourcePath}'s ${expected} after copying`);
  }
};

/** Read+validate (not copy) -- split out from `copyRawInterventionScores` so `runArchiveReadouts` can cross-check `--source` additions against this file's `(id, trainerSeed) -> armBundleSha256` map in the same invocation, before either is written. */
const readRawInterventionScores = (
  sourcePath: string
): { readonly rawBytes: Buffer; readonly runs: readonly RawInterventionRun[] } => {
  const rawBytes = readFileSync(sourcePath);
  const parsed = JSON.parse(rawBytes.toString('utf8')) as { readonly runs?: readonly RawInterventionRun[] };
  if (!Array.isArray(parsed.runs) || parsed.runs.length === 0) {
    throw new Error(`archive-readouts: ${sourcePath} has no "runs" array`);
  }
  for (const run of parsed.runs) {
    if (
      typeof run.id !== 'string' ||
      run.id.length === 0 ||
      typeof run.trainerSeed !== 'number' ||
      !Array.isArray(run.movementScore)
    ) {
      throw new Error(`archive-readouts: ${sourcePath} has a malformed run entry: ${JSON.stringify(run)}`);
    }
  }
  return { rawBytes, runs: parsed.runs };
};

export const copyRawInterventionScores = (sourcePath: string, outPath: string): void => {
  const { rawBytes } = readRawInterventionScores(sourcePath);
  copyVerbatim(sourcePath, outPath, rawBytes);
};

/**
 * Cross-check every `kind: "intervention"` addition's `armBundleSha256`
 * against the raw scores file's own `(graphId, trainerSeed) -> armBundleSha256`
 * mapping -- a stronger check than `assertArmMatchesGraphId` alone (which
 * cannot distinguish e.g. a `C000` label from a `C001` run, since both are
 * `arm: "rewired"`), available whenever `--source` and
 * `--raw-intervention-scores` are supplied together in one invocation (a
 * dual-review finding). Silently skips an addition with no matching raw
 * entry (e.g. a bigq addition, or an intervention id the raw file doesn't
 * cover) rather than requiring full coverage in either direction.
 */
export const assertArmBundlesMatchRawScores = (
  additions: readonly ArchivedReadout[],
  rawRuns: readonly RawInterventionRun[]
): void => {
  const rawByKey = new Map(rawRuns.map((run) => [`${run.id}\u0000${run.trainerSeed}`, run]));
  for (const entry of additions) {
    if (entry.kind !== 'intervention') continue;
    const match = rawByKey.get(`${entry.graphId}\u0000${entry.trainerSeed}`);
    if (match && typeof match.armBundleSha256 === 'string' && match.armBundleSha256 !== entry.armBundleSha256) {
      throw new Error(
        `archive-readouts: --source ${entry.graphId} (trainerSeed ${entry.trainerSeed}) has armBundleSha256 ` +
          `${entry.armBundleSha256}, but the raw intervention scores file records ${match.armBundleSha256} for ` +
          `id "${entry.graphId}" -- mismatched --source label, or wrong run directory?`
      );
    }
  }
};

// ---------------------------------------------------------------------------
// training/archive/intervention-index-v1.json
// ---------------------------------------------------------------------------

/**
 * `graph-list-index.ts`'s `GraphListIndex` (`scripts/analysis/interventions.py`'s
 * `index.json`): copied byte-for-byte (never re-serialized) so this file's
 * own sha256 stays equal to `pathway-interventions-v1.json`'s recorded
 * `sources.indexSha` -- the one independent, already-published check this
 * archive's copy can be verified against -- and so WP2's regenerated
 * intervention graphs can be sha-checked against this repository's own
 * committed record, without depending on any worktree.
 */
interface RawGraphListIndex {
  readonly sourceArtifact?: unknown;
  readonly sourceSha256?: unknown;
  readonly entries?: readonly unknown[];
}

/** Read+validate (not copy) -- split out so `runArchiveReadouts` can validate this input before writing anything, matching `readRawInterventionScores`'s split. */
const readGraphListIndexBytes = (sourcePath: string): Buffer => {
  const rawBytes = readFileSync(sourcePath);
  const parsed = JSON.parse(rawBytes.toString('utf8')) as RawGraphListIndex;
  if (typeof parsed.sourceSha256 !== 'string' || !Array.isArray(parsed.entries) || parsed.entries.length === 0) {
    throw new Error(`archive-readouts: ${sourcePath} is not a valid graph-list index.json`);
  }
  return rawBytes;
};

export const copyInterventionIndex = (sourcePath: string, outPath: string): void => {
  copyVerbatim(sourcePath, outPath, readGraphListIndexBytes(sourcePath));
};

// ---------------------------------------------------------------------------
// training/archive/intervention-swaps-v1.json
// ---------------------------------------------------------------------------

export interface SwapEdge {
  readonly pre: number;
  readonly post: number;
}

interface AttributionStep {
  readonly step: number;
  readonly accepted: boolean;
  readonly addedEdge: SwapEdge;
  readonly addedEdge2: SwapEdge;
  readonly removedEdge: SwapEdge;
  readonly removedEdge2: SwapEdge;
}

interface AttributionEntry {
  readonly kind: string;
  readonly steps: readonly AttributionStep[];
  /** `interventions.py`'s own count of accepted swap steps -- cross-checked against `steps.filter(accepted).length` below, a free integrity check on the parse. */
  readonly swaps?: number;
}

export interface InterventionSwapEntry {
  readonly id: string;
  /** Every accepted step's `addedEdge`/`addedEdge2`, in step order -- H3 uses these edges' thrust endpoints as "newly connected thrust neurons" (`00-overview.md`). */
  readonly addedEdges: readonly SwapEdge[];
  /** Every accepted step's `removedEdge`/`removedEdge2`, in step order. */
  readonly removedEdges: readonly SwapEdge[];
}

export interface InterventionSwapsArchive {
  readonly version: 1;
  /** sha256 of the source `attribution.json`'s raw bytes -- provenance, not a value with a separately-published record to check it against (unlike `intervention-index-v1.json`'s `sourceSha256`, which `pathway-interventions-v1.json`'s `sources.indexSha` does independently confirm). */
  readonly sourceSha256: string;
  /** `P` then `Q`, in that order -- `00-overview.md`'s H3 only ever needs `P`; `Q` is archived alongside it since `interventions.py` produces both from one run and a future analysis may need it. */
  readonly swaps: readonly InterventionSwapEntry[];
}

const isSwapEdge = (value: unknown): value is SwapEdge =>
  typeof value === 'object' &&
  value !== null &&
  Number.isInteger((value as Partial<SwapEdge>).pre) &&
  Number.isInteger((value as Partial<SwapEdge>).post);

const extractSwapsFor = (id: string, entry: AttributionEntry | undefined): InterventionSwapEntry => {
  if (!entry || !Array.isArray(entry.steps)) {
    throw new Error(`archive-readouts: attribution.json has no "${id}" entry with a "steps" array`);
  }
  const accepted = entry.steps.filter((step) => step.accepted === true);
  if (typeof entry.swaps === 'number' && accepted.length !== entry.swaps) {
    throw new Error(
      `archive-readouts: attribution.json's "${id}" entry has ${accepted.length} accepted step(s) but ` +
        `swaps=${entry.swaps}`
    );
  }
  const addedEdges: SwapEdge[] = [];
  const removedEdges: SwapEdge[] = [];
  for (const step of accepted) {
    for (const edge of [step.addedEdge, step.addedEdge2, step.removedEdge, step.removedEdge2]) {
      if (!isSwapEdge(edge)) {
        throw new Error(`archive-readouts: attribution.json's "${id}" step ${step.step} has a malformed edge`);
      }
    }
    addedEdges.push({ pre: step.addedEdge.pre, post: step.addedEdge.post }, { pre: step.addedEdge2.pre, post: step.addedEdge2.post });
    removedEdges.push(
      { pre: step.removedEdge.pre, post: step.removedEdge.post },
      { pre: step.removedEdge2.pre, post: step.removedEdge2.post }
    );
  }
  return { id, addedEdges, removedEdges };
};

/** Read+validate+build (not write) -- split out so `runArchiveReadouts` can validate this input before writing anything, matching `readRawInterventionScores`'s split. */
const buildInterventionSwapsArchive = (sourcePath: string): InterventionSwapsArchive => {
  const raw = readFileSync(sourcePath);
  const parsed = JSON.parse(raw.toString('utf8')) as { readonly P?: AttributionEntry; readonly Q?: AttributionEntry };
  const swaps = [extractSwapsFor('P', parsed.P), extractSwapsFor('Q', parsed.Q)];
  return { version: 1, sourceSha256: sha256Hex(raw), swaps };
};

export const extractInterventionSwaps = (sourcePath: string, outPath: string): void => {
  const archive = buildInterventionSwapsArchive(sourcePath);
  mkdirSync(dirname(outPath), { recursive: true });
  atomicWriteFileSync(outPath, JSON.stringify(archive));
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

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
  const additions = args.sources.map(buildArchivedReadout);
  const existing = args.sources.length > 0 ? loadExistingArchive(args.out) : null;
  const mergedReadouts = args.sources.length > 0 ? mergeReadouts(existing?.readouts ?? [], additions) : null;

  const rawScores = args.rawInterventionScoresPath ? readRawInterventionScores(args.rawInterventionScoresPath) : null;
  if (rawScores) assertArmBundlesMatchRawScores(additions, rawScores.runs);

  const indexBytes = args.interventionIndexPath ? readGraphListIndexBytes(args.interventionIndexPath) : null;
  const swapsArchive = args.interventionAttributionPath
    ? buildInterventionSwapsArchive(args.interventionAttributionPath)
    : null;

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
  if (indexBytes && args.interventionIndexPath) {
    copyVerbatim(args.interventionIndexPath, args.interventionIndexOut, indexBytes);
    wrote.push(args.interventionIndexOut);
  }
  if (swapsArchive) {
    mkdirSync(dirname(args.interventionSwapsOut), { recursive: true });
    atomicWriteFileSync(args.interventionSwapsOut, JSON.stringify(swapsArchive));
    wrote.push(args.interventionSwapsOut);
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
