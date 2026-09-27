import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { readNpyFloat32Array } from '../training/npy';
import { readRunDir, type LoadedRun } from '../training/run-dir';
import type { ArmName } from '../training/arms';

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
 * of the same underlying MaleCNS graph). So `graphId` (identifying which
 * topology -- `"biological"`, `"rewired-seed0"`, `"disconnected"`, `"P"`,
 * `"C000"`, `"M1000"`, ...) is supplied by this script's caller (the
 * `--source` label), not inferred from the run directory, and
 * `armBundleSha256` (the run's own `config.json` field, already checked
 * against the arm bundle's self-certifying hash at training time --
 * `null-trained-worker.ts`'s `runTask`) is the proof that a given `graphId`
 * label is not merely an operator's claim: WP2's analyses resolve a
 * readout's graph through `graphId` + `armBundleSha256` together, and this
 * archive refuses to record an entry whose `armBundleSha256` is missing.
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
  /** `"<graphId>-seed<trainerSeed>[-<idSuffix>]"` -- unique within the archive (see `mergeReadouts`). */
  readonly id: string;
  readonly arm: ArmName;
  readonly kind: ArchivedReadoutKind;
  readonly graphId: string;
  readonly trainerSeed: number;
  /** `run.config.arenaTask`, or `'default'` when the run directory predates arena tasks. */
  readonly arenaTask: string;
  readonly graphArtifactSha256: string;
  readonly armBundleSha256: string;
  readonly D: number;
  readonly H: number;
  readonly parameterCount: number;
  /** sha256 of `theta_final.npy`'s raw file bytes (header + data) -- distinct from `run-dir.ts`'s `weightsSha256`, which hashes only the decoded float buffer. */
  readonly thetaSha256: string;
  /** The flat `theta_final.npy` payload, base64-encoded little-endian float32 (`[w1, b1, w2, b2]`, `run-dir.ts`'s documented layout). */
  readonly theta: string;
  /** `sourcePath`, relative to the repository root when the run directory is inside it (an agent worktree is not, so an absolute path is kept in that case -- still enough provenance to relocate it). */
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
  const idSuffixRaw = colon === -1 ? '' : spec.slice(colon + 1);
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
  if (!out.endsWith('.json')) throw new Error(`--out must end with ".json" (got "${out}")`);

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

/** `sourcePath` relative to the repository root when possible (every real WP1 source is an absolute path to another worktree, well outside `repoRoot` -- `relative` still produces a valid, if `../`-heavy, relative path in that case, which is fine: it is provenance, not something ever re-resolved from the archive itself). */
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

  const thetaPath = resolve(spec.runDir, 'theta_final.npy');
  // sha256 of the raw .npy FILE bytes (header + data) -- distinct from
  // `run.weightsSha256` (`run-dir.ts`), which hashes only the decoded flat
  // float32 buffer and so is insensitive to the .npy header entirely. This
  // archive's own `theta` field below is a base64 re-encoding of the decoded
  // payload (matching `readout-serialization.ts`'s `decodeBase64Float32`
  // format), so `thetaSha256` is recorded over the file this repository
  // actually shipped, not over a value this script re-derives.
  const thetaSha256 = sha256Hex(readFileSync(thetaPath));
  const theta = readNpyFloat32Array(thetaPath);
  const thetaBase64 = Buffer.from(theta.buffer, theta.byteOffset, theta.byteLength).toString('base64');

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
    theta: thetaBase64,
    sourcePath: sourcePathFor(spec.runDir)
  };
};

export const loadExistingArchive = (path: string): TrainedReadoutArchive | null => {
  if (!existsSync(path)) return null;
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<TrainedReadoutArchive>;
  if (!Array.isArray(parsed.readouts)) {
    throw new Error(`archive-readouts: ${path} has no "readouts" array`);
  }
  return { version: 1, readouts: parsed.readouts };
};

/**
 * Merge `additions` into `existing`, sorted by `id` ascending (a
 * deterministic archive, independent of `--source` argument order or which
 * invocation added which entry -- matches this codebase's existing
 * "canonical task order" convention, e.g. `null-evaluate.ts`). A duplicate
 * `id` is only ever accepted when its `thetaSha256` is byte-identical to the
 * entry already on file: re-running this script against the same run
 * directories (idempotent) must succeed, but archiving two DIFFERENT runs
 * under one id (an operator error, or a stale/rebuilt run directory) must
 * not silently overwrite the earlier entry.
 */
export const mergeReadouts = (
  existing: readonly ArchivedReadout[],
  additions: readonly ArchivedReadout[]
): readonly ArchivedReadout[] => {
  const byId = new Map<string, ArchivedReadout>();
  for (const entry of existing) byId.set(entry.id, entry);
  for (const entry of additions) {
    const prior = byId.get(entry.id);
    if (prior && prior.thetaSha256 !== entry.thetaSha256) {
      throw new Error(
        `archive-readouts: id "${entry.id}" already exists in the archive with a different thetaSha256 ` +
          `(${prior.thetaSha256} vs ${entry.thetaSha256}) -- refusing to silently overwrite it`
      );
    }
    byId.set(entry.id, entry);
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
  readonly movementScore: readonly number[];
}

export const copyRawInterventionScores = (sourcePath: string, outPath: string): void => {
  const raw = readFileSync(sourcePath, 'utf8');
  const parsed = JSON.parse(raw) as { readonly runs?: readonly RawInterventionRun[] };
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
  // A byte-for-byte copy of `raw` (not a `JSON.stringify(parsed)` re-encode):
  // this file's whole purpose is preserving this exact artifact's bytes
  // against worktree pruning, and a re-encode (different float formatting,
  // key order, etc from whatever originally produced it) would silently
  // change its own sha256 away from any sha recorded elsewhere against the
  // original file.
  mkdirSync(dirname(outPath), { recursive: true });
  atomicWriteFileSync(outPath, raw);
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

export const copyInterventionIndex = (sourcePath: string, outPath: string): void => {
  const raw = readFileSync(sourcePath, 'utf8');
  const parsed = JSON.parse(raw) as RawGraphListIndex;
  if (typeof parsed.sourceSha256 !== 'string' || !Array.isArray(parsed.entries) || parsed.entries.length === 0) {
    throw new Error(`archive-readouts: ${sourcePath} is not a valid graph-list index.json`);
  }
  mkdirSync(dirname(outPath), { recursive: true });
  atomicWriteFileSync(outPath, raw);
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

const extractSwapsFor = (id: string, entry: AttributionEntry | undefined): InterventionSwapEntry => {
  if (!entry || !Array.isArray(entry.steps)) {
    throw new Error(`archive-readouts: attribution.json has no "${id}" entry with a "steps" array`);
  }
  const accepted = entry.steps.filter((step) => step.accepted === true);
  const addedEdges: SwapEdge[] = [];
  const removedEdges: SwapEdge[] = [];
  for (const step of accepted) {
    addedEdges.push(step.addedEdge, step.addedEdge2);
    removedEdges.push(step.removedEdge, step.removedEdge2);
  }
  return { id, addedEdges, removedEdges };
};

export const extractInterventionSwaps = (sourcePath: string, outPath: string): void => {
  const raw = readFileSync(sourcePath);
  const parsed = JSON.parse(raw.toString('utf8')) as { readonly P?: AttributionEntry; readonly Q?: AttributionEntry };
  const swaps = [extractSwapsFor('P', parsed.P), extractSwapsFor('Q', parsed.Q)];
  const archive: InterventionSwapsArchive = { version: 1, sourceSha256: sha256Hex(raw), swaps };
  mkdirSync(dirname(outPath), { recursive: true });
  atomicWriteFileSync(outPath, JSON.stringify(archive));
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

export interface RunArchiveReadoutsResult {
  readonly out: string;
  readonly readoutCount: number;
  readonly wroteRawInterventionScores: boolean;
  readonly wroteInterventionIndex: boolean;
  readonly wroteInterventionSwaps: boolean;
}

export const runArchiveReadouts = (args: Readonly<ArchiveReadoutsArgs>): RunArchiveReadoutsResult => {
  let readoutCount = 0;
  if (args.sources.length > 0) {
    const additions = args.sources.map(buildArchivedReadout);
    const existing = loadExistingArchive(args.out);
    const readouts = mergeReadouts(existing?.readouts ?? [], additions);
    writeArchive(args.out, readouts);
    readoutCount = readouts.length;
  } else {
    readoutCount = loadExistingArchive(args.out)?.readouts.length ?? 0;
  }

  if (args.rawInterventionScoresPath) {
    copyRawInterventionScores(args.rawInterventionScoresPath, args.rawInterventionScoresOut);
  }
  if (args.interventionIndexPath) {
    copyInterventionIndex(args.interventionIndexPath, args.interventionIndexOut);
  }
  if (args.interventionAttributionPath) {
    extractInterventionSwaps(args.interventionAttributionPath, args.interventionSwapsOut);
  }

  return {
    out: args.out,
    readoutCount,
    wroteRawInterventionScores: args.rawInterventionScoresPath !== null,
    wroteInterventionIndex: args.interventionIndexPath !== null,
    wroteInterventionSwaps: args.interventionAttributionPath !== null
  };
};

const main = (): void => {
  try {
    const result = runArchiveReadouts(parseArgs(process.argv.slice(2)));
    // eslint-disable-next-line no-console -- CLI tool: user-facing summary.
    console.log(
      `archive-readouts: ${result.readoutCount} readout(s) in ${result.out}` +
        (result.wroteRawInterventionScores ? `; wrote ${DEFAULT_RAW_INTERVENTION_SCORES_OUT}` : '') +
        (result.wroteInterventionIndex ? `; wrote ${DEFAULT_INTERVENTION_INDEX_OUT}` : '') +
        (result.wroteInterventionSwaps ? `; wrote ${DEFAULT_INTERVENTION_SWAPS_OUT}` : '')
    );
  } catch (error) {
    // eslint-disable-next-line no-console -- CLI tool: user-facing error.
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
