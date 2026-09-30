import { mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../lib/import-graph';
import { sortKeysDeep, verifyManifestRoundTrips } from '../null/null-report';
import { assertSameArchive, computeArchiveSha256, DEFAULT_ARCHIVE_PATH, DEFAULT_MANIFEST_PATH, loadArchive } from './shared';
import type { ArchivedReadout } from './archive-readouts';
import { renderReadoutAttributionReportMarkdown } from './attribution-report-markdown';

// Re-exported (not just imported) so callers/tests of this module can reach
// the shared archive/manifest default paths through this file's own public
// surface -- mirrors every other `DEFAULT_*` constant this file exports.
export { DEFAULT_ARCHIVE_PATH, DEFAULT_MANIFEST_PATH };

/**
 * WP3 of `.agents/plans/readout-attribution` (`03-artifact-and-findings.md`):
 * combines WP1's committed archive (`training/archive/trained-readouts-v1.json`)
 * with WP2's five analysis outputs (`training/runs/attribution/{saliency,
 * independence,linkage,regime,ablation}.json`, gitignored scratch) and the
 * hypothesis-evaluation output (`.../hypotheses.json`) into
 * `public/data/readout-attribution-v1.json` (manifest key
 * `readoutAttribution`) and `docs/readout-attribution-report.md`. Same
 * "one producer script combining upstream per-item JSON into one artifact +
 * report" shape `scripts/selections/selection-report.ts`/
 * `scripts/null/task-generality-report.ts` already establish -- this file
 * holds the types, the pure builder, and the CLI; `./attribution-report-markdown.ts`
 * (a separate file, following `selection-report.ts`/`selection-report-markdown.ts`'s
 * own split) holds the report renderer, so this file stays well under the
 * repo's 1000-line-per-file limit.
 *
 * Every WP2 output is cross-checked against the archive's own current
 * sha256 (`assertSameArchive`, this WP's own "producer must cross-check
 * each input's archiveSha256 stamp" requirement) before anything is
 * trusted -- a stale intermediate left over from a different archive
 * version is a loud, immediate error instead of a silently wrong published
 * artifact. Every number in the artifact is read once from an
 * already-computed WP2 output (never re-simulated); this producer performs
 * no new simulation of its own.
 *
 * Coverage is scoped to the 23 default-task archive entries only
 * (`archive-readouts.ts`'s WP1b added 52 `kind: "task-intervention"`
 * entries across 4 non-default arena tasks that WP2's own predeclared
 * analyses and H1-H3 were never scoped around -- `./shared.ts#loadArchive`'s
 * own doc comment explains why; every one of that WP2 loader's own five
 * analysis CLIs applies the identical scope). `coverage.perTaskIncluded` is
 * therefore always `false` for this artifact version, and disclosed as such
 * in both the artifact and the report, rather than silently omitting the
 * field.
 *
 * Two fields extend the plan's own worked artifact shape beyond
 * `03-artifact-and-findings.md`'s literal field list (the same kind of
 * extension `selection-report.ts`'s own top doc comment documents for
 * itself): `ablation[].rank`/`saliencyScore`/`n` (beyond the plan's bare
 * `{input, effect, ci, inputMean, inputStd}`) carry through which half
 * (top/bottom-8) an ablated input came from and its own saliency score,
 * both needed to render the report's ablation tables honestly; `linkage`
 * carries both `rhoThrust` and `rhoYaw` (the plan's field list names only
 * `rho`) because `linkage.py` computes both axes and H1's own rule is
 * thrust-specific (`hypotheses.ts`'s `LinkageEntry.rhoThrust`) -- dropping
 * `rhoYaw` would discard real published data for no reason.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

export const DEFAULT_SALIENCY_PATH = resolve(repoRoot, 'training/runs/attribution/saliency.json');
export const DEFAULT_INDEPENDENCE_PATH = resolve(repoRoot, 'training/runs/attribution/independence.json');
export const DEFAULT_LINKAGE_PATH = resolve(repoRoot, 'training/runs/attribution/linkage.json');
export const DEFAULT_REGIME_PATH = resolve(repoRoot, 'training/runs/attribution/regime.json');
export const DEFAULT_ABLATION_PATH = resolve(repoRoot, 'training/runs/attribution/ablation.json');
export const DEFAULT_HYPOTHESES_PATH = resolve(repoRoot, 'training/runs/attribution/hypotheses.json');
export const DEFAULT_DESCENDING_TYPES_PATH = resolve(repoRoot, 'public/data/descending-types-v1.json');
export const DEFAULT_TRAINED_READOUT_ARTIFACT_PATH = resolve(repoRoot, 'public/data/trained-readout-v1.json');
export const DEFAULT_TRAINED_READOUT_MANIFEST_PATH = resolve(repoRoot, 'public/data/trained-readout-v1.manifest.json');
export const DEFAULT_PATHWAY_INTERVENTIONS_PATH = resolve(repoRoot, 'public/data/pathway-interventions-v1.json');
export const DEFAULT_OUT = resolve(repoRoot, 'public/data/readout-attribution-v1.json');
export const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/readout-attribution-report.md');

// ---------------------------------------------------------------------------
// Producer code identity
// ---------------------------------------------------------------------------

export interface ReadoutAttributionProducer {
  readonly script: string;
  readonly sourceSha256: string;
  readonly dependencies: readonly string[];
}

/** Same real-import-graph code-identity scheme as `selection-report.ts`'s `selectionRobustnessProducer`/`task-generality-report.ts`'s producer -- `scripts/lib/import-graph.ts`'s `collectRepoRelativeDependencies`, not a hand-maintained filename list. Its dependency closure includes `./attribution-report-markdown.ts` and `./archive-readouts.ts` (the `ArchivedReadout`/`loadArchive` imports), since both are statically imported from this file. */
export const readoutAttributionProducer = (): ReadoutAttributionProducer => {
  const dependencies = collectRepoRelativeDependencies(fileURLToPath(import.meta.url), repoRoot);
  return {
    script: 'scripts/attribution/attribution-report.ts',
    sourceSha256: computeSourceIdentitySha256(repoRoot, dependencies),
    dependencies
  };
};

// ---------------------------------------------------------------------------
// Artifact types
// ---------------------------------------------------------------------------

export interface ReadoutAttributionSaliency {
  readonly thrust: readonly number[];
  readonly yaw: readonly number[];
}

export interface ReadoutAttributionAblationEntry {
  readonly input: number;
  readonly rank: 'top' | 'bottom';
  readonly saliencyScore: number;
  readonly inputMean: number;
  readonly inputStd: number;
  readonly n: number;
  readonly effect: number;
  readonly ci: readonly [number, number];
}

export interface ReadoutAttributionIndependence {
  readonly trainedMean: number;
  readonly silencedMean: number;
  readonly defined: boolean;
  readonly ratio: number | null;
}

export interface ReadoutAttributionLinkage {
  readonly degenerate: boolean;
  readonly rhoThrust: number | null;
  readonly rhoYaw: number | null;
  readonly ciCluster: readonly [number, number] | null;
  readonly ciNeuron: readonly [number, number] | null;
  readonly clusterCount: number;
  readonly clusterSizes: readonly number[];
}

export interface ReadoutAttributionRegime {
  readonly clampFraction: number;
  readonly steadyStateDistance: number;
  readonly valid: boolean;
}

export interface ReadoutAttributionEntry {
  readonly id: string;
  readonly arm: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly arenaTask: string;
  readonly saliency: ReadoutAttributionSaliency;
  readonly saliencyVarWeighted: ReadoutAttributionSaliency;
  readonly saliencyInputMean: readonly number[];
  readonly saliencyInputStd: readonly number[];
  readonly ablation: readonly ReadoutAttributionAblationEntry[];
  readonly independence: ReadoutAttributionIndependence;
  readonly linkage: ReadoutAttributionLinkage;
  readonly regime: ReadoutAttributionRegime;
}

export type HypothesisOutcome = 'supported' | 'not-supported' | 'inconclusive';

export interface HypothesisResult {
  readonly outcome: HypothesisOutcome;
  readonly reason?: string;
  readonly evidence: unknown;
}

export interface ReadoutAttributionHypotheses {
  readonly hypothesisCount: number;
  readonly multipleComparisonCorrection: string;
  readonly H1: HypothesisResult;
  readonly H2: HypothesisResult;
  readonly H3: HypothesisResult;
}

export interface ReadoutAttributionCoverage {
  readonly ids: readonly string[];
  /** Always `false` for this artifact version -- see this file's top doc comment. */
  readonly perTaskIncluded: boolean;
}

export interface ReadoutAttributionArtifact {
  readonly version: 1;
  readonly sources: {
    readonly archiveSha: string;
    readonly descendingTypesSha: string;
    readonly trainedReadoutSha: string;
    readonly pathwayInterventionsSha: string;
    readonly producer: ReadoutAttributionProducer;
  };
  readonly coverage: ReadoutAttributionCoverage;
  readonly readouts: readonly ReadoutAttributionEntry[];
  readonly hypotheses: ReadoutAttributionHypotheses;
  readonly host: { readonly arch: string; readonly node: string };
}

// ---------------------------------------------------------------------------
// WP2 output reading + joining
// ---------------------------------------------------------------------------

const byId = <T extends { readonly id: string }>(entries: readonly T[]): Map<string, T> =>
  new Map(entries.map((entry) => [entry.id, entry] as const));

const readJsonWithArchiveCheck = <T>(label: string, path: string, expectedArchiveSha256: string): T => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as T;
  const archiveSha256 = (parsed as { readonly archiveSha256?: string }).archiveSha256;
  assertSameArchive('attribution-report', `${label} (${path})`, archiveSha256, expectedArchiveSha256);
  return parsed;
};

const requireEntry = <T>(map: ReadonlyMap<string, T>, id: string, label: string): T => {
  const entry = map.get(id);
  if (!entry) throw new Error(`attribution-report: ${label} has no entry for archived readout "${id}"`);
  return entry;
};

interface SaliencyJsonEntry {
  readonly id: string;
  readonly thrust: readonly number[];
  readonly yaw: readonly number[];
  readonly thrustVarWeighted: readonly number[];
  readonly yawVarWeighted: readonly number[];
  readonly inputMean: readonly number[];
  readonly inputStd: readonly number[];
}
interface IndependenceJsonEntry {
  readonly id: string;
  readonly trainedMean: number;
  readonly silencedMean: number;
  readonly defined: boolean;
  readonly ratio: number | null;
}
interface LinkageJsonEntry {
  readonly id: string;
  readonly degenerate?: boolean;
  readonly rhoThrust: number | null;
  readonly rhoYaw: number | null;
  readonly ciCluster: readonly [number, number] | null;
  readonly ciNeuron: readonly [number, number] | null;
  readonly clusterCount: number;
  readonly clusterSizes: readonly number[];
}
interface RegimeJsonEntry {
  readonly id: string;
  readonly clampFraction: number;
  readonly steadyStateDistance: number;
  readonly valid: boolean;
}
interface AblationJsonEntry {
  readonly id: string;
  readonly n: number;
  readonly baselineMean: number;
  readonly ablations: readonly {
    readonly input: number;
    readonly rank: 'top' | 'bottom';
    readonly saliencyScore: number;
    readonly inputMean: number;
    readonly inputStd: number;
    readonly effect: { readonly n: number; readonly meanDifference: number; readonly ci95: readonly [number, number] };
  }[];
}

export interface BuildReadoutAttributionArtifactInputs {
  readonly archivePath: string;
  readonly saliencyPath: string;
  readonly independencePath: string;
  readonly linkagePath: string;
  readonly regimePath: string;
  readonly ablationPath: string;
  readonly hypothesesPath: string;
  readonly descendingTypesPath: string;
  readonly trainedReadoutArtifactPath: string;
  readonly trainedReadoutManifestPath: string;
  readonly pathwayInterventionsPath: string;
  readonly manifestPath: string;
}

export const buildReadoutAttributionArtifact = (
  inputs: Readonly<BuildReadoutAttributionArtifactInputs>
): ReadoutAttributionArtifact => {
  const readouts: readonly ArchivedReadout[] = loadArchive(inputs.archivePath);
  const archiveSha = computeArchiveSha256(inputs.archivePath);

  const saliency = readJsonWithArchiveCheck<{ readonly entries: readonly SaliencyJsonEntry[] }>(
    'saliency.json',
    inputs.saliencyPath,
    archiveSha
  );
  const independence = readJsonWithArchiveCheck<{ readonly entries: readonly IndependenceJsonEntry[] }>(
    'independence.json',
    inputs.independencePath,
    archiveSha
  );
  const linkage = readJsonWithArchiveCheck<{ readonly readouts: readonly LinkageJsonEntry[] }>(
    'linkage.json',
    inputs.linkagePath,
    archiveSha
  );
  const regime = readJsonWithArchiveCheck<{ readonly entries: readonly RegimeJsonEntry[] }>(
    'regime.json',
    inputs.regimePath,
    archiveSha
  );
  const ablation = readJsonWithArchiveCheck<{ readonly entries: readonly AblationJsonEntry[] }>(
    'ablation.json',
    inputs.ablationPath,
    archiveSha
  );
  const hypothesesRaw = readJsonWithArchiveCheck<{
    readonly hypothesisCount: number;
    readonly multipleComparisonCorrection: string;
    readonly H1: HypothesisResult;
    readonly H2: HypothesisResult;
    readonly H3: HypothesisResult;
  }>('hypotheses.json', inputs.hypothesesPath, archiveSha);

  const saliencyById = byId(saliency.entries);
  const independenceById = byId(independence.entries);
  const linkageById = byId(linkage.readouts);
  const regimeById = byId(regime.entries);
  const ablationById = byId(ablation.entries);

  const readoutEntries: ReadoutAttributionEntry[] = readouts.map((entry) => {
    const s = requireEntry(saliencyById, entry.id, 'saliency.json');
    const ind = requireEntry(independenceById, entry.id, 'independence.json');
    const link = requireEntry(linkageById, entry.id, 'linkage.json');
    const reg = requireEntry(regimeById, entry.id, 'regime.json');
    const abl = requireEntry(ablationById, entry.id, 'ablation.json');
    return {
      id: entry.id,
      arm: entry.arm,
      graphId: entry.graphId,
      trainerSeed: entry.trainerSeed,
      arenaTask: entry.arenaTask,
      saliency: { thrust: s.thrust, yaw: s.yaw },
      saliencyVarWeighted: { thrust: s.thrustVarWeighted, yaw: s.yawVarWeighted },
      saliencyInputMean: s.inputMean,
      saliencyInputStd: s.inputStd,
      ablation: abl.ablations.map((a) => ({
        input: a.input,
        rank: a.rank,
        saliencyScore: a.saliencyScore,
        inputMean: a.inputMean,
        inputStd: a.inputStd,
        n: a.effect.n,
        effect: a.effect.meanDifference,
        ci: a.effect.ci95
      })),
      independence: { trainedMean: ind.trainedMean, silencedMean: ind.silencedMean, defined: ind.defined, ratio: ind.ratio },
      linkage: {
        degenerate: Boolean(link.degenerate),
        rhoThrust: link.rhoThrust,
        rhoYaw: link.rhoYaw,
        ciCluster: link.ciCluster,
        ciNeuron: link.ciNeuron,
        clusterCount: link.clusterCount,
        clusterSizes: link.clusterSizes
      },
      regime: { clampFraction: reg.clampFraction, steadyStateDistance: reg.steadyStateDistance, valid: reg.valid }
    };
  });

  const hypotheses: ReadoutAttributionHypotheses = {
    hypothesisCount: hypothesesRaw.hypothesisCount,
    multipleComparisonCorrection: hypothesesRaw.multipleComparisonCorrection,
    H1: hypothesesRaw.H1,
    H2: hypothesesRaw.H2,
    H3: hypothesesRaw.H3
  };

  // Descending-types sidecar: recomputed from the actual bytes, cross-checked
  // against the arena manifest's own `descendingTypes.sha256` -- the same
  // "recompute, don't trust another producer's own claimed hash" discipline
  // every other sidecar cross-check in this codebase already applies.
  const descendingTypesBytes = readFileSync(inputs.descendingTypesPath);
  const descendingTypesSha = sha256Hex(descendingTypesBytes);
  const arenaManifest = JSON.parse(readFileSync(inputs.manifestPath, 'utf8')) as {
    readonly binarySha256?: string;
    readonly descendingTypes?: { readonly artifact: string; readonly sha256: string };
    readonly pathwayInterventions?: { readonly artifact: string; readonly sha256: string };
  };
  if (!arenaManifest.descendingTypes) {
    throw new Error(`attribution-report: ${inputs.manifestPath} is missing "descendingTypes"`);
  }
  if (descendingTypesSha !== arenaManifest.descendingTypes.sha256) {
    throw new Error(
      `attribution-report: descending-types-v1.json sha256 ${descendingTypesSha} does not match the manifest's ` +
        `descendingTypes.sha256 (${arenaManifest.descendingTypes.sha256}) -- stale artifact or manifest`
    );
  }

  // trained-readout-v1.json: recomputed from bytes, cross-checked against
  // its own manifest's `artifactSha256` (`trained-readout-v1.manifest.json`,
  // a separate file from the arena manifest -- see `assets.ts#loadTrainedReadoutArtifact`).
  const trainedReadoutBytes = readFileSync(inputs.trainedReadoutArtifactPath);
  const trainedReadoutSha = sha256Hex(trainedReadoutBytes);
  const trainedReadoutManifest = JSON.parse(readFileSync(inputs.trainedReadoutManifestPath, 'utf8')) as {
    readonly artifactSha256: string;
  };
  if (trainedReadoutSha !== trainedReadoutManifest.artifactSha256) {
    throw new Error(
      `attribution-report: trained-readout-v1.json sha256 ${trainedReadoutSha} does not match ` +
        `trained-readout-v1.manifest.json's artifactSha256 (${trainedReadoutManifest.artifactSha256})`
    );
  }

  // pathway-interventions-v1.json: recomputed from bytes, cross-checked
  // against the arena manifest's `pathwayInterventions.sha256`.
  const pathwayInterventionsBytes = readFileSync(inputs.pathwayInterventionsPath);
  const pathwayInterventionsSha = sha256Hex(pathwayInterventionsBytes);
  if (!arenaManifest.pathwayInterventions) {
    throw new Error(`attribution-report: ${inputs.manifestPath} is missing "pathwayInterventions"`);
  }
  if (pathwayInterventionsSha !== arenaManifest.pathwayInterventions.sha256) {
    throw new Error(
      `attribution-report: pathway-interventions-v1.json sha256 ${pathwayInterventionsSha} does not match the ` +
        `manifest's pathwayInterventions.sha256 (${arenaManifest.pathwayInterventions.sha256})`
    );
  }

  return {
    version: 1,
    sources: {
      archiveSha,
      descendingTypesSha,
      trainedReadoutSha,
      pathwayInterventionsSha,
      producer: readoutAttributionProducer()
    },
    coverage: { ids: readouts.map((entry) => entry.id), perTaskIncluded: false },
    readouts: readoutEntries,
    hypotheses,
    host: { arch: process.arch, node: process.version }
  };
};

// ---------------------------------------------------------------------------
// Descending cell-type naming (for the report only -- never shipped in the JSON artifact)
// ---------------------------------------------------------------------------

export interface DescendingTypeEntry {
  readonly index: number;
  readonly bodyId: string;
  readonly type: string | null;
  readonly class: string | null;
  readonly instance: string | null;
  readonly somaSide: string | null;
}

export const loadDescendingTypeNames = (path: string): readonly DescendingTypeEntry[] => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as { readonly neurons: readonly DescendingTypeEntry[] };
  return [...parsed.neurons].sort((a, b) => a.index - b.index);
};

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Add/overwrite the manifest's `readoutAttribution` key in place -- mirrors `selection-report.ts`'s `updateManifestWithSelectionRobustness`. */
export const updateManifestWithReadoutAttribution = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.readoutAttribution = entry;
  atomicWriteFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface AttributionReportArgs {
  readonly archivePath: string;
  readonly saliencyPath: string;
  readonly independencePath: string;
  readonly linkagePath: string;
  readonly regimePath: string;
  readonly ablationPath: string;
  readonly hypothesesPath: string;
  readonly descendingTypesPath: string;
  readonly trainedReadoutArtifactPath: string;
  readonly trainedReadoutManifestPath: string;
  readonly pathwayInterventionsPath: string;
  readonly manifestPath: string;
  readonly out: string;
  readonly reportMd: string;
  readonly skipManifestUpdate: boolean;
}

export const parseAttributionReportArgs = (argv: readonly string[]): AttributionReportArgs => {
  let archivePath = DEFAULT_ARCHIVE_PATH;
  let saliencyPath = DEFAULT_SALIENCY_PATH;
  let independencePath = DEFAULT_INDEPENDENCE_PATH;
  let linkagePath = DEFAULT_LINKAGE_PATH;
  let regimePath = DEFAULT_REGIME_PATH;
  let ablationPath = DEFAULT_ABLATION_PATH;
  let hypothesesPath = DEFAULT_HYPOTHESES_PATH;
  let descendingTypesPath = DEFAULT_DESCENDING_TYPES_PATH;
  let trainedReadoutArtifactPath = DEFAULT_TRAINED_READOUT_ARTIFACT_PATH;
  let trainedReadoutManifestPath = DEFAULT_TRAINED_READOUT_MANIFEST_PATH;
  let pathwayInterventionsPath = DEFAULT_PATHWAY_INTERVENTIONS_PATH;
  let manifestPath = DEFAULT_MANIFEST_PATH;
  let out = DEFAULT_OUT;
  let reportMd = DEFAULT_REPORT_MD;
  let skipManifestUpdate = false;

  let i = 0;
  const resolveArg = (value: string): string => resolve(process.cwd(), requireValue(argv[i], value));
  const handlers: Record<string, (value: string) => void> = {
    '--archive': (v) => (archivePath = resolveArg(v)),
    '--saliency': (v) => (saliencyPath = resolveArg(v)),
    '--independence': (v) => (independencePath = resolveArg(v)),
    '--linkage': (v) => (linkagePath = resolveArg(v)),
    '--regime': (v) => (regimePath = resolveArg(v)),
    '--ablation': (v) => (ablationPath = resolveArg(v)),
    '--hypotheses': (v) => (hypothesesPath = resolveArg(v)),
    '--descending-types': (v) => (descendingTypesPath = resolveArg(v)),
    '--trained-readout-artifact': (v) => (trainedReadoutArtifactPath = resolveArg(v)),
    '--trained-readout-manifest': (v) => (trainedReadoutManifestPath = resolveArg(v)),
    '--pathway-interventions': (v) => (pathwayInterventionsPath = resolveArg(v)),
    '--manifest': (v) => (manifestPath = resolveArg(v)),
    '--out': (v) => (out = resolveArg(v)),
    '--report-md': (v) => (reportMd = resolveArg(v))
  };
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--skip-manifest-update') {
      skipManifestUpdate = true;
      i += 1;
      continue;
    }
    const handler = handlers[flag];
    if (!handler) throw new Error(`Unknown argument: ${flag}`);
    handler(argv[i + 1]);
    i += 2;
  }

  if (resolve(out) === resolve(reportMd)) {
    throw new Error('attribution-report: --out and --report-md must not be the same path');
  }
  return {
    archivePath,
    saliencyPath,
    independencePath,
    linkagePath,
    regimePath,
    ablationPath,
    hypothesesPath,
    descendingTypesPath,
    trainedReadoutArtifactPath,
    trainedReadoutManifestPath,
    pathwayInterventionsPath,
    manifestPath,
    out,
    reportMd,
    skipManifestUpdate
  };
};

export interface RunAttributionReportResult {
  readonly out: string;
  readonly reportMdPath: string;
  readonly artifactSha256: string;
  readonly artifact: ReadoutAttributionArtifact;
}

export const runAttributionReport = (args: Readonly<AttributionReportArgs>): RunAttributionReportResult => {
  if (!args.skipManifestUpdate) verifyManifestRoundTrips(args.manifestPath);

  const artifact = buildReadoutAttributionArtifact(args);
  const artifactContents = JSON.stringify(artifact);
  const artifactSha256 = sha256Hex(artifactContents);

  const descendingTypes = loadDescendingTypeNames(args.descendingTypesPath);
  const reportMdContents = renderReadoutAttributionReportMarkdown(artifact, descendingTypes);

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, artifactContents);

  if (!args.skipManifestUpdate) {
    updateManifestWithReadoutAttribution(args.manifestPath, { artifact: basename(args.out), sha256: artifactSha256 });
  }

  mkdirSync(dirname(args.reportMd), { recursive: true });
  atomicWriteFileSync(args.reportMd, reportMdContents);

  return { out: args.out, reportMdPath: args.reportMd, artifactSha256, artifact };
};

const main = (): void => {
  try {
    const args = parseAttributionReportArgs(process.argv.slice(2));
    const result = runAttributionReport(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `attribution-report: wrote ${result.out} (sha256 ${result.artifactSha256}) and ${result.reportMdPath}\n` +
        `H1=${result.artifact.hypotheses.H1.outcome} H2=${result.artifact.hypotheses.H2.outcome} H3=${result.artifact.hypotheses.H3.outcome}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`attribution-report failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
