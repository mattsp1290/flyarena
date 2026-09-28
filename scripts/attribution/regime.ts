import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import type { ReadoutWeights } from '../../src/lib/connectome/readout';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { runEpisode } from '../training/episode';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { createRegimeAccumulator } from '../null/regime-metrics';
import {
  DEFAULT_ARCHIVE_PATH,
  defaultResolveGraphConfig,
  graphForEntry,
  loadArchive,
  parsePathFlags,
  repoRoot,
  resolvePathFlag,
  SALIENCY_SEEDS,
  SALIENCY_TICKS,
  weightsForEntry
} from './shared';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 regime gate:
 * runs trained-readout episodes with the per-substep `onSubstep`
 * (`episode.ts`'s WP2 addition -- now valid for `trained`, see that file's
 * doc comment) and `scripts/null/regime-metrics.ts`'s accumulator -- "the
 * same statistic at the same granularity as the explanation study, so the
 * 20% and 0.5 thresholds apply unchanged" (`CLAMP_FRACTION_THRESHOLD`/
 * `STEADY_STATE_DISTANCE_THRESHOLD`, `scripts/analysis/explain.py`).
 *
 * H1 (`00-overview.md`) is evaluated only when a readout's trajectories
 * pass this gate; `hypotheses.ts` (not this file) applies that rule. This
 * module only measures and reports `clampFraction`/`steadyStateDistance`/
 * `valid` per readout, averaged over the same 10 held-out seeds
 * (`SALIENCY_SEEDS`, 30001-30010) `saliency.ts` collects its trajectories
 * from -- a separate simulation pass (not a reuse of `saliency.ts`'s own
 * `onReadoutInput` recordings), since regime metrics need per-SUBSTEP
 * `(rate, channelValues)`, a different granularity `onReadoutInput` never
 * observes.
 *
 * The per-graph steady-state map this needs (`M = (lambda I - g A)^-1 B`)
 * is NOT recomputed here -- it is read from `linkage.py`'s own output
 * (`--linkage <linkage.json>`), which already computes and sha-verifies one
 * sidecar per unique graph in the archive (this file's own dense O(n^3)
 * solve would otherwise duplicate that expensive step). `linkage.py` must
 * therefore run before this script.
 */

const CLAMP_FRACTION_THRESHOLD = 0.2; // `scripts/analysis/explain.py`'s `CLAMP_FRACTION_THRESHOLD`.
const STEADY_STATE_DISTANCE_THRESHOLD = 0.5; // `scripts/analysis/explain.py`'s `STEADY_STATE_DISTANCE_THRESHOLD`.

export interface RegimeResult {
  readonly clampFraction: number;
  readonly steadyStateDistance: number;
  readonly valid: boolean;
}

interface LinkageGraphManifestEntry {
  readonly sidecarPath: string;
  readonly sidecarSha256: string;
  readonly neuronCount: number;
  readonly inputChannelCount: number;
}

interface LinkageManifest {
  readonly graphs: Readonly<Record<string, LinkageGraphManifestEntry>>;
}

const loadSteadyStateMap = (path: string, expectedSha256: string): Float64Array => {
  const buffer = readFileSync(path);
  const actualSha256 = sha256Hex(buffer);
  if (actualSha256 !== expectedSha256) {
    throw new Error(`regime: ${path} sha256 ${actualSha256} does not match linkage.py's manifest entry ${expectedSha256}`);
  }
  const aligned = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  if (aligned.byteLength % 8 !== 0) {
    throw new Error(`regime: ${path} is not a whole number of float64 values (${aligned.byteLength} bytes)`);
  }
  return new Float64Array(aligned);
};

/** Average `clampFraction`/`steadyStateDistance` over `seeds`, one fresh accumulator per episode. */
export const runRegimeForReadout = (
  graph: Readonly<ConnectomeGraph>,
  weights: Readonly<ReadoutWeights>,
  steadyStateMap: Float64Array,
  seeds: readonly number[],
  ticks: number,
  arenaTask?: string
): RegimeResult => {
  const { neuronCount, inputChannelCount, rateMin, rateMax, inputClampMin, inputClampMax } = graph.metadata;
  if (!(rateMin < 0 && 0 < rateMax)) {
    throw new Error(
      `regime: graph has rateMin=${rateMin}, rateMax=${rateMax} -- the clamp-fraction metric assumes ` +
        'rateMin < 0 < rateMax'
    );
  }
  const expectedLength = neuronCount * inputChannelCount;
  if (steadyStateMap.length !== expectedLength) {
    throw new Error(
      `regime: steady-state map has ${steadyStateMap.length} values, expected ${expectedLength} ` +
        `(neuronCount ${neuronCount} x inputChannelCount ${inputChannelCount})`
    );
  }

  let clampFractionSum = 0;
  let steadyStateDistanceSum = 0;
  for (const seed of seeds) {
    const accumulator = createRegimeAccumulator({
      neuronCount,
      inputChannelCount,
      rateMin,
      rateMax,
      inputClampMin,
      inputClampMax,
      steadyStateMap,
      substeps: NEURAL_SUBSTEPS_PER_TICK
    });
    runEpisode({
      seed,
      ticks,
      arenaTask,
      substeps: NEURAL_SUBSTEPS_PER_TICK,
      left: { decoder: 'trained', graph, weights, onSubstep: accumulator.onSubstep },
      right: { decoder: 'parked' }
    });
    const { clampFraction, steadyStateDistance } = accumulator.result(`seed ${seed}`);
    clampFractionSum += clampFraction;
    steadyStateDistanceSum += steadyStateDistance;
  }
  const clampFraction = clampFractionSum / seeds.length;
  const steadyStateDistance = steadyStateDistanceSum / seeds.length;
  const valid = clampFraction <= CLAMP_FRACTION_THRESHOLD && steadyStateDistance <= STEADY_STATE_DISTANCE_THRESHOLD;
  return { clampFraction, steadyStateDistance, valid };
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

interface RegimeOutputEntry extends RegimeResult {
  readonly id: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly arenaTask: string;
  readonly seeds: readonly number[];
  readonly ticks: number;
}

interface RegimeArgs {
  readonly archivePath: string;
  readonly linkagePath: string;
  readonly manifestPath?: string;
  readonly interventionIndexPath?: string;
  readonly archivedInterventionIndexPath?: string;
  readonly out: string;
}

const parseArgs = (argv: readonly string[]): RegimeArgs => {
  let archivePath = DEFAULT_ARCHIVE_PATH;
  let linkagePath = resolve(repoRoot, 'training/runs/attribution/linkage.json');
  let manifestPath: string | undefined;
  let interventionIndexPath: string | undefined;
  let archivedInterventionIndexPath: string | undefined;
  let out = resolve(process.cwd(), 'training/runs/attribution/regime.json');
  parsePathFlags('regime', argv, {
    '--archive': (v) => (archivePath = resolvePathFlag(v)),
    '--linkage': (v) => (linkagePath = resolvePathFlag(v)),
    '--manifest': (v) => (manifestPath = resolvePathFlag(v)),
    '--intervention-index': (v) => (interventionIndexPath = resolvePathFlag(v)),
    '--archived-intervention-index': (v) => (archivedInterventionIndexPath = resolvePathFlag(v)),
    '--out': (v) => (out = resolvePathFlag(v))
  });
  return { archivePath, linkagePath, manifestPath, interventionIndexPath, archivedInterventionIndexPath, out };
};

export const runRegime = (args: Readonly<RegimeArgs>): { readonly out: string; readonly count: number; readonly sha256: string } => {
  const readouts = loadArchive(args.archivePath);
  const resolveConfig = defaultResolveGraphConfig(args);
  const linkage = JSON.parse(readFileSync(args.linkagePath, 'utf8')) as LinkageManifest;
  const linkageDir = repoRoot;

  const steadyStateByGraph = new Map<string, Float64Array>();
  const steadyStateFor = (graphId: string): Float64Array => {
    const cached = steadyStateByGraph.get(graphId);
    if (cached) return cached;
    const manifestEntry = linkage.graphs[graphId];
    if (!manifestEntry) throw new Error(`regime: linkage.json has no steady-state entry for graphId "${graphId}"`);
    const map = loadSteadyStateMap(resolve(linkageDir, manifestEntry.sidecarPath), manifestEntry.sidecarSha256);
    steadyStateByGraph.set(graphId, map);
    return map;
  };

  const entries: RegimeOutputEntry[] = [];
  for (const entry of readouts) {
    const weights = weightsForEntry(entry);
    const graph = graphForEntry(entry, resolveConfig);
    const steadyStateMap = steadyStateFor(entry.graphId);
    const result = runRegimeForReadout(graph, weights, steadyStateMap, SALIENCY_SEEDS, SALIENCY_TICKS, entry.arenaTask);
    entries.push({
      id: entry.id,
      graphId: entry.graphId,
      trainerSeed: entry.trainerSeed,
      arenaTask: entry.arenaTask,
      seeds: SALIENCY_SEEDS,
      ticks: SALIENCY_TICKS,
      ...result
    });
  }
  entries.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const body = JSON.stringify({
    version: 1,
    clampFractionThreshold: CLAMP_FRACTION_THRESHOLD,
    steadyStateDistanceThreshold: STEADY_STATE_DISTANCE_THRESHOLD,
    entries
  });
  mkdirSync(resolve(args.out, '..'), { recursive: true });
  atomicWriteFileSync(args.out, body);
  return { out: args.out, count: entries.length, sha256: sha256Hex(body) };
};

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));
  const result = runRegime(args);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing summary output.
  console.log(`regime: wrote ${result.out} (${result.count} readouts, sha256 ${result.sha256})`);
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
