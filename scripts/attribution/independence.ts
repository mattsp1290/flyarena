import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import type { ReadoutWeights } from '../../src/lib/connectome/readout';
import { runEpisode } from '../training/episode';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { mean } from '../training/stats';
import {
  computeArchiveSha256,
  DEFAULT_ARCHIVE_PATH,
  defaultResolveGraphConfig,
  graphForEntry,
  loadArchive,
  parseFlags,
  resolvePathFlag,
  SCORING_SEEDS,
  SCORING_TICKS,
  weightsForEntry
} from './shared';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 analysis 3:
 * input-independence share (`00-overview.md`'s predeclared analysis 3):
 * per readout, `mean(silenced) / mean(trained)` over the 100 held-out
 * seeds -- **a ratio of means, never per-episode** (per-episode ratios
 * would be dominated by episodes with a near-zero trained score). `score`
 * is `movementScore` (`AgentScoreResult`, `scripts/training/episode.ts`),
 * matching every other score comparison in this repository
 * (`evaluate.ts`'s `scoreCache`, `null-worker.ts`'s `NullSeedResult`, ...).
 *
 * When `mean(trained) <= 1`, the share is reported as undefined
 * (`defined: false`, `ratio: null`) rather than a division producing a
 * huge or negative-looking number from an already-near-zero denominator --
 * "It measures how much of the policy ignores the circuit", which is not a
 * meaningful question when the trained policy itself barely scores.
 */

export interface IndependenceResult {
  readonly trainedMean: number;
  readonly silencedMean: number;
  readonly defined: boolean;
  readonly ratio: number | null;
}

export const computeIndependence = (
  trainedScores: readonly number[],
  silencedScores: readonly number[]
): IndependenceResult => {
  if (trainedScores.length === 0 || silencedScores.length === 0) {
    throw new Error('independence: trainedScores/silencedScores must be non-empty');
  }
  const trainedMean = mean(trainedScores);
  const silencedMean = mean(silencedScores);
  const defined = trainedMean > 1;
  return { trainedMean, silencedMean, defined, ratio: defined ? silencedMean / trainedMean : null };
};

const scoreEpisode = (
  graph: Readonly<ConnectomeGraph>,
  weights: Readonly<ReadoutWeights>,
  decoder: 'trained' | 'silenced',
  seed: number,
  ticks: number,
  arenaTask?: string
): number =>
  runEpisode({
    seed,
    ticks,
    arenaTask,
    left: { decoder, graph, weights },
    right: { decoder: 'parked' }
  }).left.movementScore;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

interface IndependenceOutputEntry extends IndependenceResult {
  readonly id: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly arenaTask: string;
  readonly n: number;
}

interface IndependenceArgs {
  readonly archivePath: string;
  readonly manifestPath?: string;
  readonly interventionIndexPath?: string;
  readonly archivedInterventionIndexPath?: string;
  readonly out: string;
}

const parseArgs = (argv: readonly string[]): IndependenceArgs => {
  let archivePath = DEFAULT_ARCHIVE_PATH;
  let manifestPath: string | undefined;
  let interventionIndexPath: string | undefined;
  let archivedInterventionIndexPath: string | undefined;
  let out = resolve(process.cwd(), 'training/runs/attribution/independence.json');
  parseFlags('independence', argv, {
    '--archive': (v) => (archivePath = resolvePathFlag(v)),
    '--manifest': (v) => (manifestPath = resolvePathFlag(v)),
    '--intervention-index': (v) => (interventionIndexPath = resolvePathFlag(v)),
    '--archived-intervention-index': (v) => (archivedInterventionIndexPath = resolvePathFlag(v)),
    '--out': (v) => (out = resolvePathFlag(v))
  });
  return { archivePath, manifestPath, interventionIndexPath, archivedInterventionIndexPath, out };
};

export const runIndependence = (
  args: Readonly<IndependenceArgs>
): { readonly out: string; readonly count: number; readonly sha256: string } => {
  const readouts = loadArchive(args.archivePath);
  const resolveConfig = defaultResolveGraphConfig(args);
  const entries: IndependenceOutputEntry[] = [];
  for (const entry of readouts) {
    const weights = weightsForEntry(entry);
    const graph = graphForEntry(entry, resolveConfig);
    const trainedScores = SCORING_SEEDS.map((seed) =>
      scoreEpisode(graph, weights, 'trained', seed, SCORING_TICKS, entry.arenaTask)
    );
    const silencedScores = SCORING_SEEDS.map((seed) =>
      scoreEpisode(graph, weights, 'silenced', seed, SCORING_TICKS, entry.arenaTask)
    );
    const result = computeIndependence(trainedScores, silencedScores);
    entries.push({
      id: entry.id,
      graphId: entry.graphId,
      trainerSeed: entry.trainerSeed,
      arenaTask: entry.arenaTask,
      n: SCORING_SEEDS.length,
      ...result
    });
  }
  entries.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const body = JSON.stringify({ version: 1, archiveSha256: computeArchiveSha256(args.archivePath), entries });
  mkdirSync(resolve(args.out, '..'), { recursive: true });
  atomicWriteFileSync(args.out, body);
  return { out: args.out, count: entries.length, sha256: sha256Hex(body) };
};

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));
  const result = runIndependence(args);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing summary output.
  console.log(`independence: wrote ${result.out} (${result.count} readouts, sha256 ${result.sha256})`);
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
