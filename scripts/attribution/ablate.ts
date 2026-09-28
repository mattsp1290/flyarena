import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runShardedEvaluation } from '../null/sharded-evaluation';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { conditionRng, pairedStats, type PairedStats } from '../training/stats';
import type { AblateSeedResult, AblateWorkerMessage, AblateWorkerTask } from './ablate-task';
import {
  DEFAULT_ARCHIVE_PATH,
  DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH,
  DEFAULT_MANIFEST_PATH,
  loadArchive,
  parsePathFlags,
  resolvePathFlag,
  SCORING_SEEDS,
  SCORING_TICKS,
  weightsForEntry
} from './shared';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 analysis 2:
 * readout-input ablation. Per readout, ablates the top-8 and bottom-8
 * descending inputs by saliency (thrust + yaw saliency summed, from
 * `saliency.ts`'s own output) one at a time via `episode.ts`'s
 * `readoutMask`, on the 100-seed `SCORING_SEEDS`, and reports each one's
 * paired score difference against the SAME readout's unmasked baseline
 * (`pairedStats`, `scripts/training/stats.ts`) with a fixed bootstrap seed
 * (`--bootstrap-seed`, a CLI constant for the whole run -- `conditionRng`
 * still derives an independent stream per `(readout, input)` label from it,
 * matching this codebase's established "per-label-independent bootstrap"
 * convention, never one shared sequentially-consumed stream).
 *
 * Sharded like `scripts/null/null-evaluate.ts`
 * (`runShardedEvaluation`/`ablate-worker.ts`): one task per `(readout,
 * ablated-input-or-baseline)` pair, all queued together so idle shards pick
 * up the next task regardless of which readout it belongs to.
 */

export interface AblationEntry {
  readonly input: number;
  readonly rank: 'top' | 'bottom';
  readonly saliencyScore: number;
  readonly inputMean: number;
  readonly inputStd: number;
  readonly effect: PairedStats;
}

/** Rank inputs by `thrust[d] + yaw[d]` descending; ties broken by ascending `d` for determinism. Returns the top-8 and bottom-8 input indices (16 distinct inputs whenever `D >= 16`), each sorted ascending. */
export const rankAblationInputs = (
  thrust: readonly number[],
  yaw: readonly number[],
  count = 8
): { readonly top: readonly number[]; readonly bottom: readonly number[] } => {
  if (thrust.length !== yaw.length) throw new Error('ablate: thrust/yaw saliency arrays must have equal length');
  const D = thrust.length;
  if (count * 2 > D) throw new Error(`ablate: cannot select ${count} top + ${count} bottom inputs from only ${D}`);
  const combined = Array.from({ length: D }, (_, d) => ({ d, score: thrust[d] + yaw[d] }));
  const bySalienceDesc = [...combined].sort((a, b) => b.score - a.score || a.d - b.d);
  const top = bySalienceDesc.slice(0, count).map((e) => e.d).sort((a, b) => a - b);
  const bottom = bySalienceDesc
    .slice(D - count)
    .map((e) => e.d)
    .sort((a, b) => a - b);
  return { top, bottom };
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const here = dirname(fileURLToPath(import.meta.url));
const WORKER_PATH = resolve(here, 'ablate-worker.ts');

interface SaliencyOutputEntry {
  readonly id: string;
  readonly thrust: readonly number[];
  readonly yaw: readonly number[];
  readonly inputMean: readonly number[];
  readonly inputStd: readonly number[];
}

interface AblateOutputEntry {
  readonly id: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly arenaTask: string;
  readonly n: number;
  readonly baselineMean: number;
  readonly ablations: readonly AblationEntry[];
}

interface AblateArgs {
  readonly archivePath: string;
  readonly saliencyPath: string;
  readonly manifestPath: string;
  readonly interventionIndexPath?: string;
  readonly archivedInterventionIndexPath: string;
  readonly out: string;
  readonly shards: number;
  readonly bootstrapSeed: number;
  readonly resamples: number;
}

const parseArgs = (argv: readonly string[]): AblateArgs => {
  let archivePath = DEFAULT_ARCHIVE_PATH;
  let saliencyPath = resolve(process.cwd(), 'training/runs/attribution/saliency.json');
  let manifestPath = DEFAULT_MANIFEST_PATH;
  let interventionIndexPath: string | undefined;
  let archivedInterventionIndexPath = DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH;
  let out = resolve(process.cwd(), 'training/runs/attribution/ablation.json');
  let shards = 16;
  let bootstrapSeed = 1;
  let resamples = 2000;
  parsePathFlags('ablate', argv, {
    '--archive': (v) => (archivePath = resolvePathFlag(v)),
    '--saliency': (v) => (saliencyPath = resolvePathFlag(v)),
    '--manifest': (v) => (manifestPath = resolvePathFlag(v)),
    '--intervention-index': (v) => (interventionIndexPath = resolvePathFlag(v)),
    '--archived-intervention-index': (v) => (archivedInterventionIndexPath = resolvePathFlag(v)),
    '--out': (v) => (out = resolvePathFlag(v)),
    '--shards': (v) => (shards = Number(v)),
    '--bootstrap-seed': (v) => (bootstrapSeed = Number(v)),
    '--resamples': (v) => (resamples = Number(v))
  });
  if (!Number.isInteger(shards) || shards <= 0) throw new Error(`ablate: --shards must be a positive integer, got ${shards}`);
  if (!Number.isInteger(bootstrapSeed)) throw new Error('ablate: --bootstrap-seed must be an integer');
  if (!Number.isInteger(resamples) || resamples <= 0) throw new Error('ablate: --resamples must be a positive integer');
  return { archivePath, saliencyPath, manifestPath, interventionIndexPath, archivedInterventionIndexPath, out, shards, bootstrapSeed, resamples };
};

const baselineKey = (readoutId: string): string => `${readoutId}|baseline`;
const ablationKey = (readoutId: string, input: number): string => `${readoutId}|d${input}`;

export const runAblate = async (
  args: Readonly<AblateArgs>
): Promise<{ readonly out: string; readonly count: number; readonly sha256: string }> => {
  const readouts = loadArchive(args.archivePath);
  const saliencyById = new Map<string, SaliencyOutputEntry>(
    (JSON.parse(readFileSync(args.saliencyPath, 'utf8')) as { entries: SaliencyOutputEntry[] }).entries.map((e) => [
      e.id,
      e
    ])
  );

  const tasks: AblateWorkerTask[] = [];
  const rankByReadout = new Map<string, { readonly top: readonly number[]; readonly bottom: readonly number[] }>();

  for (const entry of readouts) {
    const saliencyEntry = saliencyById.get(entry.id);
    if (!saliencyEntry) throw new Error(`ablate: no saliency entry for readout "${entry.id}" -- run saliency.ts first`);
    const ranked = rankAblationInputs(saliencyEntry.thrust, saliencyEntry.yaw);
    rankByReadout.set(entry.id, ranked);
    const inputs = [...ranked.top, ...ranked.bottom];
    // Verifies theta's weightsSha256 before it is ever sent to a worker
    // over IPC (the worker itself does not re-check it) -- `weightsForEntry`
    // is called for this side effect; its `ReadoutWeights` return isn't
    // used, since the worker rebuilds it from the flat array below.
    weightsForEntry(entry);

    const baseTask = {
      resolvedGraphId: entry.graphId,
      graphGzipSha256: entry.graphGzipSha256,
      graphBinarySha256: entry.graphBinarySha256,
      manifestPath: args.manifestPath,
      interventionIndexPath: args.interventionIndexPath,
      archivedInterventionIndexPath: args.archivedInterventionIndexPath,
      theta: decodeThetaArray(entry.theta),
      D: entry.D,
      H: entry.H,
      heldOutSeeds: SCORING_SEEDS,
      ticks: SCORING_TICKS,
      arenaTask: entry.arenaTask
    };

    tasks.push({ ...baseTask, graphId: baselineKey(entry.id), mask: null });
    for (const input of inputs) {
      tasks.push({ ...baseTask, graphId: ablationKey(entry.id, input), mask: [input] });
    }
  }

  const results = await runShardedEvaluation<AblateWorkerTask, AblateSeedResult, AblateWorkerMessage>(
    tasks,
    args.shards,
    WORKER_PATH
  );

  const outEntries: AblateOutputEntry[] = [];
  for (const entry of readouts) {
    const saliencyEntry = saliencyById.get(entry.id);
    if (!saliencyEntry) throw new Error(`ablate: no saliency entry for readout "${entry.id}"`);
    const ranked = rankByReadout.get(entry.id);
    if (!ranked) throw new Error(`ablate: internal error -- no rank computed for readout "${entry.id}"`);

    const baselineResults = results.get(baselineKey(entry.id));
    if (!baselineResults) throw new Error(`ablate: missing baseline results for readout "${entry.id}"`);
    const baselineScores = baselineResults.map((r) => r.movementScore);
    const baselineMean = baselineScores.reduce((a, b) => a + b, 0) / baselineScores.length;

    const ablations: AblationEntry[] = [];
    for (const [rank, inputs] of [
      ['top', ranked.top],
      ['bottom', ranked.bottom]
    ] as const) {
      for (const input of inputs) {
        const ablatedResults = results.get(ablationKey(entry.id, input));
        if (!ablatedResults) throw new Error(`ablate: missing results for readout "${entry.id}" input ${input}`);
        const ablatedScores = ablatedResults.map((r) => r.movementScore);
        const rng = conditionRng(args.bootstrapSeed, `${entry.id}|${input}`);
        const effect = pairedStats(ablatedScores, baselineScores, args.resamples, rng);
        ablations.push({
          input,
          rank,
          saliencyScore: saliencyEntry.thrust[input] + saliencyEntry.yaw[input],
          inputMean: saliencyEntry.inputMean[input],
          inputStd: saliencyEntry.inputStd[input],
          effect
        });
      }
    }
    ablations.sort((a, b) => a.input - b.input);

    outEntries.push({
      id: entry.id,
      graphId: entry.graphId,
      trainerSeed: entry.trainerSeed,
      arenaTask: entry.arenaTask,
      n: baselineScores.length,
      baselineMean,
      ablations
    });
  }
  outEntries.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));

  const body = JSON.stringify({ version: 1, bootstrapSeed: args.bootstrapSeed, resamples: args.resamples, entries: outEntries });
  mkdirSync(resolve(args.out, '..'), { recursive: true });
  atomicWriteFileSync(args.out, body);
  return { out: args.out, count: outEntries.length, sha256: sha256Hex(body) };
};

/** Decode an archive entry's base64 `theta` into a plain flat `number[]` (JSON-transportable over `child_process.fork`'s IPC). */
const decodeThetaArray = (base64: string): number[] => {
  const buffer = Buffer.from(base64, 'base64');
  const aligned = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  return Array.from(new Float32Array(aligned));
};

const main = async (): Promise<void> => {
  const args = parseArgs(process.argv.slice(2));
  const result = await runAblate(args);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing summary output.
  console.log(`ablate: wrote ${result.out} (${result.count} readouts, sha256 ${result.sha256})`);
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
