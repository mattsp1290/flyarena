import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { OUTPUT_POPULATION } from '../../src/lib/arena/actions';
import { outputNeuronIndices, type ReadoutWeights } from '../../src/lib/connectome/readout';
import { runEpisode } from '../training/episode';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import {
  DEFAULT_ARCHIVE_PATH,
  defaultResolveGraphConfig,
  graphForEntry,
  loadArchive,
  parsePathFlags,
  resolvePathFlag,
  SALIENCY_SEEDS,
  SALIENCY_TICKS,
  weightsForEntry
} from './shared';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 analysis 1:
 * input saliency per descending neuron (`00-overview.md`'s predeclared
 * analysis 1). The full chain rule for the readout's `D -> H -> 3` MLP
 * (`readoutForward`, `src/lib/connectome/readout.ts`):
 *
 *   d(out_o)/d(r_d) = f_o'(z2_o) * sum_h W2[o,h] * (1 - h_h^2) * W1[h,d]
 *
 * where `h = tanh(W1 r + b1)`, `z2 = W2 h + b2`, `f_o'` is `1 - tanh(z2_o)^2
 * = 1 - out_o^2` for the `tanh` outputs (thrust, yaw; brake's `sigmoid`
 * derivative is not needed -- only thrust/yaw saliency is predeclared).
 * `readoutSaliency` recomputes `h`/`out` itself, in the exact same
 * `Float32Array`-write precision `readoutForward` uses at every step (not
 * by calling `readoutForward` and separately re-deriving `h`, which would
 * risk a silent precision drift between the two) -- see
 * `tests/unit/saliency.test.ts`'s finite-difference check for why this
 * matters (the gate is "matches finite differences ... within 1e-6").
 *
 * `readoutSaliency`'s reported figure is the MEAN ABSOLUTE gradient over
 * the trajectory: `d(out_o)/d(r_d)` is computed once per timestep (summed
 * over `h` first, THEN multiplied by `f_o'`, THEN `Math.abs`'d), and those
 * per-timestep absolute values are averaged over `T` -- not the sum of
 * per-hidden-unit absolute contributions, which would overstate the true
 * gradient magnitude whenever hidden units partially cancel.
 *
 * The secondary "variance-weighted" saliency (`00-overview.md`: "gradient
 * x that input's trajectory std") is the already-computed mean-absolute
 * saliency for input `d`, scaled by that same input's own empirical
 * trajectory standard deviation -- a post-hoc rescaling by how much the
 * input actually varies, not a second per-timestep gradient computation.
 */

export interface SaliencyResult {
  readonly thrust: readonly number[];
  readonly yaw: readonly number[];
  readonly thrustVarWeighted: readonly number[];
  readonly yawVarWeighted: readonly number[];
  readonly inputMean: readonly number[];
  readonly inputStd: readonly number[];
}

const sigmoid = (value: number): number => 1 / (1 + Math.exp(-value));

/**
 * `rates`: `T` rows, each the readout's own `D`-length gathered input
 * vector (`outputNeuronIndices(graph)` order) for one tick of one held-out
 * seed -- NOT the full per-neuron rate array `onReadoutInput` itself
 * reports (`collectReadoutInputTrajectory` below does that gathering).
 */
export const readoutSaliency = (
  weights: Readonly<ReadoutWeights>,
  rates: readonly Float32Array[]
): SaliencyResult => {
  const { inputSize: D, hiddenSize: H, w1, b1, w2, b2 } = weights;
  if (rates.length === 0) throw new Error('saliency: rates must have at least one row');
  for (const row of rates) {
    if (row.length !== D) {
      throw new Error(`saliency: a trajectory row has length ${row.length}, expected inputSize ${D}`);
    }
  }
  const T = rates.length;

  const inputMean = new Float64Array(D);
  for (const row of rates) for (let d = 0; d < D; d += 1) inputMean[d] += row[d];
  for (let d = 0; d < D; d += 1) inputMean[d] /= T;

  const inputVarSum = new Float64Array(D);
  for (const row of rates) {
    for (let d = 0; d < D; d += 1) {
      const diff = row[d] - inputMean[d];
      inputVarSum[d] += diff * diff;
    }
  }
  const inputStd = new Float64Array(D);
  for (let d = 0; d < D; d += 1) inputStd[d] = Math.sqrt(inputVarSum[d] / T);

  const hidden = new Float32Array(H);
  const out = new Float32Array(3);
  const gradThrustAtTick = new Float64Array(D);
  const gradYawAtTick = new Float64Array(D);
  const thrustAbsSum = new Float64Array(D);
  const yawAbsSum = new Float64Array(D);

  for (const row of rates) {
    // Recomputes readoutForward's own hidden/output pass exactly (same
    // per-element Float32Array writes), so the derivative below is taken
    // of the exact same numeric function the production readout evaluates.
    for (let h = 0; h < H; h += 1) {
      let sum = b1[h];
      const rowOffset = h * D;
      for (let d = 0; d < D; d += 1) sum += w1[rowOffset + d] * row[d];
      hidden[h] = Math.tanh(sum);
    }
    for (let o = 0; o < 3; o += 1) {
      let sum = b2[o];
      const rowOffset = o * H;
      for (let h = 0; h < H; h += 1) sum += w2[rowOffset + h] * hidden[h];
      out[o] = o === OUTPUT_POPULATION.brake ? sigmoid(sum) : Math.tanh(sum);
    }

    const fThrust = 1 - out[OUTPUT_POPULATION.thrust] * out[OUTPUT_POPULATION.thrust];
    const fYaw = 1 - out[OUTPUT_POPULATION.yaw] * out[OUTPUT_POPULATION.yaw];

    gradThrustAtTick.fill(0);
    gradYawAtTick.fill(0);
    for (let h = 0; h < H; h += 1) {
      const dtanh = 1 - hidden[h] * hidden[h];
      const cThrust = w2[OUTPUT_POPULATION.thrust * H + h] * dtanh;
      const cYaw = w2[OUTPUT_POPULATION.yaw * H + h] * dtanh;
      const w1RowOffset = h * D;
      for (let d = 0; d < D; d += 1) {
        const w1hd = w1[w1RowOffset + d];
        gradThrustAtTick[d] += cThrust * w1hd;
        gradYawAtTick[d] += cYaw * w1hd;
      }
    }

    for (let d = 0; d < D; d += 1) {
      thrustAbsSum[d] += Math.abs(fThrust * gradThrustAtTick[d]);
      yawAbsSum[d] += Math.abs(fYaw * gradYawAtTick[d]);
    }
  }

  const thrust = new Array<number>(D);
  const yaw = new Array<number>(D);
  const thrustVarWeighted = new Array<number>(D);
  const yawVarWeighted = new Array<number>(D);
  for (let d = 0; d < D; d += 1) {
    thrust[d] = thrustAbsSum[d] / T;
    yaw[d] = yawAbsSum[d] / T;
    thrustVarWeighted[d] = thrust[d] * inputStd[d];
    yawVarWeighted[d] = yaw[d] * inputStd[d];
  }

  return {
    thrust,
    yaw,
    thrustVarWeighted,
    yawVarWeighted,
    inputMean: Array.from(inputMean),
    inputStd: Array.from(inputStd)
  };
};

/**
 * Drive `seeds.length` held-out `trained`-decoder episodes and gather the
 * readout's own `D`-length input vector at every tick (`onReadoutInput`,
 * gathered through `indices` -- see this module's doc comment), across all
 * seeds, in seed order. Runs in-process (not sharded): at 10 seeds x 1,800
 * ticks per readout this is minutes of work total, per
 * `02-analyses.md`'s "Runs" section.
 */
export const collectReadoutInputTrajectory = (
  graph: Readonly<ConnectomeGraph>,
  weights: Readonly<ReadoutWeights>,
  seeds: readonly number[],
  ticks: number,
  arenaTask?: string
): Float32Array[] => {
  const indices = outputNeuronIndices(graph);
  const rows: Float32Array[] = [];
  for (const seed of seeds) {
    runEpisode({
      seed,
      ticks,
      arenaTask,
      left: {
        decoder: 'trained',
        graph,
        weights,
        onReadoutInput: (rate) => {
          const row = new Float32Array(indices.length);
          for (let i = 0; i < indices.length; i += 1) row[i] = rate[indices[i]];
          rows.push(row);
        }
      },
      right: { decoder: 'parked' }
    });
  }
  return rows;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

interface SaliencyOutputEntry extends SaliencyResult {
  readonly id: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly arenaTask: string;
  readonly seeds: readonly number[];
  readonly ticks: number;
}

interface SaliencyArgs {
  readonly archivePath: string;
  readonly manifestPath?: string;
  readonly interventionIndexPath?: string;
  readonly archivedInterventionIndexPath?: string;
  readonly out: string;
}

const parseArgs = (argv: readonly string[]): SaliencyArgs => {
  let archivePath = DEFAULT_ARCHIVE_PATH;
  let manifestPath: string | undefined;
  let interventionIndexPath: string | undefined;
  let archivedInterventionIndexPath: string | undefined;
  let out = resolve(process.cwd(), 'training/runs/attribution/saliency.json');
  parsePathFlags('saliency', argv, {
    '--archive': (v) => (archivePath = resolvePathFlag(v)),
    '--manifest': (v) => (manifestPath = resolvePathFlag(v)),
    '--intervention-index': (v) => (interventionIndexPath = resolvePathFlag(v)),
    '--archived-intervention-index': (v) => (archivedInterventionIndexPath = resolvePathFlag(v)),
    '--out': (v) => (out = resolvePathFlag(v))
  });
  return { archivePath, manifestPath, interventionIndexPath, archivedInterventionIndexPath, out };
};

export const runSaliency = (args: Readonly<SaliencyArgs>): { readonly out: string; readonly count: number; readonly sha256: string } => {
  const readouts = loadArchive(args.archivePath);
  const resolveConfig = defaultResolveGraphConfig(args);
  const entries: SaliencyOutputEntry[] = [];
  for (const entry of readouts) {
    const weights = weightsForEntry(entry);
    const graph = graphForEntry(entry, resolveConfig);
    const rows = collectReadoutInputTrajectory(graph, weights, SALIENCY_SEEDS, SALIENCY_TICKS, entry.arenaTask);
    const result = readoutSaliency(weights, rows);
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
  const body = JSON.stringify({ version: 1, entries });
  mkdirSync(resolve(args.out, '..'), { recursive: true });
  atomicWriteFileSync(args.out, body);
  return { out: args.out, count: entries.length, sha256: sha256Hex(body) };
};

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));
  const result = runSaliency(args);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing summary output.
  console.log(`saliency: wrote ${result.out} (${result.count} readouts, sha256 ${result.sha256})`);
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
