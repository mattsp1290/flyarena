import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

import { buildGraphBufferForMode } from '../../src/lib/experiment/bindings';
import { parseGraphBinary, type ConnectomeGraph, type GraphMode } from '../../src/lib/connectome/format';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { runEpisode } from '../training/episode';
import { sha256Hex } from '../training/fsio';

/**
 * `regime-check.ts`'s child process: the WP2 counterpart of
 * `null-worker.ts`, forked once per shard exactly the same way (see that
 * file's own module doc comment for why `fork` + `execArgv` inheritance,
 * not `worker_threads`). Deliberately a **separate, new** file rather than
 * an edit to `null-worker.ts` itself: `.agents/plans/null-explanation/
 * 02-transfer-and-features.md`'s WP2 (this file) and a concurrent bean
 * (`null-worker.ts`/`null-evaluate.ts`/`null-report.ts`'s owner) are both in
 * flight against `main`, so this module owns its own worker/task/result/
 * message protocol instead of touching those files. It duplicates a small
 * amount of `null-worker.ts` logic that exists there as private
 * (non-exported) helpers -- `loadVerifiedGraphBinary`/`graphFromTask` --
 * which this file cannot import without either exporting them (editing the
 * other bean's file) or forking the whole module; the duplicated surface is
 * intentionally minimal (gzip + sha256 verify + `buildGraphBufferForMode`),
 * not a second copy of anything sharded/scored.
 *
 * Computes, per held-out seed, per the plan's "Regime check" section:
 * - `clampFraction`: the fraction of neuron-substeps (`neuronCount *
 *   substeps * ticks`) with `rate` sitting exactly at `rateMin`/`rateMax`
 *   (`stepModel`'s `clamp` assigns the exact bound when a value is out of
 *   range, so exact float equality is the correct "clamp active" test, not
 *   an approximation);
 * - `steadyStateDistance`: `||r_t - r*(u_t)|| / ||r*(u_t)||` averaged over
 *   ticks, where `r_t` is the network's actual rate at the end of a tick's
 *   substep loop and `r*(u_t) = (lambda I - g A)^-1 B clamp(u_t)` is the
 *   linear fixed point for that tick's held-constant input -- computed via
 *   the per-graph steady-state map `scripts/analysis/transfer.py` already
 *   solved and wrote to `--steady-state-dir` (`M = (lambda I - g A)^-1 B`,
 *   shape `neuronCount x inputChannelCount`, so `r*(u_t) = M @
 *   clamp(u_t)` is one `O(neuronCount * inputChannelCount)` matrix-vector
 *   product per tick, not a second dense solve in TypeScript). When
 *   `||r*(u_t)|| == 0` (e.g. an all-zero clamped input), the tick is scored
 *   `0` if `r_t` is also exactly zero (both sides of the fixed point
 *   trivially agree) and `1` otherwise (maximally divergent from a
 *   zero-length reference), rather than dividing by zero.
 *
 * Uses `runEpisode`'s new `onSubstep` hook (`AgentEpisodeConfig`,
 * `scripts/training/episode.ts`) exactly as `.agents/plans/
 * null-explanation/02-transfer-and-features.md` specifies: authored decoder
 * only, opponent parked (matching `null-worker.ts`'s own condition), so
 * this reuses the same closed-loop tick order the authored null evaluation
 * itself uses -- no second, hand-rolled simulation loop.
 */

export type RegimeTaskMode = GraphMode;

export interface RegimeWorkerTask {
  readonly graphId: string;
  readonly mode: RegimeTaskMode;
  /** Gzip-compressed graph binary path. For `disconnected`, this is the *biological* source graph, matching `null-worker.ts`'s own `NullWorkerTask.path` convention. */
  readonly path: string;
  /** Expected sha256 of the *decompressed* binary at `path`. */
  readonly expectedSha256: string;
  /** `scripts/analysis/transfer.py`'s per-graph steady-state map sidecar (`<graphId>.steadystate.f64`): raw float64, row-major `neuronCount x inputChannelCount`. */
  readonly steadyStatePath: string;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
}

export interface RegimeSeedResult {
  readonly seed: number;
  readonly clampFraction: number;
  readonly steadyStateDistance: number;
}

export interface RegimeWorkerResultMessage {
  readonly type: 'result';
  readonly graphId: string;
  readonly results: readonly RegimeSeedResult[];
}

export interface RegimeWorkerErrorMessage {
  readonly type: 'error';
  readonly graphId: string;
  readonly message: string;
}

export type RegimeWorkerMessage = RegimeWorkerResultMessage | RegimeWorkerErrorMessage;

const clamp = (value: number, minimum: number, maximum: number): number =>
  value < minimum ? minimum : value > maximum ? maximum : value;

/** Mirrors `null-worker.ts`'s private `loadVerifiedGraphBinary` (not importable -- see this file's module doc comment). */
const loadVerifiedGraphBinary = (path: string, expectedSha256: string): ArrayBuffer => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actualSha256 = sha256Hex(binary);
  if (actualSha256 !== expectedSha256) {
    throw new Error(
      `regime-worker: ${path} decompressed sha256 ${actualSha256} does not match expected ${expectedSha256}`
    );
  }
  return binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
};

const EMPTY_BUFFER = new ArrayBuffer(0);

/** Mirrors `null-worker.ts`'s private `graphFromTask` (not importable -- see this file's module doc comment). */
const graphFromTask = (task: RegimeWorkerTask, graphBinary: ArrayBuffer): ConnectomeGraph => {
  const baseBuffer = task.mode === 'biological' || task.mode === 'disconnected' ? graphBinary : EMPTY_BUFFER;
  const rewiredBuffer = task.mode === 'rewired' ? graphBinary : EMPTY_BUFFER;
  const modeBuffer = buildGraphBufferForMode(baseBuffer, rewiredBuffer, task.mode);
  return parseGraphBinary(modeBuffer);
};

/** Load a `transfer.py`-written steady-state sidecar as a `Float64Array`. `.slice()` on the underlying `ArrayBuffer` guarantees 8-byte alignment regardless of the `Buffer`'s own pool offset (the same pattern `loadVerifiedGraphBinary` uses for the decompressed graph binary above). */
const loadSteadyStateMap = (path: string): Float64Array => {
  const buffer = readFileSync(path);
  const aligned = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  if (aligned.byteLength % 8 !== 0) {
    throw new Error(`regime-worker: ${path} is not a whole number of float64 values (${aligned.byteLength} bytes)`);
  }
  return new Float64Array(aligned);
};

/** Exported for `tests/unit/regime-check.test.ts` to call directly (in-process, no `fork`/IPC) against a hand-built fixture. */
export const runTask = (task: RegimeWorkerTask): readonly RegimeSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary(task.path, task.expectedSha256);
  const graph = graphFromTask(task, graphBinary);
  const steadyStateMap = loadSteadyStateMap(task.steadyStatePath);

  const { neuronCount, inputChannelCount, rateMin, rateMax, inputClampMin, inputClampMax } = graph.metadata;
  const expectedSteadyStateLength = neuronCount * inputChannelCount;
  if (steadyStateMap.length !== expectedSteadyStateLength) {
    throw new Error(
      `regime-worker: ${task.steadyStatePath} has ${steadyStateMap.length} float64 values, expected ` +
        `${expectedSteadyStateLength} (neuronCount ${neuronCount} x inputChannelCount ${inputChannelCount})`
    );
  }

  const substeps = NEURAL_SUBSTEPS_PER_TICK;
  const uClamped = new Float64Array(inputChannelCount);

  return task.heldOutSeeds.map((seed) => {
    let clampedNeuronSubsteps = 0;
    let totalNeuronSubsteps = 0;
    let substepIndex = 0;
    let tickCount = 0;
    let distanceSum = 0;

    const onSubstep = (rate: Float32Array, channelValues: ArrayLike<number>): void => {
      for (let neuron = 0; neuron < neuronCount; neuron += 1) {
        totalNeuronSubsteps += 1;
        const value = rate[neuron];
        if (value === rateMin || value === rateMax) clampedNeuronSubsteps += 1;
      }

      substepIndex += 1;
      if (substepIndex < substeps) return;
      substepIndex = 0;
      tickCount += 1;

      for (let channel = 0; channel < inputChannelCount; channel += 1) {
        uClamped[channel] = clamp(channelValues[channel] ?? 0, inputClampMin, inputClampMax);
      }

      let diffSquaredSum = 0;
      let normSquaredSum = 0;
      for (let neuron = 0; neuron < neuronCount; neuron += 1) {
        let predicted = 0;
        const rowBase = neuron * inputChannelCount;
        for (let channel = 0; channel < inputChannelCount; channel += 1) {
          predicted += steadyStateMap[rowBase + channel] * uClamped[channel];
        }
        const diff = rate[neuron] - predicted;
        diffSquaredSum += diff * diff;
        normSquaredSum += predicted * predicted;
      }

      const norm = Math.sqrt(normSquaredSum);
      // See this file's module doc comment for the zero-norm convention.
      const tickDistance = norm > 0 ? Math.sqrt(diffSquaredSum) / norm : diffSquaredSum === 0 ? 0 : 1;
      distanceSum += tickDistance;
    };

    // The episode's own score (movementScore/foodPickups/hazardContacts) is
    // not this metric's concern -- only the per-substep observations
    // `onSubstep` recorded above are. `null-worker.ts`'s equivalent path
    // reads `result.left`; this one deliberately does not.
    runEpisode({
      seed,
      ticks: task.ticks,
      left: { decoder: 'authored', graph, onSubstep },
      right: { decoder: 'parked' }
    });

    const clampFraction = totalNeuronSubsteps > 0 ? clampedNeuronSubsteps / totalNeuronSubsteps : 0;
    const steadyStateDistance = tickCount > 0 ? distanceSum / tickCount : 0;
    if (!Number.isFinite(clampFraction) || !Number.isFinite(steadyStateDistance)) {
      throw new Error(
        `regime-worker: ${task.graphId} seed ${seed} produced a non-finite regime metric ` +
          `(clampFraction=${clampFraction}, steadyStateDistance=${steadyStateDistance})`
      );
    }
    return { seed, clampFraction, steadyStateDistance };
  });
};

/**
 * Wire protocol: byte-for-byte the same shape as `null-worker-shared.ts`'s
 * `runWorkerMain` (not reused directly -- that helper is generic only over
 * `Task`, hard-coding `NullSeedResult`/`NullWorkerMessage` as its result/
 * message types, which do not fit this file's own result shape).
 */
process.on('message', (task: RegimeWorkerTask) => {
  try {
    const results = runTask(task);
    const message: RegimeWorkerMessage = { type: 'result', graphId: task.graphId, results };
    process.send?.(message);
  } catch (error) {
    const message: RegimeWorkerMessage = {
      type: 'error',
      graphId: task.graphId,
      message: error instanceof Error ? error.message : String(error)
    };
    process.send?.(message);
  }
});
