import { readFileSync } from 'node:fs';

import type { GraphMode } from '../../src/lib/connectome/format';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { runEpisode } from '../training/episode';
import { sha256Hex } from '../training/fsio';
import { graphFromTaskMode, loadVerifiedGraphBinary } from './null-worker-shared';

/**
 * Side-effect-free implementation of the WP2 regime-check task
 * (`.agents/plans/null-explanation/02-transfer-and-features.md`): everything
 * `scripts/null/regime-worker.ts` needs, minus the `process.on('message',
 * ...)` IPC wiring. Split out (a dual-review finding) so
 * `tests/unit/regime-check.test.ts` can import `runTask` as a value without
 * also registering a message listener on the *test process's* own IPC
 * channel -- `regime-worker.ts` was the first WP2 module a test value-imports
 * (the existing `null-worker.ts` tests only `import type`, which is erased
 * at runtime and registers nothing); under a process-pool test runner, the
 * test process can itself have an IPC channel, and an unguarded top-level
 * `process.on('message', ...)` would receive and mishandle the runner's own
 * messages.
 *
 * Deliberately a **separate, new** file rather than an edit to
 * `null-worker.ts`: `.agents/plans/null-explanation/02-transfer-and-features.md`'s
 * WP2 (this file) and a concurrent bean (`null-worker.ts`/`null-evaluate.ts`/
 * `null-report.ts`'s owner) were both in flight against `main`, so this
 * module owns its own worker/task/result/message protocol instead of
 * touching those files directly. It originally duplicated four small pieces
 * of logic that existed in the (then in-flight) protected files only as
 * private helpers -- `null-worker.ts`'s `loadVerifiedGraphBinary`/
 * `graphFromTask`, `null-worker-shared.ts`'s `runWorkerMain` IPC wrapper
 * (used by `regime-worker.ts`, not here), and `null-evaluate.ts`'s
 * `verifyBiologicalSource` (used by `regime-check.ts`). Now that the
 * concurrent bean has merged, those four have been de-duplicated (a
 * thermo-maintainability review finding): `loadVerifiedGraphBinary`/
 * `graphFromTaskMode` moved into the already-side-effect-free
 * `null-worker-shared.ts` and are imported from there below;
 * `runWorkerMain` was generalized to be generic over the `Result`/`Message`
 * types too (see that file), so `regime-worker.ts` now reuses it instead of
 * hand-rolling its own IPC wiring; and `null-evaluate.ts`'s
 * `verifyBiologicalSource` was exported and is imported directly by
 * `regime-check.ts`. Deliberately *not* importing `null-worker.ts` itself
 * here or in `regime-worker.ts`: that module's own body calls
 * `runWorkerMain(runTask)` at top level, registering a real
 * `process.on('message', ...)` listener as a side effect of merely being
 * imported -- exactly what this file's own split from `regime-worker.ts`
 * (described below) exists to avoid on the *test* process's IPC channel.
 * `null-worker-shared.ts` has no such top-level side effects, so importing
 * from it is safe here.
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
  /**
   * sha256 of the sidecar file's raw bytes at `steadyStatePath`, read from
   * `transfer.py`'s `steady-state/manifest.json` by `regime-check.ts`
   * (`.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2: "the
   * CLIs verify each graph file's sha256 ... before computing" -- this
   * extends that same discipline to the sidecar, which is otherwise the one
   * input in this pipeline with no verification tying it to the bytes it
   * was computed from. A stale or partial sidecar left over from a
   * different `transfer.py` run has the same `neuronCount *
   * inputChannelCount` length as a fresh one for every rewiring -- the
   * length check alone (below) cannot catch that; a dual-review finding).
   */
  readonly steadyStateSha256: string;
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

/**
 * Load a `transfer.py`-written steady-state sidecar as a `Float64Array`,
 * verifying its sha256 against `expectedSha256` first (the sidecar's own
 * verification layer -- see `RegimeWorkerTask.steadyStateSha256`'s doc
 * comment). `.slice()` on the underlying `ArrayBuffer` guarantees 8-byte
 * alignment regardless of the `Buffer`'s own pool offset (the same pattern
 * `loadVerifiedGraphBinary` uses for the decompressed graph binary above).
 */
const loadSteadyStateMap = (path: string, expectedSha256: string): Float64Array => {
  const buffer = readFileSync(path);
  const actualSha256 = sha256Hex(buffer);
  if (actualSha256 !== expectedSha256) {
    throw new Error(
      `regime-worker: ${path} sha256 ${actualSha256} does not match the manifest's expected ${expectedSha256} ` +
        '(a stale or partial steady-state sidecar -- rerun scripts/analysis/transfer.py)'
    );
  }
  const aligned = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  if (aligned.byteLength % 8 !== 0) {
    throw new Error(`regime-worker: ${path} is not a whole number of float64 values (${aligned.byteLength} bytes)`);
  }
  return new Float64Array(aligned);
};

/** The pure regime-check task, run once per graph per shard. Exported for `tests/unit/regime-check.test.ts` and `scripts/null/regime-worker.ts` alike -- see this module's own doc comment for why the two are split. */
export const runTask = (task: RegimeWorkerTask): readonly RegimeSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary('regime-worker', task.path, task.expectedSha256);
  const graph = graphFromTaskMode(task.mode, graphBinary);
  const steadyStateMap = loadSteadyStateMap(task.steadyStatePath, task.steadyStateSha256);

  const { neuronCount, inputChannelCount, rateMin, rateMax, inputClampMin, inputClampMax } = graph.metadata;
  const expectedSteadyStateLength = neuronCount * inputChannelCount;
  if (steadyStateMap.length !== expectedSteadyStateLength) {
    throw new Error(
      `regime-worker: ${task.steadyStatePath} has ${steadyStateMap.length} float64 values, expected ` +
        `${expectedSteadyStateLength} (neuronCount ${neuronCount} x inputChannelCount ${inputChannelCount})`
    );
  }
  // The `onSubstep` clamp-fraction check below (`value === rateMin || value
  // === rateMax`) assumes `rateMin < 0 < rateMax`: every real graph today
  // has `rateMin = -2, rateMax = 2` (symmetric around the model's zero-rate
  // rest state, per `docs/graph-format.md`), so a lesioned or never-driven
  // neuron sits at `0`, strictly between the bounds, and is never miscounted
  // as clamped. A hypothetical `rateMin = 0` (a rectified model) would
  // break that assumption -- every silent neuron would then read as
  // "clamped at rateMin" -- so this asserts it rather than silently
  // changing the metric's meaning if that ever changes (a round-2
  // dual-review finding).
  if (!(rateMin < 0 && 0 < rateMax)) {
    throw new Error(
      `regime-worker: ${task.graphId} has rateMin=${rateMin}, rateMax=${rateMax} -- the clamp-fraction metric ` +
        'assumes rateMin < 0 < rateMax (so a silent/lesioned neuron at rate 0 is never miscounted as clamped)'
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
    //
    // `substeps` passed explicitly (not left to `runEpisode`'s own default,
    // which happens to be the same `NEURAL_SUBSTEPS_PER_TICK` this file
    // already uses for tick-boundary detection above): ties the two to one
    // variable, so a future caller that overrides `runEpisode`'s `substeps`
    // could never silently desync `onSubstep`'s tick-boundary counting from
    // the episode's real substep count (a round-2 dual-review finding; no
    // live bug today, since nothing in this file's own call path overrides
    // it, but the coupling was implicit before this line existed).
    runEpisode({
      seed,
      ticks: task.ticks,
      substeps,
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
