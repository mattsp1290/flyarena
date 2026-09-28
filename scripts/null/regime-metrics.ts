import type { SubstepObserver } from '../../src/lib/connectome/model';

/**
 * Decoder-agnostic per-substep regime accumulator, extracted from
 * `regime-task.ts`'s `runTask` (`.agents/plans/readout-attribution/
 * 02-analyses.md`'s WP2: "Extracted from `regime-task.ts`: the
 * decoder-agnostic per-substep clamp-fraction and steady-state-distance
 * accumulators. `regime-task.ts` is refactored to use it, with output bytes
 * unchanged"). `regime-task.ts`'s own module doc comment documents exactly
 * what `clampFraction`/`steadyStateDistance` mean and why (zero-norm tick
 * convention, `rateMin < 0 < rateMax` assumption enforced by callers before
 * constructing an accumulator, `M @ clamp(u_t)` steady-state prediction) --
 * this module is a pure, side-effect-free re-implementation of that same
 * arithmetic, in the exact same operation order, so its numeric output is
 * byte-identical to the pre-extraction inline version for the same inputs.
 *
 * `scripts/attribution/regime.ts` (WP2) reuses this same accumulator
 * against trained-readout episodes (the authored decoder's own `onSubstep`
 * only fires for the authored family until WP2's `episode.ts` change; this
 * module itself has no decoder opinion at all -- it only ever sees
 * `(rate, channelValues)` from whichever `onSubstep` hook the caller wires
 * it to) -- "the same statistic at the same granularity as the explanation
 * study, so the 20% and 0.5 thresholds apply unchanged" (`02-analyses.md`).
 */

export interface RegimeMetricsConfig {
  readonly neuronCount: number;
  readonly inputChannelCount: number;
  readonly rateMin: number;
  readonly rateMax: number;
  readonly inputClampMin: number;
  readonly inputClampMax: number;
  /** `scripts/analysis/transfer.py`'s per-graph steady-state map: raw float64, row-major `neuronCount x inputChannelCount`. */
  readonly steadyStateMap: Float64Array;
  /** Substeps per tick -- must match the `onSubstep` caller's own `runEpisode({ substeps })`, since this accumulator detects tick boundaries by counting substeps, not by any signal from the episode itself. */
  readonly substeps: number;
}

export interface RegimeMetricsResult {
  readonly clampFraction: number;
  readonly steadyStateDistance: number;
}

export interface RegimeAccumulator {
  /** Pass directly as `runEpisode`'s `onSubstep`/`runSubsteps`' `onSubstep` for one episode. Stateful -- construct a fresh accumulator per episode (`createRegimeAccumulator`), never reuse one across episodes/seeds. */
  readonly onSubstep: SubstepObserver;
  /** Finalize this episode's accumulated `clampFraction`/`steadyStateDistance`. Throws if either is non-finite (matching `regime-task.ts`'s pre-extraction guard). `label` is folded into the thrown message so a caller with many concurrent accumulators (e.g. one per readout/seed) can tell which one failed. */
  readonly result: (label: string) => RegimeMetricsResult;
}

const clamp = (value: number, minimum: number, maximum: number): number =>
  value < minimum ? minimum : value > maximum ? maximum : value;

/**
 * One fresh accumulator's worth of mutable state (all `let`s, closed over
 * by the returned `onSubstep`/`result`), matching `regime-task.ts`'s
 * pre-extraction per-seed locals exactly -- same variable names, same
 * update order, same zero-norm tick convention (`result`'s
 * `norm > 0 ? ... : diffSquaredSum === 0 ? 0 : 1`) as the module doc
 * comment above describes.
 *
 * Callers (`regime-task.ts`, `scripts/attribution/regime.ts`) are
 * responsible for asserting `rateMin < 0 && 0 < rateMax` before
 * constructing an accumulator (`regime-task.ts`'s own pre-existing guard,
 * not repeated here -- this module has no decoder or graph-loading
 * opinion, only the accumulator arithmetic).
 */
export const createRegimeAccumulator = (config: Readonly<RegimeMetricsConfig>): RegimeAccumulator => {
  const { neuronCount, inputChannelCount, rateMin, rateMax, inputClampMin, inputClampMax, steadyStateMap, substeps } =
    config;

  let clampedNeuronSubsteps = 0;
  let totalNeuronSubsteps = 0;
  let substepIndex = 0;
  let tickCount = 0;
  let distanceSum = 0;
  const uClamped = new Float64Array(inputChannelCount);

  const onSubstep: SubstepObserver = (rate, channelValues) => {
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
    const tickDistance = norm > 0 ? Math.sqrt(diffSquaredSum) / norm : diffSquaredSum === 0 ? 0 : 1;
    distanceSum += tickDistance;
  };

  const result = (label: string): RegimeMetricsResult => {
    const clampFraction = totalNeuronSubsteps > 0 ? clampedNeuronSubsteps / totalNeuronSubsteps : 0;
    const steadyStateDistance = tickCount > 0 ? distanceSum / tickCount : 0;
    if (!Number.isFinite(clampFraction) || !Number.isFinite(steadyStateDistance)) {
      throw new Error(
        `regime-metrics: ${label} produced a non-finite regime metric ` +
          `(clampFraction=${clampFraction}, steadyStateDistance=${steadyStateDistance})`
      );
    }
    return { clampFraction, steadyStateDistance };
  };

  return { onSubstep, result };
};
