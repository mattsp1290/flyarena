import type { NeuralModelState } from './model';

/**
 * Compact per-step summary of network activity. This is deliberately not a
 * per-neuron snapshot: the Worker runtime returns only this summary plus
 * action features, never the full `rate` array, so the main thread never
 * receives full-neuron state each frame.
 */
export interface NeuralTelemetry {
  meanRate: number;
  minRate: number;
  maxRate: number;
  /** Fraction of neurons whose |rate| exceeds a small activity threshold. */
  activeFraction: number;
}

/** |rate| above this counts a neuron as "active" for `activeFraction`. */
const ACTIVITY_THRESHOLD = 1e-3;

/** Summarize `state.rate` in a single allocation-light pass. */
export const computeTelemetry = (state: Readonly<NeuralModelState>): NeuralTelemetry => {
  const { rate } = state;
  const count = rate.length;
  if (count === 0) {
    return { meanRate: 0, minRate: 0, maxRate: 0, activeFraction: 0 };
  }

  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let active = 0;
  for (let neuron = 0; neuron < count; neuron += 1) {
    const value = rate[neuron];
    sum += value;
    if (value < min) min = value;
    if (value > max) max = value;
    if (Math.abs(value) > ACTIVITY_THRESHOLD) active += 1;
  }

  return {
    meanRate: sum / count,
    minRate: min,
    maxRate: max,
    activeFraction: active / count
  };
};
