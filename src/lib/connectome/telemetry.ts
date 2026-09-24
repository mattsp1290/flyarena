import type { NeuralModelState } from './model';

/**
 * Compact per-step summary of network activity. This remains a summary, not
 * a per-neuron snapshot: `computeTelemetry` itself never returns the full
 * `rate` array, and a caller gets one regardless of this. The Worker
 * protocol has a *separate*, opt-in full-neuron channel for the anatomical
 * activity view — `StepWorkerSuccess.rates` (`worker/protocol.ts`), sent
 * only while a caller has enabled it via `set-activity` (default off, and
 * always off again after a fresh `init`). With that view closed, the main
 * thread still never receives full-neuron state each frame, exactly as
 * before; with it open, `rates` is additive to (not a replacement for) this
 * summary, which every caller keeps receiving unconditionally either way.
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
    // Math.min/Math.max (unlike a manual `<`/`>` comparison) propagate NaN:
    // a non-finite rate should make a non-finite summary visible rather than
    // being silently skipped and reported as a misleadingly ordinary 0.
    min = Math.min(min, value);
    max = Math.max(max, value);
    if (Math.abs(value) > ACTIVITY_THRESHOLD) active += 1;
  }

  return {
    meanRate: sum / count,
    minRate: min,
    maxRate: max,
    activeFraction: active / count
  };
};
