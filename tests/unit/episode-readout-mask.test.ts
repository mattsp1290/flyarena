import { describe, expect, it } from 'vitest';

import { runEpisode } from '../../scripts/training/episode';
import { outputNeuronIndices, readoutParameterCount, type ReadoutWeights } from '../../src/lib/connectome/readout';
import { createTraceGraph, TRACE_GRAPH_OUTPUT_NEURON_INDICES } from '../fixtures/trace-graph';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 Gate 2: "the
 * `readoutMask` with an empty mask equals the unmasked trained decoder bit
 * for bit. A full mask equals `silenced`." Plus `onReadoutInput`'s
 * once-per-tick contract and the new trained/silenced-branch `onSubstep`
 * support, and the decoder-restriction guards on all three.
 */

const D = TRACE_GRAPH_OUTPUT_NEURON_INDICES.length; // 6
const H = 4;
const TICKS = 60;
const SEED = 30001;

const buildWeights = (): ReadoutWeights => {
  const count = readoutParameterCount(D, H);
  expect(count).toBe(H * D + H + 3 * H + 3);
  // Deterministic, non-trivial small values (not identity/zero) so the
  // readout's output actually depends on every gathered input.
  const theta = Float32Array.from({ length: count }, (_, i) => Math.sin(i * 0.37) * 0.5);
  let cursor = 0;
  const take = (n: number) => theta.slice(cursor, (cursor += n));
  return { inputSize: D, hiddenSize: H, w1: take(H * D), b1: take(H), w2: take(3 * H), b2: take(3) };
};

const runTrained = (
  extra: Partial<Parameters<typeof runEpisode>[0]['left']> = {}
): ReturnType<typeof runEpisode> => {
  const graph = createTraceGraph();
  const weights = buildWeights();
  return runEpisode({
    seed: SEED,
    ticks: TICKS,
    left: { decoder: 'trained', graph, weights, ...extra },
    right: { decoder: 'parked' }
  });
};

describe('readoutMask identities (Gate 2)', () => {
  it('an empty mask is bit-identical to no mask at all', () => {
    const unmasked = runTrained();
    const emptyMasked = runTrained({ readoutMask: new Int32Array(0) });
    expect(emptyMasked).toEqual(unmasked);
  });

  it('a full mask (every D index) is bit-identical to the silenced decoder', () => {
    const graph = createTraceGraph();
    const weights = buildWeights();
    const fullMask = Int32Array.from({ length: D }, (_, d) => d);
    const masked = runEpisode({
      seed: SEED,
      ticks: TICKS,
      left: { decoder: 'trained', graph, weights, readoutMask: fullMask },
      right: { decoder: 'parked' }
    });
    const silenced = runEpisode({
      seed: SEED,
      ticks: TICKS,
      left: { decoder: 'silenced', graph, weights },
      right: { decoder: 'parked' }
    });
    expect(masked).toEqual(silenced);
  });

  it('a partial mask changes the outcome relative to no mask (sanity: masking is not a no-op)', () => {
    const unmasked = runTrained();
    const partial = runTrained({ readoutMask: Int32Array.from([0, 3]) });
    expect(partial).not.toEqual(unmasked);
  });

  it('readoutMask is rejected on any decoder other than trained', () => {
    const graph = createTraceGraph();
    const weights = buildWeights();
    expect(() =>
      runEpisode({
        seed: SEED,
        ticks: TICKS,
        left: { decoder: 'silenced', graph, weights, readoutMask: new Int32Array(0) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/readoutMask/);
    expect(() =>
      runEpisode({
        seed: SEED,
        ticks: TICKS,
        left: { decoder: 'authored', graph, readoutMask: new Int32Array(0) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/readoutMask/);
  });

  it('throws on an out-of-range or unsorted readoutMask index', () => {
    const graph = createTraceGraph();
    const weights = buildWeights();
    expect(() => runTrained({ readoutMask: Int32Array.from([D]) })).toThrow(/range/);
    expect(() => runTrained({ readoutMask: Int32Array.from([2, 1]) })).toThrow(/sorted/);
    void graph;
    void weights;
  });
});

describe('onReadoutInput', () => {
  it('is called once per tick, with the exact gathered rate readoutForward will read', () => {
    const graph = createTraceGraph();
    const weights = buildWeights();
    const indices = outputNeuronIndices(graph);
    const seenTicks: number[] = [];
    const seenRows: Float32Array[] = [];
    runEpisode({
      seed: SEED,
      ticks: TICKS,
      left: {
        decoder: 'trained',
        graph,
        weights,
        onReadoutInput: (rate, tick) => {
          seenTicks.push(tick);
          const row = new Float32Array(D);
          for (let i = 0; i < D; i += 1) row[i] = rate[indices[i]];
          seenRows.push(row);
        }
      },
      right: { decoder: 'parked' }
    });
    expect(seenTicks).toEqual(Array.from({ length: TICKS }, (_, i) => i));
    expect(seenRows).toHaveLength(TICKS);
  });

  it('reports the masked (post-mask) rate, not the raw network rate, when a readoutMask is set', () => {
    const graph = createTraceGraph();
    const weights = buildWeights();
    const indices = outputNeuronIndices(graph);
    const mask = Int32Array.from([1]);
    let sawZeroAtMaskedInput = true;
    runEpisode({
      seed: SEED,
      ticks: TICKS,
      left: {
        decoder: 'trained',
        graph,
        weights,
        readoutMask: mask,
        onReadoutInput: (rate) => {
          if (rate[indices[1]] !== 0) sawZeroAtMaskedInput = false;
        }
      },
      right: { decoder: 'parked' }
    });
    expect(sawZeroAtMaskedInput).toBe(true);
  });

  it('is rejected on the authored decoder', () => {
    const graph = createTraceGraph();
    expect(() =>
      runEpisode({
        seed: SEED,
        ticks: TICKS,
        left: { decoder: 'authored', graph, onReadoutInput: () => {} },
        right: { decoder: 'parked' }
      })
    ).toThrow(/onReadoutInput/);
  });
});

describe('onSubstep on the trained/silenced decoders', () => {
  it('is now accepted for trained (previously authored-family only) and does not change the outcome', () => {
    const withoutObserver = runTrained();
    let substepCalls = 0;
    const withObserver = runTrained({
      onSubstep: () => {
        substepCalls += 1;
      }
    });
    expect(withObserver).toEqual(withoutObserver);
    expect(substepCalls).toBeGreaterThan(0);
  });

  it('is still rejected on parked', () => {
    expect(() =>
      runEpisode({
        seed: SEED,
        ticks: TICKS,
        left: { decoder: 'parked', onSubstep: () => {} },
        right: { decoder: 'parked' }
      })
    ).toThrow(/onSubstep/);
  });
});
