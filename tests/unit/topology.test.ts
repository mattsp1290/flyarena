import { describe, expect, it } from 'vitest';
import { createDisconnectedGraph, validateGraph } from '../../src/lib/connectome/format';
import { createRandomGraph } from '../fixtures/tiny-graph';

describe('createDisconnectedGraph', () => {
  const source = createRandomGraph(0x1234, { neuronCount: 24, inputChannelCount: 8, outputPopulationCount: 3 });

  it('produces edgeCount 0 with every presynaptic row empty', () => {
    const disconnected = createDisconnectedGraph(source);
    expect(disconnected.metadata.edgeCount).toBe(0);
    expect(disconnected.postsynapticIndices.length).toBe(0);
    expect(disconnected.contactMagnitudes.length).toBe(0);
    expect(Array.from(disconnected.presynapticOffsets)).toEqual(
      new Array(source.metadata.neuronCount + 1).fill(0)
    );
  });

  it('preserves the neuron set, biological ids, signs, and the I/O channel/population mapping exactly', () => {
    const disconnected = createDisconnectedGraph(source);
    expect(disconnected.metadata.neuronCount).toBe(source.metadata.neuronCount);
    expect(Array.from(disconnected.biologicalIds)).toEqual(Array.from(source.biologicalIds));
    expect(Array.from(disconnected.presynapticSigns)).toEqual(Array.from(source.presynapticSigns));
    expect(Array.from(disconnected.inputChannelIndex)).toEqual(Array.from(source.inputChannelIndex));
    expect(Array.from(disconnected.inputWeight)).toEqual(Array.from(source.inputWeight));
    expect(Array.from(disconnected.outputPopulationIndex)).toEqual(Array.from(source.outputPopulationIndex));
    expect(Array.from(disconnected.outputWeight)).toEqual(Array.from(source.outputWeight));
    expect(disconnected.metadata.inputChannelCount).toBe(source.metadata.inputChannelCount);
    expect(disconnected.metadata.outputPopulationCount).toBe(source.metadata.outputPopulationCount);
  });

  it('preserves every other dynamics parameter unchanged', () => {
    const disconnected = createDisconnectedGraph(source);
    const { edgeCount: _sourceEdgeCount, ...restSource } = source.metadata;
    const { edgeCount: _disconnectedEdgeCount, ...restDisconnected } = disconnected.metadata;
    expect(restDisconnected).toEqual(restSource);
  });

  it('is itself a valid graph', () => {
    const disconnected = createDisconnectedGraph(source);
    expect(() => validateGraph(disconnected)).not.toThrow();
  });

  it('does not mutate the source graph', () => {
    const beforeEdgeCount = source.metadata.edgeCount;
    const beforePostsynaptic = Array.from(source.postsynapticIndices);
    createDisconnectedGraph(source);
    expect(source.metadata.edgeCount).toBe(beforeEdgeCount);
    expect(Array.from(source.postsynapticIndices)).toEqual(beforePostsynaptic);
  });
});
