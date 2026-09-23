import { describe, expect, it } from 'vitest';
import {
  encodeGraphBinary,
  parseGraphBinary,
  SUPPORTED_FORMAT_VERSION,
  validateGraph,
  type ConnectomeGraph
} from '../../src/lib/connectome/format';
import { createTinyGraph } from '../fixtures/tiny-graph';

describe('connectome graph binary format', () => {
  it('round-trips a graph through encodeGraphBinary/parseGraphBinary byte-for-byte equivalent', () => {
    const original = createTinyGraph();

    const parsed = parseGraphBinary(encodeGraphBinary(original));

    expect(parsed.metadata).toEqual(original.metadata);
    expect(Array.from(parsed.biologicalIds)).toEqual(Array.from(original.biologicalIds));
    expect(Array.from(parsed.presynapticOffsets)).toEqual(Array.from(original.presynapticOffsets));
    expect(Array.from(parsed.postsynapticIndices)).toEqual(
      Array.from(original.postsynapticIndices)
    );
    expect(Array.from(parsed.contactMagnitudes)).toEqual(Array.from(original.contactMagnitudes));
    expect(Array.from(parsed.presynapticSigns)).toEqual(Array.from(original.presynapticSigns));
    expect(Array.from(parsed.inputChannelIndex)).toEqual(Array.from(original.inputChannelIndex));
    expect(Array.from(parsed.inputWeight)).toEqual(Array.from(original.inputWeight));
    expect(Array.from(parsed.outputPopulationIndex)).toEqual(
      Array.from(original.outputPopulationIndex)
    );
    expect(Array.from(parsed.outputWeight)).toEqual(Array.from(original.outputWeight));
  });

  it('parses a zero-edge disconnected graph with the same node/IO layout', () => {
    const biological = createTinyGraph();
    const disconnected: ConnectomeGraph = {
      ...biological,
      metadata: { ...biological.metadata, edgeCount: 0 },
      presynapticOffsets: Uint32Array.from([0, 0, 0, 0, 0]),
      postsynapticIndices: new Uint32Array(0),
      contactMagnitudes: new Float32Array(0)
    };

    const parsed = parseGraphBinary(encodeGraphBinary(disconnected));

    expect(parsed.metadata.edgeCount).toBe(0);
    expect(parsed.metadata.neuronCount).toBe(biological.metadata.neuronCount);
    expect(Array.from(parsed.inputChannelIndex)).toEqual(
      Array.from(biological.inputChannelIndex)
    );
    expect(Array.from(parsed.outputPopulationIndex)).toEqual(
      Array.from(biological.outputPopulationIndex)
    );
  });

  it('rejects a buffer whose magic bytes do not match', () => {
    const buffer = encodeGraphBinary(createTinyGraph());
    new DataView(buffer).setUint8(0, 'X'.charCodeAt(0));

    expect(() => parseGraphBinary(buffer)).toThrow(/magic/);
  });

  it('rejects an unsupported format version before stepping', () => {
    const buffer = encodeGraphBinary(createTinyGraph());
    new DataView(buffer).setUint32(4, SUPPORTED_FORMAT_VERSION + 1, true);

    expect(() => parseGraphBinary(buffer)).toThrow(/formatVersion/);
  });

  it('rejects a truncated buffer', () => {
    const buffer = encodeGraphBinary(createTinyGraph());

    expect(() => parseGraphBinary(buffer.slice(0, buffer.byteLength - 4))).toThrow(/truncated/);
  });

  it('rejects mismatched section array lengths', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      presynapticOffsets: Uint32Array.from([0, 1, 2, 2]) // missing the trailing entry
    };

    expect(() => validateGraph(malformed)).toThrow(/presynapticOffsets/);
  });

  it('rejects an out-of-range postsynaptic index', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      postsynapticIndices: Uint32Array.from([99, 3])
    };

    expect(() => validateGraph(malformed)).toThrow(/postsynapticIndices/);
  });

  it('rejects a non-positive contact magnitude', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      contactMagnitudes: Float32Array.from([0, 1.5])
    };

    expect(() => validateGraph(malformed)).toThrow(/contactMagnitudes/);
  });

  it('rejects a presynaptic sign that is not exactly -1 or 1', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      presynapticSigns: Int8Array.from([1, 0, 1, 1])
    };

    expect(() => validateGraph(malformed)).toThrow(/presynapticSigns/);
  });

  it('rejects an out-of-range input channel index', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      inputChannelIndex: Int32Array.from([0, 7, -1, -1]) // inputChannelCount is 2
    };

    expect(() => validateGraph(malformed)).toThrow(/inputChannelIndex/);
  });

  it('rejects an out-of-range output population index', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      outputPopulationIndex: Int32Array.from([-1, -1, 0, 9]) // outputPopulationCount is 2
    };

    expect(() => validateGraph(malformed)).toThrow(/outputPopulationIndex/);
  });

  it('rejects non-finite metadata parameters', () => {
    const graph = createTinyGraph({ leakRate: Number.NaN });

    expect(() => validateGraph(graph)).toThrow(/leakRate/);
  });

  it('rejects inverted rate/clamp bounds', () => {
    expect(() => validateGraph(createTinyGraph({ rateMin: 1, rateMax: -1 }))).toThrow(/rateMin/);
    expect(() =>
      validateGraph(createTinyGraph({ inputClampMin: 1, inputClampMax: -1 }))
    ).toThrow(/inputClampMin/);
  });
});
