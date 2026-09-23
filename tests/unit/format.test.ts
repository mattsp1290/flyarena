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
  it('matches the documented worked offset example in docs/graph-format.md exactly', () => {
    // Pins the byte layout, not just decoded values: encodeGraphBinary and
    // parseGraphBinary share one `computeGraphLayout`, so a round-trip test
    // alone could not detect the two drifting from docs/graph-format.md's
    // documented offsets together. neuronCount=4, edgeCount=2 (createTinyGraph).
    const buffer = encodeGraphBinary(createTinyGraph());
    const view = new DataView(buffer);

    expect(buffer.byteLength).toBe(200);
    expect(String.fromCharCode(...new Uint8Array(buffer, 0, 4))).toBe('FANG');
    expect(view.getUint32(4, true)).toBe(SUPPORTED_FORMAT_VERSION);
    expect(view.getUint32(52, true)).toBe(0); // reserved flags

    expect(Array.from(new BigUint64Array(buffer, 56, 4))).toEqual([1n, 2n, 3n, 4n]); // biologicalIds @56
    expect(Array.from(new Uint32Array(buffer, 88, 5))).toEqual([0, 1, 2, 2, 2]); // presynapticOffsets @88
    expect(Array.from(new Uint32Array(buffer, 112, 2))).toEqual([2, 3]); // postsynapticIndices @112
    expect(Array.from(new Float32Array(buffer, 120, 2))).toEqual([2, 1.5]); // contactMagnitudes @120
    expect(Array.from(new Int8Array(buffer, 128, 4))).toEqual([1, -1, 1, 1]); // presynapticSigns @128
    expect(Array.from(new Int32Array(buffer, 136, 4))).toEqual([0, 1, -1, -1]); // inputChannelIndex @136
    expect(Array.from(new Float32Array(buffer, 152, 4))).toEqual([1, 1, 0, 0]); // inputWeight @152
    expect(Array.from(new Int32Array(buffer, 168, 4))).toEqual([-1, -1, 0, 1]); // outputPopulationIndex @168
    expect(Array.from(new Float32Array(buffer, 184, 4))).toEqual([0, 0, 1, 1]); // outputWeight @184

    // Every padding gap the doc's worked example declares (rounding each
    // section up to the next 8-byte boundary) must be zero-filled, not
    // leftover/uninitialized bytes.
    expect(Array.from(new Uint8Array(buffer, 108, 4))).toEqual([0, 0, 0, 0]); // 108..112
    expect(Array.from(new Uint8Array(buffer, 132, 4))).toEqual([0, 0, 0, 0]); // 132..136
  });

  it('round-trips every section value through encodeGraphBinary/parseGraphBinary', () => {
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

  it('rejects a negative globalGain (sign belongs to presynapticSigns, not globalGain)', () => {
    expect(() => validateGraph(createTinyGraph({ globalGain: -1 }))).toThrow(/globalGain/);
  });

  it('accepts a zero globalGain', () => {
    expect(() => validateGraph(createTinyGraph({ globalGain: 0 }))).not.toThrow();
  });

  it('rejects a duplicate (pre, post) edge within a presynaptic row', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      // Row 0 now has two edges, both targeting neuron 2.
      presynapticOffsets: Uint32Array.from([0, 2, 2, 2, 2]),
      postsynapticIndices: Uint32Array.from([2, 2]),
      contactMagnitudes: Float32Array.from([1, 1])
    };

    expect(() => validateGraph(malformed)).toThrow(/strictly increasing/);
  });

  it('rejects an out-of-order (non-ascending) postsynaptic index within a row', () => {
    const graph = createTinyGraph();
    const malformed: ConnectomeGraph = {
      ...graph,
      // Row 0 has two edges, descending: 3 then 2.
      presynapticOffsets: Uint32Array.from([0, 2, 2, 2, 2]),
      postsynapticIndices: Uint32Array.from([3, 2]),
      contactMagnitudes: Float32Array.from([1, 1])
    };

    expect(() => validateGraph(malformed)).toThrow(/strictly increasing/);
  });

  it('permits a self-loop (pre === post) as an ordinary single-entry row', () => {
    const graph = createTinyGraph({ edgeCount: 1 });
    const selfLoop: ConnectomeGraph = {
      ...graph,
      // Row 0 has one edge targeting neuron 0 itself.
      presynapticOffsets: Uint32Array.from([0, 1, 1, 1, 1]),
      postsynapticIndices: Uint32Array.from([0]),
      contactMagnitudes: Float32Array.from([1])
    };

    expect(() => validateGraph(selfLoop)).not.toThrow();
  });
});
