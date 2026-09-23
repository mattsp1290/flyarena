/**
 * Versioned binary format for a sparse connectome graph.
 *
 * The exact byte layout is the shared contract between this parser and the
 * offline Python compiler (a later bean) that will emit real MaleCNS-derived
 * artifacts. `docs/graph-format.md` specifies the layout precisely (magic,
 * version, endianness, header fields, section order, dtypes, alignment) so a
 * Python implementer can produce a compatible file without reading this
 * module. Keep the two in lockstep: any change here is a format version bump
 * and a documentation update.
 *
 * Topology label ("biological" | "rewired" | "disconnected") lives outside
 * the binary: the parser and the model below are topology-agnostic. A
 * disconnected control is just a graph with `edgeCount === 0` and the same
 * node/IO layout as its biological counterpart.
 */

/** Magic bytes identifying a FlyArena Neural Graph file, ASCII "FANG". */
const MAGIC = 'FANG';
const MAGIC_BYTES = 4;

/** The only format version this parser accepts. */
export const SUPPORTED_FORMAT_VERSION = 1;

/** Fixed header size in bytes; see docs/graph-format.md for the field table. */
const HEADER_BYTES = 56;

/** Round a byte offset up to the next multiple of 8 (the widest section element). */
const alignTo8 = (offset: number): number => (offset + 7) & ~7;

export type GraphMode = 'biological' | 'rewired' | 'disconnected';

/**
 * Global, per-file scalar parameters. Everything here is authored/calibrated
 * per the model ledger, never a per-neuron measurement.
 */
export interface GraphMetadata {
  formatVersion: number;
  neuronCount: number;
  edgeCount: number;
  inputChannelCount: number;
  outputPopulationCount: number;
  /** Seconds of simulated time integrated per neural substep. */
  timestepSeconds: number;
  /**
   * Continuous-time leak (decay-to-zero) rate in units of 1/second, NOT a
   * per-substep fraction: the per-substep decay is `leakRate * timestepSeconds`
   * (see docs/graph-format.md's "Dynamics" section for the full update
   * equation). Should satisfy `leakRate * timestepSeconds <= 1` for the leak
   * term to decay monotonically rather than overshoot toward the opposite
   * rate bound.
   */
  leakRate: number;
  rateMin: number;
  rateMax: number;
  inputClampMin: number;
  inputClampMax: number;
  /** Global gain multiplying every recurrent synaptic contribution. */
  globalGain: number;
}

/**
 * A parsed, validated sparse connectome graph. All typed-array fields are
 * zero-copy views over one `ArrayBuffer`; nothing here is defensively copied,
 * so callers must not mutate them once construction has returned.
 *
 * Rows are grouped by presynaptic neuron (`presynapticOffsets` is CSR row
 * pointers into `postsynapticIndices`/`contactMagnitudes`): the edges leaving
 * neuron `pre` are `postsynapticIndices[presynapticOffsets[pre] .. presynapticOffsets[pre + 1])`.
 */
export interface ConnectomeGraph {
  readonly metadata: Readonly<GraphMetadata>;
  /** Opaque per-neuron source-dataset identifiers (e.g. MaleCNS body IDs). */
  readonly biologicalIds: BigUint64Array;
  /** CSR row pointers, length `neuronCount + 1`. */
  readonly presynapticOffsets: Uint32Array;
  /** CSR column indices (postsynaptic neuron per edge), length `edgeCount`. */
  readonly postsynapticIndices: Uint32Array;
  /** Positive contact-count magnitudes per edge, length `edgeCount`. */
  readonly contactMagnitudes: Float32Array;
  /** Dale's-law sign (-1 or 1) applied to every edge leaving a neuron. */
  readonly presynapticSigns: Int8Array;
  /** Observation channel index driving a neuron, or -1 if not an input neuron. */
  readonly inputChannelIndex: Int32Array;
  /** Per-neuron scale applied to its assigned input channel value. */
  readonly inputWeight: Float32Array;
  /** Output population index a neuron contributes to, or -1 if not an output neuron. */
  readonly outputPopulationIndex: Int32Array;
  /** Per-neuron scale applied when aggregating into its output population. */
  readonly outputWeight: Float32Array;
}

interface SectionLayout {
  byteOffset: number;
  byteLength: number;
}

interface GraphLayout {
  biologicalIds: SectionLayout;
  presynapticOffsets: SectionLayout;
  postsynapticIndices: SectionLayout;
  contactMagnitudes: SectionLayout;
  presynapticSigns: SectionLayout;
  inputChannelIndex: SectionLayout;
  inputWeight: SectionLayout;
  outputPopulationIndex: SectionLayout;
  outputWeight: SectionLayout;
  totalBytes: number;
}

/**
 * Compute every section's byte offset/length from the neuron/edge counts
 * alone. Shared by the encoder and the parser so their layouts can never
 * drift apart; docs/graph-format.md documents this exact formula.
 */
const computeGraphLayout = (neuronCount: number, edgeCount: number): GraphLayout => {
  let cursor = HEADER_BYTES;
  const section = (elementBytes: number, count: number): SectionLayout => {
    const byteOffset = cursor;
    const byteLength = elementBytes * count;
    cursor = alignTo8(byteOffset + byteLength);
    return { byteOffset, byteLength };
  };

  const biologicalIds = section(8, neuronCount);
  const presynapticOffsets = section(4, neuronCount + 1);
  const postsynapticIndices = section(4, edgeCount);
  const contactMagnitudes = section(4, edgeCount);
  const presynapticSigns = section(1, neuronCount);
  const inputChannelIndex = section(4, neuronCount);
  const inputWeight = section(4, neuronCount);
  const outputPopulationIndex = section(4, neuronCount);
  const outputWeight = section(4, neuronCount);

  return {
    biologicalIds,
    presynapticOffsets,
    postsynapticIndices,
    contactMagnitudes,
    presynapticSigns,
    inputChannelIndex,
    inputWeight,
    outputPopulationIndex,
    outputWeight,
    totalBytes: cursor
  };
};

const invalidGraph = (message: string): never => {
  throw new Error(`Invalid connectome graph: ${message}`);
};

const requireFinite = (value: number, label: string): void => {
  if (!Number.isFinite(value)) invalidGraph(`${label} must be finite`);
};

const requireNonNegativeInteger = (value: number, label: string): void => {
  if (!Number.isInteger(value) || value < 0) invalidGraph(`${label} must be a non-negative integer`);
};

/**
 * Validate structural and numeric invariants without inspecting per-step
 * dynamics. Called by `parseGraphBinary` and available standalone so
 * hand-built graphs (fixtures, future callers) fail before stepping.
 */
export const validateGraph = (graph: ConnectomeGraph): ConnectomeGraph => {
  const { metadata } = graph;

  if (metadata.formatVersion !== SUPPORTED_FORMAT_VERSION) {
    invalidGraph(
      `unsupported formatVersion ${metadata.formatVersion}; expected ${SUPPORTED_FORMAT_VERSION}`
    );
  }
  requireNonNegativeInteger(metadata.neuronCount, 'neuronCount');
  requireNonNegativeInteger(metadata.edgeCount, 'edgeCount');
  requireNonNegativeInteger(metadata.inputChannelCount, 'inputChannelCount');
  requireNonNegativeInteger(metadata.outputPopulationCount, 'outputPopulationCount');
  for (const label of [
    'timestepSeconds',
    'leakRate',
    'rateMin',
    'rateMax',
    'inputClampMin',
    'inputClampMax',
    'globalGain'
  ] as const) {
    requireFinite(metadata[label], label);
  }
  if (metadata.timestepSeconds <= 0) invalidGraph('timestepSeconds must be positive');
  if (metadata.leakRate < 0) invalidGraph('leakRate must be non-negative');
  if (metadata.rateMin > metadata.rateMax) invalidGraph('rateMin must not exceed rateMax');
  if (metadata.inputClampMin > metadata.inputClampMax) {
    invalidGraph('inputClampMin must not exceed inputClampMax');
  }

  const { neuronCount, edgeCount, inputChannelCount, outputPopulationCount } = metadata;
  const lengthChecks: ReadonlyArray<readonly [string, number, number]> = [
    ['biologicalIds', graph.biologicalIds.length, neuronCount],
    ['presynapticOffsets', graph.presynapticOffsets.length, neuronCount + 1],
    ['postsynapticIndices', graph.postsynapticIndices.length, edgeCount],
    ['contactMagnitudes', graph.contactMagnitudes.length, edgeCount],
    ['presynapticSigns', graph.presynapticSigns.length, neuronCount],
    ['inputChannelIndex', graph.inputChannelIndex.length, neuronCount],
    ['inputWeight', graph.inputWeight.length, neuronCount],
    ['outputPopulationIndex', graph.outputPopulationIndex.length, neuronCount],
    ['outputWeight', graph.outputWeight.length, neuronCount]
  ];
  for (const [label, actual, expected] of lengthChecks) {
    if (actual !== expected) {
      invalidGraph(`${label} length ${actual} does not match expected length ${expected}`);
    }
  }

  if (graph.presynapticOffsets[0] !== 0) invalidGraph('presynapticOffsets must start at 0');
  if (graph.presynapticOffsets[neuronCount] !== edgeCount) {
    invalidGraph('presynapticOffsets must end at edgeCount');
  }
  for (let pre = 0; pre < neuronCount; pre += 1) {
    const start = graph.presynapticOffsets[pre];
    const end = graph.presynapticOffsets[pre + 1];
    if (end < start) invalidGraph(`presynapticOffsets must be non-decreasing at row ${pre}`);
  }

  for (let edge = 0; edge < edgeCount; edge += 1) {
    const post = graph.postsynapticIndices[edge];
    if (!Number.isInteger(post) || post < 0 || post >= neuronCount) {
      invalidGraph(`postsynapticIndices[${edge}] is out of range`);
    }
    const magnitude = graph.contactMagnitudes[edge];
    if (!Number.isFinite(magnitude) || magnitude <= 0) {
      invalidGraph(`contactMagnitudes[${edge}] must be finite and positive`);
    }
  }

  for (let neuron = 0; neuron < neuronCount; neuron += 1) {
    const sign = graph.presynapticSigns[neuron];
    if (sign !== -1 && sign !== 1) {
      invalidGraph(`presynapticSigns[${neuron}] must be -1 or 1`);
    }
    const channel = graph.inputChannelIndex[neuron];
    if (!Number.isInteger(channel) || channel < -1 || channel >= inputChannelCount) {
      invalidGraph(`inputChannelIndex[${neuron}] is out of range`);
    }
    if (!Number.isFinite(graph.inputWeight[neuron])) {
      invalidGraph(`inputWeight[${neuron}] must be finite`);
    }
    const population = graph.outputPopulationIndex[neuron];
    if (!Number.isInteger(population) || population < -1 || population >= outputPopulationCount) {
      invalidGraph(`outputPopulationIndex[${neuron}] is out of range`);
    }
    if (!Number.isFinite(graph.outputWeight[neuron])) {
      invalidGraph(`outputWeight[${neuron}] must be finite`);
    }
  }

  return graph;
};

/**
 * Parse and validate a binary graph buffer produced per docs/graph-format.md.
 * All returned typed arrays are zero-copy views over `buffer`; the caller
 * must not mutate `buffer` afterward.
 */
export const parseGraphBinary = (buffer: ArrayBuffer): ConnectomeGraph => {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error('Invalid connectome graph: buffer is smaller than the fixed header');
  }
  const view = new DataView(buffer);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );
  if (magic !== MAGIC) {
    throw new Error(`Invalid connectome graph: bad magic "${magic}", expected "${MAGIC}"`);
  }

  const metadata: GraphMetadata = {
    formatVersion: view.getUint32(4, true),
    neuronCount: view.getUint32(8, true),
    edgeCount: view.getUint32(12, true),
    inputChannelCount: view.getUint32(16, true),
    outputPopulationCount: view.getUint32(20, true),
    timestepSeconds: view.getFloat32(24, true),
    leakRate: view.getFloat32(28, true),
    rateMin: view.getFloat32(32, true),
    rateMax: view.getFloat32(36, true),
    inputClampMin: view.getFloat32(40, true),
    inputClampMax: view.getFloat32(44, true),
    globalGain: view.getFloat32(48, true)
  };
  if (metadata.formatVersion !== SUPPORTED_FORMAT_VERSION) {
    throw new Error(
      `Invalid connectome graph: unsupported formatVersion ${metadata.formatVersion}; expected ${SUPPORTED_FORMAT_VERSION}`
    );
  }

  const layout = computeGraphLayout(metadata.neuronCount, metadata.edgeCount);
  if (buffer.byteLength < layout.totalBytes) {
    throw new Error(
      `Invalid connectome graph: buffer is truncated (${buffer.byteLength} bytes, expected at least ${layout.totalBytes})`
    );
  }

  const graph: ConnectomeGraph = {
    metadata,
    biologicalIds: new BigUint64Array(
      buffer,
      layout.biologicalIds.byteOffset,
      metadata.neuronCount
    ),
    presynapticOffsets: new Uint32Array(
      buffer,
      layout.presynapticOffsets.byteOffset,
      metadata.neuronCount + 1
    ),
    postsynapticIndices: new Uint32Array(
      buffer,
      layout.postsynapticIndices.byteOffset,
      metadata.edgeCount
    ),
    contactMagnitudes: new Float32Array(
      buffer,
      layout.contactMagnitudes.byteOffset,
      metadata.edgeCount
    ),
    presynapticSigns: new Int8Array(
      buffer,
      layout.presynapticSigns.byteOffset,
      metadata.neuronCount
    ),
    inputChannelIndex: new Int32Array(
      buffer,
      layout.inputChannelIndex.byteOffset,
      metadata.neuronCount
    ),
    inputWeight: new Float32Array(buffer, layout.inputWeight.byteOffset, metadata.neuronCount),
    outputPopulationIndex: new Int32Array(
      buffer,
      layout.outputPopulationIndex.byteOffset,
      metadata.neuronCount
    ),
    outputWeight: new Float32Array(buffer, layout.outputWeight.byteOffset, metadata.neuronCount)
  };

  return validateGraph(graph);
};

/**
 * Encode a validated graph into the binary format described by
 * docs/graph-format.md. Primarily a test/tooling counterpart to
 * `parseGraphBinary`: the production artifact is emitted by the offline
 * Python compiler, but round-tripping through this encoder is how this
 * package's own tests and fixtures exercise the exact wire format.
 */
export const encodeGraphBinary = (graph: Readonly<ConnectomeGraph>): ArrayBuffer => {
  validateGraph(graph as ConnectomeGraph);
  const { metadata } = graph;
  const layout = computeGraphLayout(metadata.neuronCount, metadata.edgeCount);
  const buffer = new ArrayBuffer(layout.totalBytes);
  const view = new DataView(buffer);

  for (let index = 0; index < MAGIC_BYTES; index += 1) {
    view.setUint8(index, MAGIC.charCodeAt(index));
  }
  view.setUint32(4, metadata.formatVersion, true);
  view.setUint32(8, metadata.neuronCount, true);
  view.setUint32(12, metadata.edgeCount, true);
  view.setUint32(16, metadata.inputChannelCount, true);
  view.setUint32(20, metadata.outputPopulationCount, true);
  view.setFloat32(24, metadata.timestepSeconds, true);
  view.setFloat32(28, metadata.leakRate, true);
  view.setFloat32(32, metadata.rateMin, true);
  view.setFloat32(36, metadata.rateMax, true);
  view.setFloat32(40, metadata.inputClampMin, true);
  view.setFloat32(44, metadata.inputClampMax, true);
  view.setFloat32(48, metadata.globalGain, true);
  view.setUint32(52, 0, true);

  new BigUint64Array(buffer, layout.biologicalIds.byteOffset, metadata.neuronCount).set(
    graph.biologicalIds
  );
  new Uint32Array(
    buffer,
    layout.presynapticOffsets.byteOffset,
    metadata.neuronCount + 1
  ).set(graph.presynapticOffsets);
  new Uint32Array(buffer, layout.postsynapticIndices.byteOffset, metadata.edgeCount).set(
    graph.postsynapticIndices
  );
  new Float32Array(buffer, layout.contactMagnitudes.byteOffset, metadata.edgeCount).set(
    graph.contactMagnitudes
  );
  new Int8Array(buffer, layout.presynapticSigns.byteOffset, metadata.neuronCount).set(
    graph.presynapticSigns
  );
  new Int32Array(buffer, layout.inputChannelIndex.byteOffset, metadata.neuronCount).set(
    graph.inputChannelIndex
  );
  new Float32Array(buffer, layout.inputWeight.byteOffset, metadata.neuronCount).set(
    graph.inputWeight
  );
  new Int32Array(buffer, layout.outputPopulationIndex.byteOffset, metadata.neuronCount).set(
    graph.outputPopulationIndex
  );
  new Float32Array(buffer, layout.outputWeight.byteOffset, metadata.neuronCount).set(
    graph.outputWeight
  );

  return buffer;
};
