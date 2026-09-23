import type { ConnectomeGraph, GraphMetadata } from '../../src/lib/connectome/format';
import { mulberry32 } from '../../src/lib/random/mulberry32';

/**
 * FIXTURE-ONLY degree-preserving directed double-edge-swap rewiring, for
 * `scripts/training/export-arms.ts`'s trace-graph development/testing mode
 * only (`--fixture-rewire`, which `export-arms.ts` refuses to combine with
 * `--graph`).
 *
 * This is NOT the product's rewiring algorithm. The real rewired control
 * (`public/data/malecns-arena-v1-rewired-seed0.bin.gz`) is produced offline
 * by `scripts/data/rewire.py` — a separate, independently-reviewed
 * implementation (see `docs/data-provenance.md`'s "Rewired control arm"
 * section) that this file does not call, wrap, or attempt to reproduce
 * bit-for-bit. This exists solely so the trace graph can exercise all three
 * exported arms (biological/rewired/disconnected) end-to-end — including
 * `export-arms.ts`'s "D equal across arms" gate — without a real rewired
 * artifact on hand. `SerializedArmBundle.provenance` records
 * `{ kind: 'rewired-fixture-only-swap' }` for any graph built this way, so
 * it can never be mistaken for a real artifact downstream.
 *
 * Preserves, by construction (not by a post-hoc repair pass): the node
 * count, every neuron's in-degree and out-degree, the edge-weight
 * multiset, and Dale's-law sign ownership (`presynapticSigns` is
 * per-presynaptic-neuron and is never read or written here — a swap only
 * ever changes which postsynaptic neuron an existing presynaptic row's
 * edge targets). Rejects a candidate swap that would create a self-loop or
 * a duplicate `(pre, post)` edge, mirroring `rewire.py`'s own policy (see
 * `docs/data-provenance.md`'s "Self-loop / duplicate policy").
 */
export const createFixtureRewiredTraceGraph = (
  graph: Readonly<ConnectomeGraph>,
  seed: number
): ConnectomeGraph => {
  const { neuronCount, edgeCount } = graph.metadata;
  const random = mulberry32(seed);

  interface MutableEdge {
    pre: number;
    post: number;
    magnitude: number;
  }
  const edges: MutableEdge[] = [];
  const adjacency: Array<Set<number>> = Array.from({ length: neuronCount }, () => new Set());
  for (let pre = 0; pre < neuronCount; pre += 1) {
    const start = graph.presynapticOffsets[pre];
    const end = graph.presynapticOffsets[pre + 1];
    for (let edge = start; edge < end; edge += 1) {
      const post = graph.postsynapticIndices[edge];
      edges.push({ pre, post, magnitude: graph.contactMagnitudes[edge] });
      adjacency[pre].add(post);
    }
  }

  // 20x the edge count, matching the order of magnitude of rewire.py's own
  // attempt budget (docs/data-provenance.md records "20x the edge count"
  // for the real artifact); this fixture graph is tiny, so this comfortably
  // mixes the topology without needing a convergence check.
  const attempts = edgeCount * 20;
  for (let attempt = 0; attempt < attempts && edges.length >= 2; attempt += 1) {
    const i = Math.floor(random() * edges.length);
    const j = Math.floor(random() * edges.length);
    if (i === j) continue;
    const edgeI = edges[i];
    const edgeJ = edges[j];
    const { pre: a, post: b } = edgeI;
    const { pre: c, post: d } = edgeJ;
    if (a === c || b === d) continue; // degenerate: shared source or shared target
    if (a === d || c === b) continue; // would create a self-loop
    if (adjacency[a].has(d) || adjacency[c].has(b)) continue; // would create a duplicate edge

    adjacency[a].delete(b);
    adjacency[c].delete(d);
    adjacency[a].add(d);
    adjacency[c].add(b);
    edgeI.post = d;
    edgeJ.post = b;
  }

  const rowsByPre: MutableEdge[][] = Array.from({ length: neuronCount }, () => []);
  for (const edge of edges) rowsByPre[edge.pre].push(edge);
  for (const row of rowsByPre) row.sort((left, right) => left.post - right.post);

  const presynapticOffsets = new Uint32Array(neuronCount + 1);
  let cursor = 0;
  for (let pre = 0; pre < neuronCount; pre += 1) {
    presynapticOffsets[pre] = cursor;
    cursor += rowsByPre[pre].length;
  }
  presynapticOffsets[neuronCount] = cursor;

  const postsynapticIndices = new Uint32Array(edgeCount);
  const contactMagnitudes = new Float32Array(edgeCount);
  let index = 0;
  for (const row of rowsByPre) {
    for (const edge of row) {
      postsynapticIndices[index] = edge.post;
      contactMagnitudes[index] = edge.magnitude;
      index += 1;
    }
  }

  const metadata: GraphMetadata = { ...graph.metadata, edgeCount: postsynapticIndices.length };

  return {
    metadata,
    biologicalIds: graph.biologicalIds.slice(),
    presynapticOffsets,
    postsynapticIndices,
    contactMagnitudes,
    presynapticSigns: graph.presynapticSigns.slice(),
    inputChannelIndex: graph.inputChannelIndex.slice(),
    inputWeight: graph.inputWeight.slice(),
    outputPopulationIndex: graph.outputPopulationIndex.slice(),
    outputWeight: graph.outputWeight.slice()
  };
};
