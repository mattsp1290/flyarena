// Tiny CLI shim for `tests_python/test_interventions.py`'s CLI gate test:
// reads a JSON array of gzip-compressed graph-binary file paths from stdin,
// decompresses and parses each through the REAL `parseGraphBinary`
// (`src/lib/connectome/format.ts` -- not a reimplementation, and it calls
// `validateGraph` internally, so a file this script accepts is guaranteed
// TS-side valid), and writes each graph's degree/weight/sign/labeling
// summary as JSON to stdout. This is `02-intervention-graphs.md`'s gate
// "round-trips through TS `parseGraphBinary` ... or a small tsx check" --
// `scripts/null/lookup-rewired-artifact.ts` only reads an index.json
// artifact field, it never parses a graph binary itself, so this fixture
// exists to actually exercise the TS parser on `interventions.py`'s output.
//
// Run via `node_modules/.bin/tsx` (see `tests_python/fixtures/
// null_stats_cross_check.ts`'s identical stdin/stdout/skip-if-missing
// pattern, which `test_null_stats_cross_check.py` already establishes).

import { gunzipSync } from 'node:zlib';
import { readFileSync } from 'node:fs';

import { parseGraphBinary } from '../../src/lib/connectome/format';

interface Input {
  readonly paths: readonly string[];
}

interface GraphSummary {
  readonly neuronCount: number;
  readonly edgeCount: number;
  readonly outDegree: readonly number[];
  readonly inDegree: readonly number[];
  readonly sortedContactMagnitudes: readonly number[];
  readonly presynapticSigns: readonly number[];
  readonly inputChannelIndex: readonly number[];
  readonly inputWeight: readonly number[];
  readonly outputPopulationIndex: readonly number[];
  readonly outputWeight: readonly number[];
  readonly biologicalIds: readonly string[];
}

const summarize = (path: string): GraphSummary => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  const graph = parseGraphBinary(arrayBuffer);

  const { neuronCount, edgeCount } = graph.metadata;
  const outDegree: number[] = [];
  for (let i = 0; i < neuronCount; i += 1) {
    outDegree.push(graph.presynapticOffsets[i + 1] - graph.presynapticOffsets[i]);
  }
  const inDegree = new Array<number>(neuronCount).fill(0);
  for (let e = 0; e < edgeCount; e += 1) {
    inDegree[graph.postsynapticIndices[e]] += 1;
  }

  return {
    neuronCount,
    edgeCount,
    outDegree,
    inDegree,
    sortedContactMagnitudes: Array.from(graph.contactMagnitudes).sort((a, b) => a - b),
    presynapticSigns: Array.from(graph.presynapticSigns),
    inputChannelIndex: Array.from(graph.inputChannelIndex),
    inputWeight: Array.from(graph.inputWeight),
    outputPopulationIndex: Array.from(graph.outputPopulationIndex),
    outputWeight: Array.from(graph.outputWeight),
    biologicalIds: Array.from(graph.biologicalIds, (id) => id.toString()),
  };
};

const readStdin = async (): Promise<string> => {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf8');
};

const main = async (): Promise<void> => {
  const raw = await readStdin();
  const input = JSON.parse(raw) as Input;
  const summaries = input.paths.map(summarize);
  process.stdout.write(JSON.stringify(summaries));
};

void main();
