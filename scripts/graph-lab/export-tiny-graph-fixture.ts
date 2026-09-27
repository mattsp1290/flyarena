import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { createRandomGraph } from '../../tests/fixtures/tiny-graph';
import { sha256Hex } from '../training/fsio';

/**
 * One-off generator for `backend/graph_lab/tests/fixtures/tiny.bin.gz` and
 * `tiny.manifest.json`: `backend/graph_lab/tests/test_engines.py`'s fixture
 * graph (`.agents/plans/graph-lab/02-job-engines.md`'s Tests section:
 * "`test_engines.py` runs fixture-graph versions of each engine, with small
 * bounds and CPU-only atlas"). Not part of the WP2 runtime path -- a
 * one-time (rerun only if this file's own `createRandomGraph` call
 * changes) fixture generator, in the same spirit as
 * `scripts/analysis/export-trace-graph-fixture.ts` producing
 * `tests_python/fixtures/trace-graph-transfer.json`. Its output is
 * committed and read back by Python without spawning Node for this part --
 * `test_engines.py`'s own engine-dispatch tests still spawn Node/atlas_cli
 * for the parts that are genuinely under test.
 *
 * `createRandomGraph` (not `createTinyGraph`, `tests/unit/
 * graph-lab-entries.test.ts`'s own lesion/swapset fixture): the swap-set
 * engine's class-matched controls (`swap_ops.random_class_swaps`) need at
 * least one "bridge" neuron (neither input- nor output-labeled --
 * `swap_ops.bridge_mask_of`'s own definition) to have any legal candidate
 * class at all, and `createTinyGraph`'s 4 neurons are *all*
 * input-or-output-labeled (confirmed empirically: `random_class_swaps`
 * raises "empty candidate class" against it) -- `createRandomGraph`'s
 * default topology always leaves `neuronCount - inputChannelCount -
 * outputPopulationCount` neurons unlabeled, so a small
 * (`neuronCount: 12`) instance has 8 bridge neurons while staying fast
 * enough for a CPU-only engine test.
 */
const graph = createRandomGraph(42, { neuronCount: 12, inputChannelCount: 2, outputPopulationCount: 2 });
const binary = new Uint8Array(encodeGraphBinary(graph));
const binarySha256 = sha256Hex(binary);
const gzipped = gzipSync(binary);
const gzipSha256 = sha256Hex(gzipped);

const here = dirname(fileURLToPath(import.meta.url));
const fixturesDir = resolve(here, '../../backend/graph_lab/tests/fixtures');
mkdirSync(fixturesDir, { recursive: true });

writeFileSync(resolve(fixturesDir, 'tiny.bin.gz'), gzipped);
writeFileSync(
  resolve(fixturesDir, 'tiny.manifest.json'),
  `${JSON.stringify({ artifact: 'tiny.bin.gz', binarySha256, gzipSha256 }, null, 2)}\n`
);

// eslint-disable-next-line no-console -- one-off fixture-generation tool.
console.log(`wrote ${fixturesDir}/tiny.bin.gz (binarySha256=${binarySha256}, gzipSha256=${gzipSha256})`);
