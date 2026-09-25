import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createTraceGraph } from '../../tests/fixtures/trace-graph';
import { createModelState, createOutputBuffer, createStepScratch, stepModel, aggregateOutputs } from '../../src/lib/connectome/model';
import { atomicWriteFileSync } from '../training/fsio';

/**
 * One-off generator for `tests_python/fixtures/trace-graph-transfer.json`:
 * `.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2 gate 2
 * ("On the trace graph fixture ... `T` equals ... a TS finite-difference
 * estimate (constant `u = e_c` run to steady state, compare `O·r`) within
 * 1e-3 relative"). Not part of the WP2 change surface's runtime path (no
 * production script imports this file); it is a one-time (rerun only if
 * `tests/fixtures/trace-graph.ts` or `src/lib/connectome/model.ts`'s
 * dynamics change) fixture generator, in the same spirit as
 * `scripts/training/export-traces.ts` producing `tests/fixtures/golden/`.
 * Its output is committed and read back by `tests_python/test_transfer.py`
 * without spawning Node from pytest.
 *
 * Serializes `createTraceGraph()`'s full `ConnectomeGraph` (so
 * `tests_python/test_transfer.py` can build the same dense `A`/`B`/`O`
 * matrices `scripts/analysis/graph_io.py` would, independent of the binary
 * wire format), plus, for each of the 8 input channels, the result of
 * running `stepModel` repeatedly with a constant, *small-amplitude* input on
 * that channel alone (`u = INPUT_AMPLITUDE * e_c`) until the discretized
 * dynamics have converged, then `O · rate / INPUT_AMPLITUDE`
 * (`aggregateOutputs`, rescaled). Rescaling a small-amplitude probe rather
 * than driving a full unit input matters here: this fixture graph's true
 * (unclamped) linear steady state exceeds `rateMax` in magnitude for some
 * channels at unit input (verified numerically -- `cond(lambda I - g A) ~=
 * 192`, well-conditioned, but the *gain* from some channels to some
 * neurons is large), so a unit-amplitude probe would hit `stepModel`'s
 * `[rateMin, rateMax]` clamp partway through convergence and settle at a
 * clamped fixed point instead of the true linear one -- silently testing
 * the wrong thing. `T` is linear (`r*(a*u) = a*r*(u)` for any scalar `a`,
 * as long as neither `a*u` nor `a*r*` ever crosses a clamp boundary along
 * the way), so a small enough `a` recovers the same `T` exactly, up to the
 * same ~1e-7-relative-per-op float32 rounding budget
 * (`docs/graph-format.md`'s tolerance note) compounded over many
 * iterations to convergence -- exactly the "TS finite-difference estimate"
 * gate 2 asks for. `runToConvergence` asserts no neuron ever reaches
 * `rateMin`/`rateMax` during the run, so a future edit to this fixture (or
 * to `createTraceGraph`) that breaks this assumption fails loudly here
 * instead of silently producing a wrong reference value.
 */

const ITERATIONS = 400_000;
const CONVERGENCE_CHECK_EVERY = 20_000;
const CONVERGENCE_L2_THRESHOLD = 1e-9;
/** Small enough that this fixture's true linear steady state never approaches `graph.metadata.rateMin`/`rateMax` for any of the 8 channels (verified by `runToConvergence`'s own assertion below), large enough that float32 rounding stays well clear of subnormal/precision-loss territory. */
const INPUT_AMPLITUDE = 0.01;

const graph = createTraceGraph();

const serializeGraph = () => ({
  metadata: { ...graph.metadata },
  biologicalIds: Array.from(graph.biologicalIds, (id) => id.toString()),
  presynapticOffsets: Array.from(graph.presynapticOffsets),
  postsynapticIndices: Array.from(graph.postsynapticIndices),
  contactMagnitudes: Array.from(graph.contactMagnitudes),
  presynapticSigns: Array.from(graph.presynapticSigns),
  inputChannelIndex: Array.from(graph.inputChannelIndex),
  inputWeight: Array.from(graph.inputWeight),
  outputPopulationIndex: Array.from(graph.outputPopulationIndex),
  outputWeight: Array.from(graph.outputWeight)
});

const runToConvergence = (channel: number): { outputs: number[]; iterations: number; converged: boolean } => {
  const state = createModelState(graph);
  const scratch = createStepScratch(graph);
  const outputs = createOutputBuffer(graph);
  const channelValues = new Float64Array(graph.metadata.inputChannelCount);
  channelValues[channel] = INPUT_AMPLITUDE;
  const { rateMin, rateMax } = graph.metadata;

  let previous = new Float32Array(state.rate.length);
  let converged = false;
  let iterationsRun = ITERATIONS;
  for (let iteration = 1; iteration <= ITERATIONS; iteration += 1) {
    stepModel(graph, state, scratch, channelValues);
    for (let i = 0; i < state.rate.length; i += 1) {
      if (state.rate[i] === rateMin || state.rate[i] === rateMax) {
        throw new Error(
          `export-trace-graph-fixture: channel ${channel} clamped at neuron ${i} on iteration ${iteration} ` +
            `-- INPUT_AMPLITUDE (${INPUT_AMPLITUDE}) is too large for this graph's true linear gain; reduce it`
        );
      }
    }
    if (iteration % CONVERGENCE_CHECK_EVERY === 0) {
      let sumSquares = 0;
      for (let i = 0; i < state.rate.length; i += 1) {
        const diff = state.rate[i] - previous[i];
        sumSquares += diff * diff;
      }
      previous = Float32Array.from(state.rate);
      if (Math.sqrt(sumSquares) < CONVERGENCE_L2_THRESHOLD) {
        converged = true;
        iterationsRun = iteration;
        break;
      }
    }
  }

  aggregateOutputs(graph, state, outputs);
  const rescaledOutputs = Array.from(outputs, (value) => value / INPUT_AMPLITUDE);
  return { outputs: rescaledOutputs, iterations: iterationsRun, converged };
};

const steadyState = Array.from({ length: graph.metadata.inputChannelCount }, (_, channel) =>
  runToConvergence(channel)
);

for (const [channel, entry] of steadyState.entries()) {
  if (!entry.converged) {
    throw new Error(
      `export-trace-graph-fixture: channel ${channel} did not converge within ${ITERATIONS} iterations ` +
        '(the fixture graph may be unstable at its current globalGain; reduce it and regenerate)'
    );
  }
  // eslint-disable-next-line no-console -- one-off fixture-generation tool.
  console.log(`channel ${channel}: converged after ${entry.iterations} iterations -> outputs ${JSON.stringify(entry.outputs)}`);
}

const payload = {
  description:
    'tests_python/test_transfer.py gate 2 fixture: createTraceGraph() serialized, plus per-channel TS finite-difference T columns (O . rate / INPUT_AMPLITUDE, after stepModel run to convergence with constant u = INPUT_AMPLITUDE * e_c, never clamping -- see this file module doc comment). steadyStateByChannel[c] is directly comparable to transfer_matrix(graph)["T"][:, c].',
  graph: serializeGraph(),
  inputAmplitude: INPUT_AMPLITUDE,
  steadyStateByChannel: steadyState.map((entry) => entry.outputs),
  convergence: { iterationsRequested: ITERATIONS, l2Threshold: CONVERGENCE_L2_THRESHOLD }
};

const here = dirname(fileURLToPath(import.meta.url));
const outPath = resolve(here, '../../tests_python/fixtures/trace-graph-transfer.json');
mkdirSync(dirname(outPath), { recursive: true });
atomicWriteFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
// eslint-disable-next-line no-console -- one-off fixture-generation tool.
console.log(`wrote ${outPath}`);
