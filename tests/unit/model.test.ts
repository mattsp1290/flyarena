import { describe, expect, it } from 'vitest';
import {
  aggregateOutputs,
  createModelState,
  createOutputBuffer,
  createStepScratch,
  resetModelState,
  runSubsteps,
  stepModel
} from '../../src/lib/connectome/model';
import { createTinyGraph } from '../fixtures/tiny-graph';

describe('bounded leaky rate network', () => {
  it('drives the postsynaptic target from a presynaptic rate but never the reverse (orientation)', () => {
    const graph = createTinyGraph();
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);

    // Step 1: channel 0 drives neuron 0 to 0.4. The 0 -> 2 edge scatters from
    // the *pre-step* rate (0), so neuron 2 is still untouched this step.
    stepModel(graph, state, scratch, [0.4, 0]);
    expect(state.rate[0]).toBeCloseTo(0.4, 5);
    expect(state.rate[2]).toBe(0);

    // Step 2: neuron 0's now-nonzero rate scatters onto neuron 2.
    stepModel(graph, state, scratch, [0, 0]);
    expect(state.rate[2]).toBeCloseTo(0.8, 5); // sign(+1) * rate(0.4) * gain(1) * magnitude(2)
    expect(state.rate[0]).toBeCloseTo(0.4, 5); // unaffected: no incoming edge to neuron 0
    expect(state.rate[1]).toBe(0);
    expect(state.rate[3]).toBe(0);
  });

  it('does not propagate a directly-set postsynaptic rate back to its presynaptic source', () => {
    const graph = createTinyGraph();
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);
    state.rate[2] = 0.9; // simulate an already-active postsynaptic neuron

    stepModel(graph, state, scratch, [0, 0]);

    expect(state.rate[0]).toBe(0); // no 2 -> 0 edge exists
    expect(state.rate[2]).toBeCloseTo(0.9, 5); // leakRate is 0, so it holds
  });

  it('applies inhibitory sign as a negative postsynaptic drive', () => {
    const graph = createTinyGraph();
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);

    stepModel(graph, state, scratch, [0, 0.5]);
    expect(state.rate[1]).toBeCloseTo(0.5, 5);
    expect(state.rate[3]).toBe(0);

    stepModel(graph, state, scratch, [0, 0]);
    expect(state.rate[3]).toBeCloseTo(-0.75, 5); // sign(-1) * rate(0.5) * gain(1) * magnitude(1.5)
  });

  it('decays an isolated neuron toward zero at the configured leak rate', () => {
    const graph = createTinyGraph({ leakRate: 0.5 });
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);
    state.rate[2] = 0.8; // no incoming drive this step: rate[0] is 0

    stepModel(graph, state, scratch, [0, 0]);

    expect(state.rate[2]).toBeCloseTo(0.4, 5); // 0.8 + 1 * (-0.5 * 0.8 + 0)
  });

  it('scales both the leak term and the drive term by timestepSeconds, not just one of them', () => {
    // Every other case in this file uses timestepSeconds: 1, where `dt * x`
    // and `x` are indistinguishable. A missing `dt` factor anywhere in the
    // update, or one applied to only the leak or only the drive, would still
    // pass every one of those cases; this one would not.
    const graph = createTinyGraph({ timestepSeconds: 0.25, leakRate: 2 });
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);
    state.rate[2] = 0.8; // pre-existing rate on an otherwise-undriven neuron this step

    stepModel(graph, state, scratch, [0.4, 0]);

    // neuron 0 (drive-only, channel 0 = 0.4): 0 + 0.25 * (-2*0 + 1*0.4) = 0.1
    expect(state.rate[0]).toBeCloseTo(0.1, 5);
    // neuron 2 (leak-only this step: rate[0] was 0 pre-step): 0.8 + 0.25 * (-2*0.8 + 0) = 0.4
    expect(state.rate[2]).toBeCloseTo(0.4, 5);
  });

  it('clamps external channel input to [inputClampMin, inputClampMax] before injecting it', () => {
    const highClamp = createTinyGraph({ inputClampMax: 0.2 });
    const highState = createModelState(highClamp);
    stepModel(highClamp, highState, createStepScratch(highClamp), [5, 0]);
    expect(highState.rate[0]).toBeCloseTo(0.2, 5);

    const lowClamp = createTinyGraph({ inputClampMin: -0.3 });
    const lowState = createModelState(lowClamp);
    stepModel(lowClamp, lowState, createStepScratch(lowClamp), [-5, 0]);
    expect(lowState.rate[0]).toBeCloseTo(-0.3, 5);
  });

  it('clamps every neuron rate to [rateMin, rateMax], including recurrently-driven neurons', () => {
    const graph = createTinyGraph({ rateMax: 0.5, inputClampMax: 10, leakRate: 0 });
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);

    stepModel(graph, state, scratch, [2, 0]);
    expect(state.rate[0]).toBeCloseTo(0.5, 5); // 2 clamped to rateMax, not inputClampMax(10)

    stepModel(graph, state, scratch, [0, 0]);
    expect(state.rate[2]).toBeCloseTo(0.5, 5); // 1.0 raw, clamped to rateMax
  });

  it('resets every neuron rate to zero without touching graph buffers', () => {
    const graph = createTinyGraph();
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);
    stepModel(graph, state, scratch, [0.4, 0.4]);
    expect(Array.from(state.rate).some((value) => value !== 0)).toBe(true);

    resetModelState(state);

    expect(Array.from(state.rate)).toEqual([0, 0, 0, 0]);
  });

  it('aggregates per-neuron rates into weighted output populations', () => {
    const graph = createTinyGraph();
    const state = createModelState(graph);
    state.rate.set([0, 0, 0.6, -0.4]);
    const outputs = createOutputBuffer(graph);

    aggregateOutputs(graph, state, outputs);

    expect(outputs[0]).toBeCloseTo(0.6, 5);
    expect(outputs[1]).toBeCloseTo(-0.4, 5);
  });

  it('runSubsteps matches an equivalent manual stepModel loop plus aggregation', () => {
    const graph = createTinyGraph({ leakRate: 0.1 });
    const manualState = createModelState(graph);
    const manualScratch = createStepScratch(graph);
    const manualOutputs = createOutputBuffer(graph);
    for (let step = 0; step < 3; step += 1) {
      stepModel(graph, manualState, manualScratch, [0.3, 0.2]);
    }
    aggregateOutputs(graph, manualState, manualOutputs);

    const combinedState = createModelState(graph);
    const combinedScratch = createStepScratch(graph);
    const combinedOutputs = createOutputBuffer(graph);
    runSubsteps(graph, combinedState, combinedScratch, [0.3, 0.2], 3, combinedOutputs);

    expect(Array.from(combinedState.rate)).toEqual(Array.from(manualState.rate));
    expect(Array.from(combinedOutputs)).toEqual(Array.from(manualOutputs));
  });
});
