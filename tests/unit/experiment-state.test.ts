import { describe, expect, it } from 'vitest';
import {
  canPause,
  canReset,
  canResume,
  canStart,
  transition,
  type ExperimentStatus
} from '../../src/lib/experiment/state';

describe('experiment state machine', () => {
  it('moves loading -> ready on assetsReady, and loading -> error on assetsFailed', () => {
    expect(transition('loading', { type: 'assetsReady' })).toBe('ready');
    expect(transition('loading', { type: 'assetsFailed' })).toBe('error');
  });

  it('moves ready -> running on start', () => {
    expect(transition('ready', { type: 'start' })).toBe('running');
  });

  it('moves running -> paused on pause, and paused -> running on resume', () => {
    expect(transition('running', { type: 'pause' })).toBe('paused');
    expect(transition('paused', { type: 'resume' })).toBe('running');
  });

  it('moves every live (non-loading) state to error on runtimeError', () => {
    // Not just running/paused: a failed topology switch or a failed
    // post-reset rebind can happen while sitting at ready/finished between
    // runs (see ExperimentRunner#fail()), and must still be reportable.
    for (const state of ['ready', 'running', 'paused', 'finished'] as const) {
      expect(transition(state, { type: 'runtimeError' })).toBe('error');
    }
  });

  it('loading does not accept runtimeError (only assetsFailed reaches error from loading)', () => {
    expect(transition('loading', { type: 'runtimeError' })).toBe('loading');
  });

  it('reset returns to ready from ready, running, paused, and finished', () => {
    for (const state of ['ready', 'running', 'paused', 'finished'] as const) {
      expect(transition(state, { type: 'reset' })).toBe('ready');
    }
  });

  it('reset from loading or error is a no-op (no valid graph to reset into)', () => {
    expect(transition('loading', { type: 'reset' })).toBe('loading');
    expect(transition('error', { type: 'reset' })).toBe('error');
  });

  it('tickCompleted only finishes a running state, and only once tick >= totalTicks', () => {
    expect(transition('running', { type: 'tickCompleted', tick: 50, totalTicks: 2700 })).toBe('running');
    expect(transition('running', { type: 'tickCompleted', tick: 2700, totalTicks: 2700 })).toBe('finished');
    expect(transition('running', { type: 'tickCompleted', tick: 2701, totalTicks: 2700 })).toBe('finished');
  });

  it('a stray tickCompleted arriving after a race (state no longer running) is ignored', () => {
    for (const state of ['ready', 'paused', 'finished', 'error', 'loading'] as const) {
      expect(transition(state, { type: 'tickCompleted', tick: 2700, totalTicks: 2700 })).toBe(state);
    }
  });

  it('an event with no listed transition for the current state is a no-op', () => {
    expect(transition('ready', { type: 'pause' })).toBe('ready');
    expect(transition('finished', { type: 'start' })).toBe('finished');
    expect(transition('error', { type: 'start' })).toBe('error');
  });

  it('canStart/canPause/canResume/canReset guards match the transition table', () => {
    const all: readonly ExperimentStatus[] = ['loading', 'ready', 'running', 'paused', 'finished', 'error'];
    for (const state of all) {
      expect(canStart(state)).toBe(state === 'ready');
      expect(canPause(state)).toBe(state === 'running');
      expect(canResume(state)).toBe(state === 'paused');
      expect(canReset(state)).toBe(state === 'ready' || state === 'running' || state === 'paused' || state === 'finished');
    }
  });
});
