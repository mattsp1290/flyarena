import { describe, expect, it } from 'vitest';
import {
  MAX_REWIRED_SEED,
  MAX_SEED,
  NEURON_COUNT,
  parseNeuronList,
  parseNeuronToken,
  validateAtlas,
  validateGraphId,
  validateLesion,
  validateSwapset,
  validateUniqueIndices
} from '../../src/lib/graphlab/forms';

describe('validateGraphId', () => {
  it('accepts every closed-set member', () => {
    expect(validateGraphId('biological')).toBeNull();
    expect(validateGraphId('disconnected')).toBeNull();
    expect(validateGraphId('rewired:0')).toBeNull();
    expect(validateGraphId(`rewired:${MAX_REWIRED_SEED}`)).toBeNull();
  });
  it('rejects a rewired seed over the maximum', () => {
    expect(validateGraphId(`rewired:${MAX_REWIRED_SEED + 1}`)).not.toBeNull();
    expect(validateGraphId('rewired:500')).not.toBeNull();
  });
  it('rejects a shell-metacharacter payload and path traversal', () => {
    expect(validateGraphId('rewired:1;rm -rf /')).not.toBeNull();
    expect(validateGraphId('../x')).not.toBeNull();
  });
  it('rejects a leading zero', () => {
    expect(validateGraphId('rewired:007')).not.toBeNull();
  });
  it('rejects a trailing newline', () => {
    expect(validateGraphId('biological\n')).not.toBeNull();
    expect(validateGraphId('rewired:5\n')).not.toBeNull();
  });
  it('rejects an unrecognized string', () => {
    expect(validateGraphId('nonsense')).not.toBeNull();
    expect(validateGraphId('')).not.toBeNull();
  });
});

describe('validateUniqueIndices', () => {
  it('rejects entries outside [0, NEURON_COUNT)', () => {
    const errors = validateUniqueIndices([NEURON_COUNT], { minLen: 1, maxLen: 64, label: 'set' });
    expect(errors.length).toBeGreaterThan(0);
  });
  it('rejects duplicate entries', () => {
    const errors = validateUniqueIndices([1, 1], { minLen: 1, maxLen: 64, label: 'set' });
    expect(errors.some((e) => e.includes('unique'))).toBe(true);
  });
  it('accepts a valid set', () => {
    expect(validateUniqueIndices([0, 1, NEURON_COUNT - 1], { minLen: 1, maxLen: 64, label: 'set' })).toEqual([]);
  });
});

describe('validateLesion', () => {
  const base = { graph: 'biological', sets: [[7]], seedStart: 30001, seedCount: 4, ticks: 300 };
  it('accepts a minimal valid request', () => {
    expect(validateLesion(base)).toEqual([]);
  });
  it('rejects too many sets', () => {
    const sets = Array.from({ length: 33 }, (_, i) => [i]);
    expect(validateLesion({ ...base, sets })).not.toHaveLength(0);
  });
  it('rejects seedCount out of bounds', () => {
    expect(validateLesion({ ...base, seedCount: 3 })).not.toHaveLength(0);
    expect(validateLesion({ ...base, seedCount: 101 })).not.toHaveLength(0);
  });
  it('rejects ticks out of bounds', () => {
    expect(validateLesion({ ...base, ticks: 299 })).not.toHaveLength(0);
    expect(validateLesion({ ...base, ticks: 1801 })).not.toHaveLength(0);
  });
  it('rejects a seed range that overflows the uint32 maximum', () => {
    expect(validateLesion({ ...base, seedStart: MAX_SEED - 1, seedCount: 4 })).not.toHaveLength(0);
  });
  it('accepts a seed range that exactly fits', () => {
    expect(validateLesion({ ...base, seedStart: MAX_SEED - 3, seedCount: 4 })).toEqual([]);
  });
});

describe('validateAtlas', () => {
  const base = { graph: 'biological', searchSeed: 1729, population: 8, generations: 2, ticks: 300 };
  it('accepts a minimal valid request', () => {
    expect(validateAtlas(base)).toEqual([]);
  });
  it('rejects population/generations/ticks out of bounds', () => {
    expect(validateAtlas({ ...base, population: 3 })).not.toHaveLength(0);
    expect(validateAtlas({ ...base, population: 65 })).not.toHaveLength(0);
    expect(validateAtlas({ ...base, generations: 0 })).not.toHaveLength(0);
    expect(validateAtlas({ ...base, generations: 49 })).not.toHaveLength(0);
    expect(validateAtlas({ ...base, ticks: 299 })).not.toHaveLength(0);
    expect(validateAtlas({ ...base, ticks: 901 })).not.toHaveLength(0);
  });
  it('rejects an invalid graph id', () => {
    expect(validateAtlas({ ...base, graph: 'rewired:500' })).not.toHaveLength(0);
  });
});

describe('validateSwapset', () => {
  const base = { swaps: [{ a: 0, b: 1, c: 2, d: 3 }], controls: 2, seedStart: 30001, seedCount: 4, ticks: 300 };
  it('accepts a minimal valid request', () => {
    expect(validateSwapset(base)).toEqual([]);
  });
  it('rejects too many swaps', () => {
    const swaps = Array.from({ length: 51 }, () => ({ a: 0, b: 1, c: 2, d: 3 }));
    expect(validateSwapset({ ...base, swaps })).not.toHaveLength(0);
  });
  it('rejects a swap field out of range', () => {
    expect(validateSwapset({ ...base, swaps: [{ a: NEURON_COUNT, b: 1, c: 2, d: 3 }] })).not.toHaveLength(0);
  });
  it('rejects controls out of bounds', () => {
    expect(validateSwapset({ ...base, controls: 101 })).not.toHaveLength(0);
    expect(validateSwapset({ ...base, controls: -1 })).not.toHaveLength(0);
  });
});

describe('parseNeuronToken / parseNeuronList', () => {
  it('parses a raw index', () => {
    expect(parseNeuronToken('7')).toBe(7);
  });
  it('rejects an out-of-range raw index with no body-id list', () => {
    expect(() => parseNeuronToken(String(NEURON_COUNT))).toThrow();
  });
  it('resolves a body id via the supplied lookup', () => {
    const bodyIds = ['10010', '10030', '10038'];
    expect(parseNeuronToken('10038', bodyIds)).toBe(2);
  });
  it('throws naming the offending token when neither an index nor a body id matches', () => {
    expect(() => parseNeuronToken('not-a-neuron', ['10010'])).toThrow(/not-a-neuron/);
  });
  it('parses a comma/whitespace-separated list into sorted, unique indices', () => {
    expect(parseNeuronList(' 5, 3  5\n1')).toEqual([1, 3, 5]);
  });
  it('resolves a mixed list of indices and body ids', () => {
    const bodyIds = ['10010', '10030', '10038'];
    expect(parseNeuronList('0, 10038', bodyIds)).toEqual([0, 2]);
  });
});
