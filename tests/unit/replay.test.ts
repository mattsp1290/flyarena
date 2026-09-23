import { describe, expect, it } from 'vitest';
import { ARENA_CONFIG } from '../../src/lib/arena/config';
import {
  createReplaySummary,
  hashReplaySummary,
  runReplay,
  serializeReplaySummary
} from '../../src/lib/arena/replay';
import { createSnapshot, createWorld, stepWorld } from '../../src/lib/arena/world';
import type { ActionsByAgent } from '../../src/lib/arena/types';

const policy = (tick: number): ActionsByAgent => ({
  left: [Math.sin(tick * 0.013), Math.cos(tick * 0.007), tick % 97 === 0 ? 0.4 : 0],
  right: [Math.cos(tick * 0.011), -Math.sin(tick * 0.005), tick % 131 === 0 ? 0.2 : 0]
});

const replayWithRenderSchedule = (seed: number, alphas: readonly number[]) => {
  let world = createWorld(seed);
  for (let tick = 0; tick < 2_700; tick += 1) {
    world = stepWorld(world, policy(tick));
    for (const alpha of alphas) createSnapshot(world, alpha);
  }
  return createReplaySummary(world);
};

describe('stable replay serialization', () => {
  it('produces byte-equivalent summaries and hashes for 2,700 ticks independent of rendering', () => {
    const first = runReplay(0xdecafbad, 2_700, policy);
    const second = runReplay(0xdecafbad, 2_700, policy);
    const sparseRender = replayWithRenderSchedule(0xdecafbad, [0.25]);
    const busyRender = replayWithRenderSchedule(0xdecafbad, [0, 0.1, 0.5, 0.9, 1]);

    expect(first.serialized).toBe(second.serialized);
    expect(first.hash).toBe(second.hash);
    expect(serializeReplaySummary(sparseRender)).toBe(serializeReplaySummary(busyRender));
    expect(serializeReplaySummary(sparseRender)).toBe(first.serialized);
    expect(first.hash).toMatch(/^[0-9a-f]{16}$/);
    expect(first.hash).toBe(hashReplaySummary(first.summary));
    expect(first.summary.ticks).toBe(2_700);
    expect(runReplay(0xdecafbac, 2_700, policy).hash).not.toBe(first.hash);
  });

  it('binds schema-v2 serialization and hashes to validated custom config semantics', () => {
    const custom = { ...ARENA_CONFIG, foodScore: ARENA_CONFIG.foodScore + 1 };
    const baseline = runReplay(55, 0);
    const configured = runReplay(55, 0, undefined, custom);

    expect(baseline.summary.schemaVersion).toBe(2);
    expect(configured.summary.configFingerprint).not.toBe(baseline.summary.configFingerprint);
    expect(configured.serialized).not.toBe(baseline.serialized);
    expect(configured.hash).not.toBe(baseline.hash);
    expect(configured).toEqual(runReplay(55, 0, undefined, custom));
  });

  it('canonicalizes genuinely reordered top-level and nested object keys', () => {
    const summary = createReplaySummary(createWorld(1));
    const reverseObjectKeys = (value: unknown): unknown => {
      if (Array.isArray(value)) return value.map(reverseObjectKeys);
      if (value !== null && typeof value === 'object') {
        return Object.fromEntries(
          Object.entries(value)
            .reverse()
            .map(([key, entry]) => [key, reverseObjectKeys(entry)])
        );
      }
      return value;
    };
    const reordered = reverseObjectKeys(summary) as typeof summary;

    expect(Object.keys(reordered)).toEqual(Object.keys(summary).reverse());
    expect(Object.keys(reordered.agents[0])).toEqual(Object.keys(summary.agents[0]).reverse());
    expect(JSON.stringify(reordered)).not.toBe(JSON.stringify(summary));
    expect(serializeReplaySummary(reordered)).toBe(serializeReplaySummary(summary));
    expect(hashReplaySummary(reordered)).toBe(hashReplaySummary(summary));
  });

  it('uses locale-independent UTF-16 key order and a known UTF-8 FNV-1a vector', () => {
    const vector = { ä: 1, a: 2, Z: 3 } as unknown as Parameters<
      typeof serializeReplaySummary
    >[0];

    expect(serializeReplaySummary(vector)).toBe('{"Z":3,"a":2,"ä":1}');
    expect(hashReplaySummary(vector)).toBe('ed251b85a07e2201');
  });

  it('deeply detaches nested replay data from its source world', () => {
    const world = stepWorld(createWorld(2), { left: [1, 0.5, 0] });
    const before = JSON.stringify(world);
    const summary = createReplaySummary(world);

    summary.agents[0].position.x = 100;
    summary.agents[0].velocity.z = 101;
    summary.agents[0].score.movementScore = 102;
    summary.agents[0].activeHazardIds.push('mutated');
    summary.foods[0].position.z = 103;
    summary.hazards[0].position.x = 104;
    summary.hazards[0].velocity.z = 105;

    expect(JSON.stringify(world)).toBe(before);
  });

  it('normalizes non-finite or negative replay tick inputs', () => {
    expect(runReplay(Number.NaN, -4).summary.ticks).toBe(0);
  });

  it('passes a deeply detached readonly view to policies so mutation cannot corrupt replay', () => {
    const baseline = runReplay(123, 3);
    const mutated = runReplay(123, 3, (_tick, world) => {
      // @ts-expect-error deliberate malicious/mistaken mutation regression
      world.agents[0].position.x = 999;
      // @ts-expect-error deliberate malicious/mistaken mutation regression
      world.foods.splice(0);
      return {};
    });

    expect(mutated).toEqual(baseline);
  });
});
