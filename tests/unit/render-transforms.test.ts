import { describe, expect, it } from 'vitest';
import type { AgentId, FoodState, RenderAgentSnapshot, RenderHazardSnapshot } from '../../src/lib/arena/types';
import {
  agentTransform,
  detectFoodPickups,
  detectHazardContacts,
  pushTrailPoint,
  type FoodRespawnRecord
} from '../../src/lib/render/transforms';

describe('pushTrailPoint', () => {
  it('fills a preallocated buffer from the end and reports the growing count', () => {
    const capacity = 3;
    const positions = new Float32Array(capacity * 3);
    let filled = 0;

    filled = pushTrailPoint(positions, capacity, filled, { x: 1, y: 0, z: 1 });
    expect(filled).toBe(1);
    filled = pushTrailPoint(positions, capacity, filled, { x: 2, y: 0, z: 2 });
    filled = pushTrailPoint(positions, capacity, filled, { x: 3, y: 0, z: 3 });
    expect(filled).toBe(3);
    expect(Array.from(positions)).toEqual([1, 0, 1, 2, 0, 2, 3, 0, 3]);
  });

  it('shifts the oldest sample out once the buffer is full, without reallocating', () => {
    const capacity = 3;
    const positions = new Float32Array(capacity * 3);
    let filled = 0;
    for (const point of [
      { x: 1, y: 0, z: 1 },
      { x: 2, y: 0, z: 2 },
      { x: 3, y: 0, z: 3 }
    ]) {
      filled = pushTrailPoint(positions, capacity, filled, point);
    }
    const buffer = positions.buffer;

    filled = pushTrailPoint(positions, capacity, filled, { x: 4, y: 0, z: 4 });

    expect(filled).toBe(3);
    expect(positions.buffer).toBe(buffer);
    expect(Array.from(positions)).toEqual([2, 0, 2, 3, 0, 3, 4, 0, 4]);
  });

  it('keeps only the newest point at capacity 1', () => {
    const positions = new Float32Array(3);
    let filled = 0;
    filled = pushTrailPoint(positions, 1, filled, { x: 1, y: 0, z: 1 });
    expect(filled).toBe(1);
    filled = pushTrailPoint(positions, 1, filled, { x: 2, y: 0, z: 2 });
    expect(filled).toBe(1);
    expect(Array.from(positions)).toEqual([2, 0, 2]);
  });

  it('is a no-op for zero capacity', () => {
    const positions = new Float32Array(0);
    expect(pushTrailPoint(positions, 0, 0, { x: 1, y: 1, z: 1 })).toBe(0);
  });
});

describe('agentTransform', () => {
  it('lifts the agent above the floor and passes heading through as rotationY unchanged', () => {
    const agent: RenderAgentSnapshot = { id: 'left', position: { x: 1.5, z: -2.5 }, heading: 0.7 };
    const transform = agentTransform(agent, 0.35);
    expect(transform.position).toEqual({ x: 1.5, y: 0.35, z: -2.5 });
    expect(transform.rotationY).toBe(0.7);
  });
});

const food = (id: string, respawns: number, x = 0, z = 0): FoodState => ({
  id,
  position: { x, z },
  radius: 0.25,
  respawns
});

/** Build the `id -> {respawns, position}` map `detectFoodPickups` expects as `previous`. */
const respawnMap = (entries: readonly FoodState[]): Map<string, FoodRespawnRecord> =>
  new Map(entries.map((entry) => [entry.id, { respawns: entry.respawns, position: { ...entry.position } }]));

describe('detectFoodPickups', () => {
  it('emits nothing on the first frame, when there is no previous record', () => {
    expect(detectFoodPickups(undefined, [food('food-0', 0)])).toEqual([]);
  });

  it('emits an event at the pre-respawn position when the respawn counter increases', () => {
    const previous = respawnMap([food('food-0', 0, 1, 1)]);
    const current = [food('food-0', 1, 5, 5)];
    expect(detectFoodPickups(previous, current)).toEqual([{ id: 'food-0', position: { x: 1, z: 1 } }]);
  });

  it('emits nothing when the respawn counter is unchanged', () => {
    const previous = respawnMap([food('food-0', 2, 3, 3)]);
    const current = [food('food-0', 2, 3, 3)];
    expect(detectFoodPickups(previous, current)).toEqual([]);
  });

  it('handles multiple foods independently', () => {
    const previous = respawnMap([food('food-0', 0), food('food-1', 4)]);
    const current = [food('food-0', 1), food('food-1', 4)];
    expect(detectFoodPickups(previous, current)).toEqual([{ id: 'food-0', position: { x: 0, z: 0 } }]);
  });

  it('ignores a food id absent from the previous record, as happens when a pool grows', () => {
    const previous = respawnMap([food('food-0', 0)]);
    const current = [food('food-0', 0), food('food-1', 3)];
    expect(detectFoodPickups(previous, current)).toEqual([]);
  });

  it('emits a single event when a respawn counter jumps by more than one in a single call', () => {
    // A multi-tick catch-up step (see App.svelte's accumulator) can collect
    // the same food twice before a render happens. This is an accepted,
    // documented limitation for a decorative effect: only the position
    // before the *first* of those pickups is reported.
    const previous = respawnMap([food('food-0', 0, 2, 2)]);
    const current = [food('food-0', 2, 9, 9)];
    expect(detectFoodPickups(previous, current)).toEqual([{ id: 'food-0', position: { x: 2, z: 2 } }]);
  });
});

const agent = (id: AgentId, x: number, z: number): RenderAgentSnapshot => ({
  id,
  position: { x, z },
  heading: 0
});

const hazard = (id: string, x: number, z: number, radius = 0.6): RenderHazardSnapshot => ({
  id,
  position: { x, z },
  radius
});

describe('detectHazardContacts', () => {
  it('reports a rising-edge event the instant an agent enters a hazard radius', () => {
    const agents = [agent('left', 0, 0)];
    const hazards = [hazard('hazard-0', 0.5, 0)];

    const result = detectHazardContacts(agents, hazards, 0.35, new Set());

    expect(result.events).toEqual([{ agentId: 'left', hazardId: 'hazard-0', position: { x: 0.5, z: 0 } }]);
    expect(result.touching.has('left:hazard-0')).toBe(true);
  });

  it('does not re-emit while the pair remains in contact across calls', () => {
    const agents = [agent('left', 0, 0)];
    const hazards = [hazard('hazard-0', 0.5, 0)];

    const first = detectHazardContacts(agents, hazards, 0.35, new Set());
    const second = detectHazardContacts(agents, hazards, 0.35, first.touching);

    expect(second.events).toEqual([]);
  });

  it('clears the touching set once a pair separates, allowing a future re-trigger', () => {
    const farHazards = [hazard('hazard-0', 10, 10)];
    const agents = [agent('left', 0, 0)];

    const separated = detectHazardContacts(agents, farHazards, 0.35, new Set(['left:hazard-0']));
    expect(separated.touching.size).toBe(0);
    expect(separated.events).toEqual([]);

    const closeHazards = [hazard('hazard-0', 0, 0)];
    const retriggered = detectHazardContacts(agents, closeHazards, 0.35, separated.touching);
    expect(retriggered.events).toEqual([{ agentId: 'left', hazardId: 'hazard-0', position: { x: 0, z: 0 } }]);
  });

  it('does not report a pair outside the combined radius', () => {
    const agents = [agent('left', 0, 0)];
    const hazards = [hazard('hazard-0', 5, 5)];

    const result = detectHazardContacts(agents, hazards, 0.35, new Set());

    expect(result.events).toEqual([]);
    expect(result.touching.size).toBe(0);
  });

  it('registers exact tangency (distance equal to the combined radius) as a contact', () => {
    const agents = [agent('left', 0, 0)];
    const hazards = [hazard('hazard-0', 1, 0, 0.65)]; // 1 === 0.35 + 0.65

    const result = detectHazardContacts(agents, hazards, 0.35, new Set());

    expect(result.events).toEqual([{ agentId: 'left', hazardId: 'hazard-0', position: { x: 1, z: 0 } }]);
  });

  it('fails closed on a non-finite hazard position instead of registering a contact', () => {
    const agents = [agent('left', 0, 0)];
    const hazards = [hazard('hazard-0', Number.NaN, 0)];

    const result = detectHazardContacts(agents, hazards, 0.35, new Set());

    expect(result.events).toEqual([]);
    expect(result.touching.size).toBe(0);
  });

  it('tracks independent pairs for two agents against the same hazard', () => {
    const agents = [agent('left', 0, 0), agent('right', 0.5, 0)];
    const hazards = [hazard('hazard-0', 0, 0, 1)];

    const result = detectHazardContacts(agents, hazards, 0.35, new Set());

    expect(result.touching.has('left:hazard-0')).toBe(true);
    expect(result.touching.has('right:hazard-0')).toBe(true);
    expect(result.events).toHaveLength(2);
  });
});
