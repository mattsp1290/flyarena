import { describe, expect, it } from 'vitest';
import {
  ARENA_CONFIG,
  ARENA_CONFIG_LIMITS,
  retainArenaConfig
} from '../../src/lib/arena/config';
import { createReplaySummary } from '../../src/lib/arena/replay';
import { observeAgent } from '../../src/lib/arena/sensors';
import type { WorldState } from '../../src/lib/arena/types';
import { createSnapshot, createWorld, nextRandomState, stepWorld } from '../../src/lib/arena/world';

const overlaps = (
  a: { position: { x: number; z: number }; radius: number },
  b: { position: { x: number; z: number }; radius: number }
) => Math.hypot(a.position.x - b.position.x, a.position.z - b.position.z) <= a.radius + b.radius;

describe('arena world reset', () => {
  it('creates deterministic seeded plain-data worlds with mirrored agents', () => {
    const first = createWorld(0x1234abcd);
    const second = createWorld(0x1234abcd);
    const other = createWorld(0x1234abce);

    expect(first).toEqual(second);
    expect(first.foods).not.toEqual(other.foods);
    expect(first.tick).toBe(0);
    expect(ARENA_CONFIG.fixedDeltaSeconds).toBeCloseTo(1 / 30, 12);
    expect(first.agents).toHaveLength(2);
    expect(first.agents[0].position.x).toBe(-first.agents[1].position.x);
    expect(first.agents[0].position.z).toBe(first.agents[1].position.z);
    expect(first.agents[0].heading).toBe(-first.agents[1].heading);
    expect(() => JSON.stringify(first)).not.toThrow();
  });

  it.each([7, 25])('places every entity without overlap for regression seed %i', (seed) => {
    const world = createWorld(seed);
    const entities = [...world.agents, ...world.foods, ...world.hazards];

    for (let left = 0; left < entities.length; left += 1) {
      for (let right = left + 1; right < entities.length; right += 1) {
        expect(overlaps(entities[left], entities[right]), `${entities[left].id}/${entities[right].id}`).toBe(false);
      }
    }
    const idle = stepWorld(world);
    expect(idle.agents.map(({ score }) => score.foodPickups)).toEqual([0, 0]);
    expect(idle.agents.map(({ score }) => score.hazardContacts)).toEqual([0, 0]);
  });

  it('completes deterministic bounded placement for a constrained valid config', () => {
    const config = {
      ...ARENA_CONFIG,
      halfWidth: 2,
      halfDepth: 1.4,
      agentRadius: 0.2,
      foodRadius: 0.18,
      hazardRadius: 0.24,
      foodCount: 24,
      hazardCount: 6,
      spawnInset: 0.05
    };
    const first = createWorld(3, config);
    const second = createWorld(3, config);
    const entities = [...first.agents, ...first.foods, ...first.hazards];

    expect(first).toEqual(second);
    for (let left = 0; left < entities.length; left += 1) {
      for (let right = left + 1; right < entities.length; right += 1) {
        expect(overlaps(entities[left], entities[right])).toBe(false);
      }
    }
  });

  it('normalizes the exported PRNG zero state to the same deterministic non-zero state as reset', () => {
    expect(nextRandomState(0)).toBe(nextRandomState(createWorld(0).seed));
    expect(nextRandomState(0)).not.toBe(0);
  });

  it('validates and retains a detached immutable custom config for all subsequent operations', () => {
    const supplied = {
      ...ARENA_CONFIG,
      fixedDeltaSeconds: 0.2,
      halfWidth: 20,
      halfDepth: 10,
      foodCount: 1,
      hazardCount: 1,
      sensorRange: 40,
      maxSpeed: 3
    };
    const world = createWorld(41, supplied);
    supplied.fixedDeltaSeconds = 0.5;

    expect(world.config).not.toBe(supplied);
    expect(Object.isFrozen(world.config)).toBe(true);
    expect(stepWorld(world).timeSeconds).toBeCloseTo(0.2, 12);
    expect(() => stepWorld(world, {}, { ...world.config, fixedDeltaSeconds: 0.1 })).toThrow(
      /config.*match/i
    );
  });

  it('rejects retained-config replacement or mutation before every world-consuming operation', () => {
    const original = createWorld(42);
    if (false) {
      // @ts-expect-error WorldState.config is intentionally readonly.
      original.config = ARENA_CONFIG;
    }
    const escaped = original as WorldState & { config: Record<string, number> };
    escaped.config = { ...original.config, maxSpeed: original.config.maxSpeed + 1 };

    for (const operation of [
      () => stepWorld(escaped),
      () => observeAgent(escaped, 'left'),
      () => createSnapshot(escaped, 1),
      () => createReplaySummary(escaped)
    ]) {
      expect(operation).toThrow(/retained arena config/i);
    }
  });

  it('accepts an equivalent detached retained config and rejects deserialized-like malformed identity', () => {
    const detached = JSON.parse(JSON.stringify(createWorld(43))) as WorldState;
    expect(stepWorld(detached).tick).toBe(1);

    const malformed = JSON.parse(JSON.stringify(createWorld(43))) as WorldState;
    (malformed as WorldState & { configFingerprint?: string }).configFingerprint = 'forged';
    expect(() => stepWorld(malformed)).toThrow(/retained arena config/i);
  });

  it.each([
    ['finite positive dimensions', { halfWidth: Number.NaN }],
    ['positive radii', { foodRadius: 0 }],
    ['positive timestep', { fixedDeltaSeconds: -1 }],
    ['positive sensor range', { sensorRange: 0 }],
    ['positive max speed', { maxSpeed: Number.POSITIVE_INFINITY }],
    ['integral entity counts', { foodCount: 1.5 }],
    ['non-negative entity counts', { hazardCount: -1 }],
    ['feasible geometry', { halfWidth: 1, agentRadius: 0.4 }]
  ])('rejects invalid config: %s', (_label, change) => {
    expect(() => createWorld(1, { ...ARENA_CONFIG, ...change })).toThrow(/arena config/i);
  });

  it.each([
    ['Number.MAX_VALUE timestep', { fixedDeltaSeconds: Number.MAX_VALUE }],
    [
      'overflowing acceleration product',
      { fixedDeltaSeconds: Number.MAX_VALUE / 2, acceleration: 3 }
    ],
    ['impractical arena extent', { halfWidth: Number.MAX_VALUE }],
    ['impractical speed', { maxSpeed: Number.MAX_VALUE }],
    ['impractical turn rate', { turnRate: Number.MAX_VALUE }],
    ['impractical drag', { brakeDrag: Number.MAX_VALUE }],
    ['impractical scoring rate', { movementScorePerUnit: Number.MAX_VALUE }],
    ['impractical sensor range', { sensorRange: Number.MAX_VALUE }]
  ])('rejects finite-but-unsafe config arithmetic: %s', (_label, change) => {
    expect(() => createWorld(1, { ...ARENA_CONFIG, ...change })).toThrow(/arena config/i);
  });

  it('accepts the documented practical entity-count boundaries', () => {
    expect(
      retainArenaConfig({
        ...ARENA_CONFIG,
        halfWidth: 100,
        halfDepth: 100,
        foodRadius: 0.01,
        hazardRadius: 0.01,
        foodCount: ARENA_CONFIG_LIMITS.maxFoodCount,
        hazardCount: ARENA_CONFIG_LIMITS.maxHazardCount
      })
    ).toMatchObject({
      foodCount: ARENA_CONFIG_LIMITS.maxFoodCount,
      hazardCount: ARENA_CONFIG_LIMITS.maxHazardCount
    });
  });

  it.each([
    ['food', { foodCount: ARENA_CONFIG_LIMITS.maxFoodCount + 1 }],
    ['hazard', { hazardCount: ARENA_CONFIG_LIMITS.maxHazardCount + 1 }]
  ])('rejects an above-limit %s count before world allocation', (_label, change) => {
    expect(() =>
      createWorld(1, {
        ...ARENA_CONFIG,
        foodRadius: Number.MIN_VALUE,
        hazardRadius: Number.MIN_VALUE,
        ...change
      })
    ).toThrow(/arena config.*count.*limit/i);
  });
});

describe('fixed-step world evolution', () => {
  it.each([
    ['unsafe tick', (world: WorldState) => {
      world.tick = Number.MAX_SAFE_INTEGER;
      world.timeSeconds = world.tick * world.config.fixedDeltaSeconds;
    }],
    ['non-finite next clock', (world: WorldState) => {
      world.timeSeconds = Number.MAX_VALUE;
    }],
    ['overflow-prone velocity', (world: WorldState) => {
      world.agents[0].velocity.x = Number.MAX_VALUE;
    }]
  ])('rejects %s before mutating or stepping state', (_label, corrupt) => {
    const world = createWorld(89);
    corrupt(world);
    const before = JSON.stringify(world);
    expect(() => stepWorld(world, { left: [1, 1, 0] })).toThrow(/world (clock|state)/i);
    expect(JSON.stringify(world)).toBe(before);
  });

  it('moves seeded hazards by velocity times the fixed delta when unobstructed', () => {
    const world = createWorld(0x51eed);
    const next = stepWorld(world);

    expect(next.hazards.some((hazard, index) => hazard.position.x !== world.hazards[index].position.x || hazard.position.z !== world.hazards[index].position.z)).toBe(true);
    next.hazards.forEach((hazard, index) => {
      const previous = world.hazards[index];
      expect(hazard.position.x).toBeCloseTo(
        previous.position.x + previous.velocity.x * ARENA_CONFIG.fixedDeltaSeconds,
        12
      );
      expect(hazard.position.z).toBeCloseTo(
        previous.position.z + previous.velocity.z * ARENA_CONFIG.fixedDeltaSeconds,
        12
      );
      expect(hazard.velocity).toEqual(previous.velocity);
    });
  });

  it('clamps hazards at negative-z and positive-x walls and reflects normal velocities', () => {
    const world = createWorld(0x51eed);
    const hazard = world.hazards[0];
    const maxX = ARENA_CONFIG.halfWidth - hazard.radius;
    const minZ = -ARENA_CONFIG.halfDepth + hazard.radius;
    hazard.position = { x: maxX - 0.01, z: minZ + 0.01 };
    hazard.velocity = { x: 2, z: -3 };

    const next = stepWorld(world);

    expect(next.hazards[0].position).toEqual({ x: maxX, z: minZ });
    expect(next.hazards[0].velocity).toEqual({ x: -2, z: 3 });
    expect(next.hazards[0].previousPosition).toEqual(hazard.position);
  });

  it('integrates actions at 30 Hz, scores movement, and clamps wall collisions', () => {
    let world = createWorld(9);
    const original = world;

    world = stepWorld(world, { left: [1, 0, 0], right: [0, 0, 0] });
    expect(world.tick).toBe(1);
    expect(world.timeSeconds).toBeCloseTo(1 / 30, 12);
    expect(world.agents[0].position.x).toBeGreaterThan(original.agents[0].position.x);
    expect(world.agents[0].score.distanceTravelled).toBeGreaterThan(0);
    expect(original.tick).toBe(0);

    world.agents[0].position.x = ARENA_CONFIG.halfWidth - world.agents[0].radius;
    world.agents[0].heading = Math.PI / 2;
    world.agents[0].velocity = { x: ARENA_CONFIG.maxSpeed, z: 0 };
    const collided = stepWorld(world, { left: [1, 0, 0] });
    expect(collided.agents[0].position.x).toBeLessThanOrEqual(
      ARENA_CONFIG.halfWidth - collided.agents[0].radius
    );
    expect(collided.agents[0].velocity.x).toBe(0);
  });

  it.each([
    { wall: 'negative x', position: { x: -11.65, z: 0 }, velocity: { x: -2, z: 0 } },
    { wall: 'positive z', position: { x: 0, z: 7.65 }, velocity: { x: 0, z: 2 } },
    { wall: 'negative z', position: { x: 0, z: -7.65 }, velocity: { x: 0, z: -2 } }
  ])('clamps agents and zeros outward normal velocity at the $wall wall', ({ position, velocity }) => {
    const world = createWorld(90);
    world.agents[0].position = position;
    world.agents[0].velocity = velocity;

    const next = stepWorld(world);

    expect(next.agents[0].position).toEqual(position);
    expect(next.agents[0].velocity).toEqual({ x: 0, z: 0 });
  });

  it('scores controlled movement as distance travelled times the configured rate', () => {
    const world = createWorld(91);
    world.foods = [];
    world.hazards = [];
    const moved = stepWorld(world, { left: [1, 0, 0] });
    const score = moved.agents[0].score;

    expect(score.distanceTravelled).toBeGreaterThan(0);
    expect(score.foodPickups).toBe(0);
    expect(score.hazardContacts).toBe(0);
    expect(score.movementScore).toBe(
      score.distanceTravelled * ARENA_CONFIG.movementScorePerUnit
    );
  });

  it('picks up and deterministically respawns food and counts hazard contact edges', () => {
    const world = createWorld(77);
    const agent = world.agents[0];
    world.foods[0].position = { ...agent.position };
    world.hazards[0].position = { ...agent.position };
    world.hazards[0].velocity = { x: 0, z: 0 };

    const first = stepWorld(world);
    expect(first.agents[0].score.foodPickups).toBe(1);
    expect(first.foods[0].respawns).toBe(1);
    expect(first.foods[0].position).not.toEqual(world.foods[0].position);
    expect(first.agents[0].score.hazardContacts).toBe(1);
    expect(first.agents[0].score.movementScore).toBe(
      ARENA_CONFIG.foodScore - ARENA_CONFIG.hazardPenalty
    );

    const second = stepWorld(first);
    expect(second.agents[0].score.hazardContacts).toBe(1);
    expect(second).toEqual(stepWorld(first));
  });

  it('respawns collected food away from agents and every other entity', () => {
    const world = createWorld(88);
    world.foods[0].position = { ...world.agents[0].position };
    world.rngState = 5386; // The old first candidate overlaps the left agent.

    const respawned = stepWorld(world);
    const food = respawned.foods[0];
    const blockers = [
      ...respawned.agents,
      ...respawned.foods.filter(({ id }) => id !== food.id),
      ...respawned.hazards
    ];

    expect(food.respawns).toBe(1);
    expect(blockers.every((blocker) => !overlaps(food, blocker))).toBe(true);
    expect(stepWorld(respawned).agents[0].score.foodPickups).toBe(1);
  });
});
