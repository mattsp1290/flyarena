import { describe, expect, it } from 'vitest';
import { ARENA_CONFIG } from '../../src/lib/arena/config';
import { createSnapshot, createWorld, stepWorld } from '../../src/lib/arena/world';

describe('renderer snapshots', () => {
  it('interpolates previous/current transforms and clamps render alpha without mutating state', () => {
    const world = createWorld(4);
    const agent = world.agents[0];
    agent.previousPosition = { x: -4, z: 1 };
    agent.position = { x: 2, z: 5 };
    agent.previousHeading = Math.PI * 0.95;
    agent.heading = -Math.PI * 0.95;
    const before = JSON.stringify(world);

    const halfway = createSnapshot(world, 0.5);
    expect(halfway.agents[0].position).toEqual({ x: -1, z: 3 });
    expect(Math.abs(halfway.agents[0].heading)).toBeCloseTo(Math.PI);
    expect(createSnapshot(world, -2).agents[0].position).toEqual(agent.previousPosition);
    expect(createSnapshot(world, 9).agents[0].position).toEqual(agent.position);
    expect(JSON.stringify(world)).toBe(before);

    halfway.agents[0].position.x = 999;
    expect(world.agents[0].position.x).toBe(2);
  });

  it('normalizes both heading endpoints and takes the shortest path across the pi seam', () => {
    const world = createWorld(40);
    world.agents[0].previousHeading = Math.PI * 0.95;
    world.agents[0].heading = -Math.PI * 0.95;

    expect(createSnapshot(world, 0).agents[0].heading).toBeCloseTo(Math.PI * 0.95, 12);
    expect(createSnapshot(world, 1).agents[0].heading).toBeCloseTo(-Math.PI * 0.95, 12);
    expect(Math.abs(createSnapshot(world, 0.5).agents[0].heading)).toBeCloseTo(Math.PI, 12);
  });

  it('uses the explicit simulation config when interpolating snapshot time', () => {
    const customConfig = { ...ARENA_CONFIG, fixedDeltaSeconds: 0.2 };
    const world = stepWorld(createWorld(6, customConfig), {}, customConfig);

    expect(world.timeSeconds).toBeCloseTo(0.2, 12);
    expect(createSnapshot(world, 0.25, customConfig).timeSeconds).toBeCloseTo(0.05, 12);
  });

  it('uses retained world timing when no config is threaded and rejects a mismatched override', () => {
    const customConfig = { ...ARENA_CONFIG, fixedDeltaSeconds: 0.2 };
    const world = stepWorld(createWorld(6, customConfig));

    expect(createSnapshot(world, 0.25).timeSeconds).toBeCloseTo(0.05, 12);
    expect(() => createSnapshot(world, 1, ARENA_CONFIG)).toThrow(/config.*match/i);
  });

  it('deeply detaches food and hazard positions from simulation state', () => {
    const world = createWorld(5);
    const foodPosition = { ...world.foods[0].position };
    const hazardPosition = { ...world.hazards[0].position };
    const snapshot = createSnapshot(world, 1);

    snapshot.foods[0].position.x = 999;
    snapshot.foods[0].position.z = 998;
    snapshot.hazards[0].position.x = 997;
    snapshot.hazards[0].position.z = 996;

    expect(world.foods[0].position).toEqual(foodPosition);
    expect(world.hazards[0].position).toEqual(hazardPosition);
  });
});
