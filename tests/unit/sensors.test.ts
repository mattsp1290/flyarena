import { describe, expect, it } from 'vitest';
import { ARENA_CONFIG } from '../../src/lib/arena/config';
import { OBSERVATION_CHANNELS, observeAgent } from '../../src/lib/arena/sensors';
import { createWorld } from '../../src/lib/arena/world';

describe('egocentric observations', () => {
  it('emits exactly eight documented normalized channels without absolute coordinates', () => {
    const world = createWorld(12);
    const agent = world.agents[0];
    agent.position = { x: 0, z: 0 };
    agent.heading = 0;
    agent.velocity = { x: 0, z: ARENA_CONFIG.maxSpeed / 2 };
    world.foods = [{ id: 'food', position: { x: 0, z: 6 }, radius: 0.25, respawns: 0 }];
    world.hazards = [
      {
        id: 'hazard',
        position: { x: 6, z: 0 },
        previousPosition: { x: 6, z: 0 },
        velocity: { x: 0, z: 0 },
        radius: 0.6
      }
    ];

    const observation = observeAgent(world, 'left');
    expect(OBSERVATION_CHANNELS).toEqual([
      'foodBearing',
      'foodDistance',
      'hazardBearing',
      'hazardDistance',
      'forwardClearance',
      'leftClearance',
      'rightClearance',
      'speed'
    ]);
    expect(observation).toHaveLength(8);
    expect(observation[0]).toBeCloseTo(0);
    expect(observation[2]).toBeCloseTo(0.5);
    expect(observation[7]).toBeCloseTo(0.5);
    observation.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(index === 0 || index === 2 ? -1 : 0);
      expect(value).toBeLessThanOrEqual(1);
    });
  });

  it('reports asymmetric forward/left/right wall distances in the agent frame', () => {
    const world = createWorld(14);
    const agent = world.agents[0];
    agent.position = { x: 2, z: -1 };
    agent.heading = 0;

    const facingNorth = observeAgent(world, 'left');
    expect(facingNorth[4]).toBeCloseTo(8.65 / ARENA_CONFIG.sensorRange, 12);
    expect(facingNorth[5]).toBeCloseTo(13.65 / ARENA_CONFIG.sensorRange, 12);
    expect(facingNorth[6]).toBeCloseTo(9.65 / ARENA_CONFIG.sensorRange, 12);

    agent.heading = Math.PI / 2;
    const facingEast = observeAgent(world, 'left');
    expect(facingEast[4]).toBeCloseTo(9.65 / ARENA_CONFIG.sensorRange, 12);
    expect(facingEast[5]).toBeCloseTo(8.65 / ARENA_CONFIG.sensorRange, 12);
    expect(facingEast[6]).toBeCloseTo(6.65 / ARENA_CONFIG.sensorRange, 12);
    expect(facingEast[5]).not.toBeCloseTo(facingEast[6]);
  });

  it('keeps target channels invariant when the agent and targets translate together', () => {
    const world = createWorld(15);
    const agent = world.agents[0];
    agent.position = { x: -2, z: 1 };
    agent.heading = 0.37;
    agent.velocity = { x: 1, z: -2 };
    world.foods = [{ id: 'food', position: { x: 1, z: 5 }, radius: 0.25, respawns: 0 }];
    world.hazards = [
      {
        id: 'hazard',
        position: { x: -4, z: -2 },
        previousPosition: { x: -4, z: -2 },
        velocity: { x: 0, z: 0 },
        radius: 0.6
      }
    ];
    const original = observeAgent(world, 'left');

    const translation = { x: 3, z: -2 };
    agent.position.x += translation.x;
    agent.position.z += translation.z;
    world.foods[0].position.x += translation.x;
    world.foods[0].position.z += translation.z;
    world.hazards[0].position.x += translation.x;
    world.hazards[0].position.z += translation.z;
    const translated = observeAgent(world, 'left');

    for (const channel of [0, 1, 2, 3, 7] as const) {
      expect(translated[channel]).toBeCloseTo(original[channel], 12);
    }
  });

  it('stays finite for coincident targets and transforms bearings with agent heading', () => {
    const world = createWorld(13);
    const agent = world.agents[0];
    world.foods[0].position = { ...agent.position };
    world.hazards[0].position = { ...agent.position };
    expect(observeAgent(world, 'left').every(Number.isFinite)).toBe(true);

    world.foods[0].position = { x: agent.position.x + 2, z: agent.position.z };
    agent.heading = Math.PI / 2;
    expect(observeAgent(world, 'left')[0]).toBeCloseTo(0);
  });

  it('uses the retained custom bounds, sensor range, and max speed coherently', () => {
    const config = { ...ARENA_CONFIG, halfWidth: 20, halfDepth: 10, sensorRange: 10, maxSpeed: 4 };
    const world = createWorld(16, config);
    const agent = world.agents[0];
    agent.position = { x: 0, z: 0 };
    agent.heading = 0;
    agent.velocity = { x: 0, z: 2 };

    const observation = observeAgent(world, 'left');
    expect(observation[4]).toBeCloseTo((10 - agent.radius) / 10, 12);
    expect(observation[7]).toBeCloseTo(0.5, 12);
    expect(() => observeAgent(world, 'left', ARENA_CONFIG)).toThrow(/config.*match/i);
  });
});
