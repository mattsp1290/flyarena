import { decodeAction } from './actions';
import {
  ARENA_CONFIG,
  createArenaConfigFingerprint,
  resolveArenaConfig,
  retainArenaConfig,
  validateRetainedArenaConfig,
  type ArenaConfig
} from './config';
import type {
  ActionsByAgent,
  AgentId,
  AgentState,
  ArenaSnapshot,
  FoodState,
  HazardState,
  Vec2,
  WorldState
} from './types';

const normalizeSeed = (seed: number): number => {
  const value = Number.isFinite(seed) ? seed >>> 0 : 1;
  return value === 0 ? 0x6d2b79f5 : value;
};

/** Platform-independent xorshift32. Returns the next non-zero uint32 state. */
export const nextRandomState = (state: number): number => {
  let value = normalizeSeed(state);
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return value >>> 0;
};

const randomUnit = (state: number): [number, number] => {
  const next = nextRandomState(state);
  return [next, next / 0x1_0000_0000];
};

const randomPosition = (
  rngState: number,
  radius: number,
  config: Readonly<ArenaConfig>
): [number, Vec2] => {
  let xUnit: number;
  let zUnit: number;
  [rngState, xUnit] = randomUnit(rngState);
  [rngState, zUnit] = randomUnit(rngState);
  const inset = config.spawnInset + radius;
  return [
    rngState,
    {
      x: -config.halfWidth + inset + xUnit * (2 * (config.halfWidth - inset)),
      z: -config.halfDepth + inset + zUnit * (2 * (config.halfDepth - inset))
    }
  ];
};

interface PlacementBlocker {
  position: Vec2;
  radius: number;
}

const overlaps = (a: Vec2, aRadius: number, b: Vec2, bRadius: number): boolean =>
  Math.hypot(a.x - b.x, a.z - b.z) <= aRadius + bRadius;

const positionIsClear = (
  position: Vec2,
  radius: number,
  blockers: readonly PlacementBlocker[]
): boolean => blockers.every((blocker) => !overlaps(position, radius, blocker.position, blocker.radius));

const placeWithoutOverlap = (
  rngState: number,
  radius: number,
  config: Readonly<ArenaConfig>,
  blockers: readonly PlacementBlocker[]
): [number, Vec2] => {
  for (let attempt = 0; attempt < 64; attempt += 1) {
    let candidate: Vec2;
    [rngState, candidate] = randomPosition(rngState, radius, config);
    if (positionIsClear(candidate, radius, blockers)) return [rngState, candidate];
  }

  // Bounded, runtime-independent lattice fallback for unlucky random streams.
  const inset = config.spawnInset + radius;
  const minX = -config.halfWidth + inset;
  const minZ = -config.halfDepth + inset;
  const width = 2 * (config.halfWidth - inset);
  const depth = 2 * (config.halfDepth - inset);
  const divisions = 64;
  for (let zIndex = 0; zIndex <= divisions; zIndex += 1) {
    for (let xIndex = 0; xIndex <= divisions; xIndex += 1) {
      const candidate = {
        x: minX + (width * xIndex) / divisions,
        z: minZ + (depth * zIndex) / divisions
      };
      if (positionIsClear(candidate, radius, blockers)) return [rngState, candidate];
    }
  }
  throw new Error('Invalid arena config: bounded placement could not find non-overlapping space');
};

const createAgent = (id: AgentId, x: number, heading: number, config: Readonly<ArenaConfig>): AgentState => ({
  id,
  position: { x, z: 0 },
  previousPosition: { x, z: 0 },
  velocity: { x: 0, z: 0 },
  heading,
  previousHeading: heading,
  radius: config.agentRadius,
  activeHazardIds: [],
  score: { foodPickups: 0, hazardContacts: 0, distanceTravelled: 0, movementScore: 0 }
});

/** Create a complete reset state. The same seed and config are byte-equivalent. */
export const createWorld = (
  seed: number,
  config: Readonly<ArenaConfig> = ARENA_CONFIG
): WorldState => {
  const retainedConfig = retainArenaConfig(config);
  const normalizedSeed = normalizeSeed(seed);
  let rngState = normalizedSeed;
  const foods: FoodState[] = [];
  const hazards: HazardState[] = [];
  const agents = [
    createAgent('left', -retainedConfig.halfWidth / 4, Math.PI / 2, retainedConfig),
    createAgent('right', retainedConfig.halfWidth / 4, -Math.PI / 2, retainedConfig)
  ];
  const blockers: PlacementBlocker[] = [...agents];

  for (let index = 0; index < retainedConfig.foodCount; index += 1) {
    let position: Vec2;
    [rngState, position] = placeWithoutOverlap(
      rngState,
      retainedConfig.foodRadius,
      retainedConfig,
      blockers
    );
    const food = { id: `food-${index}`, position, radius: retainedConfig.foodRadius, respawns: 0 };
    foods.push(food);
    blockers.push(food);
  }

  for (let index = 0; index < retainedConfig.hazardCount; index += 1) {
    let position: Vec2;
    let directionUnit: number;
    [rngState, position] = placeWithoutOverlap(
      rngState,
      retainedConfig.hazardRadius,
      retainedConfig,
      blockers
    );
    [rngState, directionUnit] = randomUnit(rngState);
    const angle = directionUnit * Math.PI * 2;
    const speed = 1.25 + index * 0.25;
    hazards.push({
      id: `hazard-${index}`,
      position,
      previousPosition: { ...position },
      velocity: { x: Math.sin(angle) * speed, z: Math.cos(angle) * speed },
      radius: retainedConfig.hazardRadius
    });
    blockers.push(hazards[hazards.length - 1]);
  }

  return {
    schemaVersion: 1,
    config: retainedConfig,
    configFingerprint: createArenaConfigFingerprint(retainedConfig),
    seed: normalizedSeed,
    rngState,
    tick: 0,
    timeSeconds: 0,
    agents,
    foods,
    hazards
  };
};

const validateStepState = (world: Readonly<WorldState>, config: Readonly<ArenaConfig>): void => {
  const expectedTime = world.tick * config.fixedDeltaSeconds;
  if (
    world.schemaVersion !== 1 ||
    !Number.isSafeInteger(world.tick) ||
    world.tick < 0 ||
    world.tick >= Number.MAX_SAFE_INTEGER ||
    !Number.isFinite(world.timeSeconds) ||
    !Number.isFinite(expectedTime) ||
    !Object.is(world.timeSeconds, expectedTime)
  ) {
    throw new Error('Invalid world clock state');
  }

  const finite = (...values: readonly number[]): boolean => values.every(Number.isFinite);
  for (const agent of world.agents) {
    if (
      !finite(
        agent.position.x,
        agent.position.z,
        agent.previousPosition.x,
        agent.previousPosition.z,
        agent.velocity.x,
        agent.velocity.z,
        agent.heading,
        agent.previousHeading,
        agent.radius,
        agent.score.distanceTravelled,
        agent.score.movementScore
      ) ||
      Math.hypot(agent.velocity.x, agent.velocity.z) > config.maxSpeed * (1 + 1e-12) ||
      !Number.isSafeInteger(agent.score.foodPickups) ||
      !Number.isSafeInteger(agent.score.hazardContacts)
    ) {
      throw new Error('Invalid world state numeric value');
    }
  }
  for (const food of world.foods) {
    if (!finite(food.position.x, food.position.z, food.radius) || !Number.isSafeInteger(food.respawns)) {
      throw new Error('Invalid world state numeric value');
    }
  }
  for (const hazard of world.hazards) {
    if (
      !finite(
        hazard.position.x,
        hazard.position.z,
        hazard.previousPosition.x,
        hazard.previousPosition.z,
        hazard.velocity.x,
        hazard.velocity.z,
        hazard.radius
      )
    ) {
      throw new Error('Invalid world state numeric value');
    }
  }
};

const cloneWorld = (world: Readonly<WorldState>): WorldState => ({
  ...world,
  config: retainArenaConfig(world.config),
  agents: world.agents.map((agent) => ({
    ...agent,
    position: { ...agent.position },
    previousPosition: { ...agent.previousPosition },
    velocity: { ...agent.velocity },
    activeHazardIds: [...agent.activeHazardIds],
    score: { ...agent.score }
  })),
  foods: world.foods.map((food) => ({ ...food, position: { ...food.position } })),
  hazards: world.hazards.map((hazard) => ({
    ...hazard,
    position: { ...hazard.position },
    previousPosition: { ...hazard.previousPosition },
    velocity: { ...hazard.velocity }
  }))
});

const wrapAngle = (angle: number): number => {
  const wrapped = ((angle + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
  return wrapped === -Math.PI ? Math.PI : wrapped;
};

const clampAgentToArena = (agent: AgentState, config: Readonly<ArenaConfig>): void => {
  const maxX = config.halfWidth - agent.radius;
  const maxZ = config.halfDepth - agent.radius;
  if (agent.position.x < -maxX || agent.position.x > maxX) {
    agent.position.x = Math.max(-maxX, Math.min(maxX, agent.position.x));
    agent.velocity.x = 0;
  }
  if (agent.position.z < -maxZ || agent.position.z > maxZ) {
    agent.position.z = Math.max(-maxZ, Math.min(maxZ, agent.position.z));
    agent.velocity.z = 0;
  }
};

const processContacts = (world: WorldState, config: Readonly<ArenaConfig>): void => {
  for (const food of world.foods) {
    const collector = world.agents.find((agent) =>
      overlaps(agent.position, agent.radius, food.position, food.radius)
    );
    if (collector) {
      collector.score.foodPickups += 1;
      collector.score.movementScore += config.foodScore;
      let position: Vec2;
      const blockers = [
        ...world.agents,
        ...world.foods.filter((candidate) => candidate.id !== food.id),
        ...world.hazards
      ];
      [world.rngState, position] = placeWithoutOverlap(
        world.rngState,
        food.radius,
        config,
        blockers
      );
      food.position = position;
      food.respawns += 1;
    }
  }

  for (const agent of world.agents) {
    const previous = new Set(agent.activeHazardIds);
    const active = world.hazards
      .filter((hazard) => overlaps(agent.position, agent.radius, hazard.position, hazard.radius))
      .map((hazard) => hazard.id)
      .sort();
    for (const hazardId of active) {
      if (!previous.has(hazardId)) {
        agent.score.hazardContacts += 1;
        agent.score.movementScore -= config.hazardPenalty;
      }
    }
    agent.activeHazardIds = active;
  }
};

const moveHazards = (world: WorldState, config: Readonly<ArenaConfig>): void => {
  const dt = config.fixedDeltaSeconds;
  for (const hazard of world.hazards) {
    hazard.previousPosition = { ...hazard.position };
    hazard.position.x += hazard.velocity.x * dt;
    hazard.position.z += hazard.velocity.z * dt;
    const maxX = config.halfWidth - hazard.radius;
    const maxZ = config.halfDepth - hazard.radius;
    if (hazard.position.x < -maxX || hazard.position.x > maxX) {
      hazard.position.x = Math.max(-maxX, Math.min(maxX, hazard.position.x));
      hazard.velocity.x *= -1;
    }
    if (hazard.position.z < -maxZ || hazard.position.z > maxZ) {
      hazard.position.z = Math.max(-maxZ, Math.min(maxZ, hazard.position.z));
      hazard.velocity.z *= -1;
    }
  }
};

/**
 * Advance exactly one fixed 30 Hz tick. The input is never mutated and there
 * is deliberately no variable-delta overload, so render cadence cannot leak
 * into physics.
 */
export const stepWorld = (
  input: Readonly<WorldState>,
  actions: ActionsByAgent = {},
  suppliedConfig?: Readonly<ArenaConfig>
): WorldState => {
  const config = resolveArenaConfig(
    validateRetainedArenaConfig(input.config, input.configFingerprint),
    suppliedConfig
  );
  validateStepState(input, config);
  const world = cloneWorld(input);
  const dt = config.fixedDeltaSeconds;

  for (const agent of world.agents) {
    const action = decodeAction(actions[agent.id]);
    agent.previousPosition = { ...agent.position };
    agent.previousHeading = agent.heading;
    agent.heading = wrapAngle(agent.heading + action.yaw * config.turnRate * dt);
    agent.velocity.x += Math.sin(agent.heading) * action.thrust * config.acceleration * dt;
    agent.velocity.z += Math.cos(agent.heading) * action.thrust * config.acceleration * dt;
    const drag = Math.max(0, 1 - (config.rollingDrag + action.brake * config.brakeDrag) * dt);
    agent.velocity.x *= drag;
    agent.velocity.z *= drag;
    const speed = Math.hypot(agent.velocity.x, agent.velocity.z);
    if (speed > config.maxSpeed) {
      const scale = config.maxSpeed / speed;
      agent.velocity.x *= scale;
      agent.velocity.z *= scale;
    }
    agent.position.x += agent.velocity.x * dt;
    agent.position.z += agent.velocity.z * dt;
    clampAgentToArena(agent, config);
    const distance = Math.hypot(
      agent.position.x - agent.previousPosition.x,
      agent.position.z - agent.previousPosition.z
    );
    agent.score.distanceTravelled += distance;
    agent.score.movementScore += distance * config.movementScorePerUnit;
  }

  moveHazards(world, config);
  processContacts(world, config);
  world.tick += 1;
  world.timeSeconds = world.tick * dt;
  return world;
};

const interpolate = (previous: number, current: number, alpha: number): number =>
  previous + (current - previous) * alpha;

/**
 * Produce detached presentation data between the last two fixed states.
 * Calling this at any frequency has no side effects on simulation or replay.
 */
export const createSnapshot = (
  world: Readonly<WorldState>,
  alpha: number,
  suppliedConfig?: Readonly<ArenaConfig>
): ArenaSnapshot => {
  const config = resolveArenaConfig(
    validateRetainedArenaConfig(world.config, world.configFingerprint),
    suppliedConfig
  );
  const amount = Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 0;
  return {
    tick: world.tick,
    timeSeconds: Math.max(0, world.timeSeconds - config.fixedDeltaSeconds * (1 - amount)),
    agents: world.agents.map((agent) => ({
      id: agent.id,
      position: {
        x: interpolate(agent.previousPosition.x, agent.position.x, amount),
        z: interpolate(agent.previousPosition.z, agent.position.z, amount)
      },
      heading: wrapAngle(
        agent.previousHeading + wrapAngle(agent.heading - agent.previousHeading) * amount
      )
    })),
    foods: world.foods.map((food) => ({ ...food, position: { ...food.position } })),
    hazards: world.hazards.map((hazard) => ({
      id: hazard.id,
      position: {
        x: interpolate(hazard.previousPosition.x, hazard.position.x, amount),
        z: interpolate(hazard.previousPosition.z, hazard.position.z, amount)
      },
      radius: hazard.radius
    }))
  };
};
