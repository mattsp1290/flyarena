import { resolveArenaConfig, validateRetainedArenaConfig, type ArenaConfig } from './config';
import type { AgentId, AgentState, Observation, Vec2, WorldState } from './types';

/**
 * Public neural input order. Bearings are signed turns from the agent's nose
 * divided by pi (-1..1). Distances, axis clearances, and speed are 0..1.
 * `speed` is the eighth channel: an egocentric proprioceptive magnitude.
 * No channel contains an absolute world coordinate.
 */
export const OBSERVATION_CHANNELS = [
  'foodBearing',
  'foodDistance',
  'hazardBearing',
  'hazardDistance',
  'forwardClearance',
  'leftClearance',
  'rightClearance',
  'speed'
] as const;

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

const wrapAngle = (angle: number): number =>
  ((angle + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;

const nearestEgocentric = (
  agent: Readonly<AgentState>,
  targets: readonly { position: Vec2 }[],
  sensorRange: number
): readonly [bearing: number, distance: number] => {
  let nearest: Vec2 | undefined;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const target of targets) {
    const distance = Math.hypot(
      target.position.x - agent.position.x,
      target.position.z - agent.position.z
    );
    if (distance < nearestDistance) {
      nearest = target.position;
      nearestDistance = distance;
    }
  }
  if (!nearest) return [0, 1];
  if (nearestDistance === 0) return [0, 0];
  const absoluteBearing = Math.atan2(
    nearest.x - agent.position.x,
    nearest.z - agent.position.z
  );
  return [wrapAngle(absoluteBearing - agent.heading) / Math.PI, clamp01(nearestDistance / sensorRange)];
};

const wallClearance = (
  agent: Readonly<AgentState>,
  relativeAngle: number,
  config: Readonly<ArenaConfig>
): number => {
  const angle = agent.heading + relativeAngle;
  const dx = Math.sin(angle);
  const dz = Math.cos(angle);
  const maxX = config.halfWidth - agent.radius;
  const maxZ = config.halfDepth - agent.radius;
  const xDistance =
    Math.abs(dx) < 1e-12
      ? Number.POSITIVE_INFINITY
      : ((dx > 0 ? maxX : -maxX) - agent.position.x) / dx;
  const zDistance =
    Math.abs(dz) < 1e-12
      ? Number.POSITIVE_INFINITY
      : ((dz > 0 ? maxZ : -maxZ) - agent.position.z) / dz;
  return clamp01(Math.max(0, Math.min(xDistance, zDistance)) / config.sensorRange);
};

export const observeAgent = (
  world: Readonly<WorldState>,
  agentId: AgentId,
  suppliedConfig?: Readonly<ArenaConfig>
): Observation => {
  const config = resolveArenaConfig(
    validateRetainedArenaConfig(world.config, world.configFingerprint),
    suppliedConfig
  );
  const agent = world.agents.find(({ id }) => id === agentId);
  if (!agent) throw new Error(`Unknown arena agent: ${agentId}`);
  const [foodBearing, foodDistance] = nearestEgocentric(agent, world.foods, config.sensorRange);
  const [hazardBearing, hazardDistance] = nearestEgocentric(
    agent,
    world.hazards,
    config.sensorRange
  );

  return [
    foodBearing,
    foodDistance,
    hazardBearing,
    hazardDistance,
    wallClearance(agent, 0, config),
    wallClearance(agent, -Math.PI / 2, config),
    wallClearance(agent, Math.PI / 2, config),
    clamp01(Math.hypot(agent.velocity.x, agent.velocity.z) / config.maxSpeed)
  ];
};
