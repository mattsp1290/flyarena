/**
 * Pure, framework-agnostic presentation helpers for the arena renderer.
 *
 * Everything here operates on plain numbers/arrays and the read-only arena
 * snapshot types; nothing imports Three.js or touches the DOM. That keeps
 * the renderer's actual visual-mapping logic unit-testable under jsdom
 * (which has no WebGL), and keeps `ArenaScene.ts` focused on wiring these
 * results into scene objects rather than deriving them.
 */
import type {
  AgentId,
  FoodState,
  RenderAgentSnapshot,
  RenderHazardSnapshot,
  Vec2
} from '../arena/types';

export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

const TRAIL_POSITION_STRIDE = 3;

/**
 * Egocentric forward direction for a world heading, using the same
 * convention as `arena/sensors.ts` (`dx = Math.sin(angle)`,
 * `dz = Math.cos(angle)`; heading 0 points along +z). Three.js rotates a
 * local +Z axis to exactly `(sin(rotationY), cos(rotationY))` under a plain
 * Y-axis rotation, so an agent mesh whose nose points along local +Z can be
 * oriented with `group.rotation.y = heading` directly — no axis remapping.
 */
export const agentForwardVector = (heading: number): { x: number; z: number } => ({
  x: Math.sin(heading),
  z: Math.cos(heading)
});

/** Map one interpolated agent snapshot to a scene transform. Pure, no mutation. */
export const agentTransform = (
  agent: Readonly<RenderAgentSnapshot>,
  liftY: number
): { position: Vec3Like; rotationY: number } => ({
  position: { x: agent.position.x, y: liftY, z: agent.position.z },
  rotationY: agent.heading
});

/**
 * Shift a preallocated trail buffer left by one point and append the newest
 * point at the end. No allocation: `copyWithin` moves existing samples in
 * place. Returns the updated `filled` count (saturates at `capacity`),
 * which callers use as the draw-range length since the oldest samples sit
 * at the front of an unfilled buffer.
 */
export const pushTrailPoint = (
  positions: Float32Array,
  capacity: number,
  filled: number,
  point: Readonly<Vec3Like>
): number => {
  if (capacity <= 0) return 0;
  positions.copyWithin(0, TRAIL_POSITION_STRIDE);
  const offset = (capacity - 1) * TRAIL_POSITION_STRIDE;
  positions[offset] = point.x;
  positions[offset + 1] = point.y;
  positions[offset + 2] = point.z;
  return Math.min(capacity, filled + 1);
};

export interface FoodPickupEvent {
  id: string;
  /** The food's position immediately before it respawned elsewhere. */
  position: Vec2;
}

/** The minimal per-food state `detectFoodPickups` needs to remember between calls. */
export interface FoodRespawnRecord {
  respawns: number;
  position: Vec2;
}

/**
 * Derive visual-only food-pickup events by comparing respawn counters
 * between frames — `FoodState.respawns` already increments in
 * `arena/world.ts` whenever an agent collects that food. This never reads
 * or infers agent identity/collision internals; it only detects a change
 * that already happened, purely for triggering a decorative effect.
 *
 * Callers pass a small `id -> last-seen record` map rather than a whole
 * previous snapshot, so this cannot end up comparing an object against
 * itself if a future caller reuses/pools snapshot objects, and lookup is
 * O(current.length) instead of O(current.length * previous.length).
 *
 * If the same food is collected more than once within a single call (a
 * multi-tick catch-up step), only one event is emitted, at the position
 * recorded before the first of those pickups; this is an accepted
 * limitation for a decorative effect, not a scoring path.
 */
export const detectFoodPickups = (
  previous: ReadonlyMap<string, Readonly<FoodRespawnRecord>> | undefined,
  current: readonly Readonly<FoodState>[]
): readonly FoodPickupEvent[] => {
  if (!previous) return [];
  const events: FoodPickupEvent[] = [];
  for (const food of current) {
    const before = previous.get(food.id);
    if (before && food.respawns > before.respawns) {
      events.push({ id: food.id, position: { ...before.position } });
    }
  }
  return events;
};

export interface HazardContactEvent {
  agentId: AgentId;
  hazardId: string;
  position: Vec2;
}

export interface HazardContactResult {
  events: readonly HazardContactEvent[];
  /** Pass this back in as `previousTouching` on the next call. */
  touching: ReadonlySet<string>;
}

const touchingKey = (agentId: AgentId, hazardId: string): string => `${agentId}:${hazardId}`;

/**
 * Visual-only proximity check between interpolated agent and hazard
 * positions, independent of `AgentState.activeHazardIds` (which is not
 * part of the render snapshot contract). Only reports a rising edge — the
 * instant a pair starts overlapping — so a lingering overlap does not
 * retrigger the effect every frame.
 */
export const detectHazardContacts = (
  agents: readonly Readonly<RenderAgentSnapshot>[],
  hazards: readonly Readonly<RenderHazardSnapshot>[],
  agentRadius: number,
  previousTouching: ReadonlySet<string>
): HazardContactResult => {
  const touching = new Set<string>();
  const events: HazardContactEvent[] = [];
  for (const agent of agents) {
    for (const hazard of hazards) {
      const distance = Math.hypot(
        agent.position.x - hazard.position.x,
        agent.position.z - hazard.position.z
      );
      // Written as "not <=" rather than "> ... continue" so a NaN distance
      // (which should never happen — the world validates finite state)
      // fails closed instead of registering as a contact.
      if (!(distance <= agentRadius + hazard.radius)) continue;
      const key = touchingKey(agent.id, hazard.id);
      touching.add(key);
      if (!previousTouching.has(key)) {
        events.push({ agentId: agent.id, hazardId: hazard.id, position: { ...hazard.position } });
      }
    }
  }
  return { events, touching };
};
