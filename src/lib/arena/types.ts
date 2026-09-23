import type { ArenaConfig } from './config';

/** Recursively readonly policy/view contract, including nested arrays and objects. */
export type DeepReadonly<T> = T extends (...args: never[]) => unknown
  ? T
  : T extends readonly (infer Item)[]
    ? readonly DeepReadonly<Item>[]
    : T extends object
      ? { readonly [Key in keyof T]: DeepReadonly<T[Key]> }
      : T;

/** A horizontal arena vector. Distances are in world units and angles are radians. */
export interface Vec2 {
  x: number;
  z: number;
}

export type AgentId = 'left' | 'right';

export interface AgentScore {
  foodPickups: number;
  hazardContacts: number;
  distanceTravelled: number;
  movementScore: number;
}

export interface AgentState {
  id: AgentId;
  position: Vec2;
  previousPosition: Vec2;
  velocity: Vec2;
  heading: number;
  previousHeading: number;
  radius: number;
  activeHazardIds: string[];
  score: AgentScore;
}

export interface FoodState {
  id: string;
  position: Vec2;
  radius: number;
  respawns: number;
}

export interface HazardState {
  id: string;
  position: Vec2;
  previousPosition: Vec2;
  velocity: Vec2;
  radius: number;
}

/** JSON-serializable simulation state. rngState is the complete PRNG state. */
export interface WorldState {
  schemaVersion: 1;
  /** Validated, detached simulation semantics retained for every subsequent operation. */
  readonly config: Readonly<ArenaConfig>;
  /** Canonical identity checked before config-dependent operations. */
  readonly configFingerprint: string;
  seed: number;
  rngState: number;
  tick: number;
  timeSeconds: number;
  agents: AgentState[];
  foods: FoodState[];
  hazards: HazardState[];
}

export type ReadonlyWorldState = DeepReadonly<WorldState>;

export type Observation = readonly [
  foodBearing: number,
  foodDistance: number,
  hazardBearing: number,
  hazardDistance: number,
  forwardClearance: number,
  leftClearance: number,
  rightClearance: number,
  speed: number
];

export interface DecodedAction {
  /** Signed forward/reverse thrust, normalized to [-1, 1]. */
  thrust: number;
  /** Signed yaw command, normalized to [-1, 1]. */
  yaw: number;
  /** Brake amount, normalized to [0, 1]. */
  brake: number;
}

export type ActionInput = readonly number[] | Partial<DecodedAction>;
export type ActionsByAgent = Readonly<Partial<Record<AgentId, ActionInput>>>;

export interface RenderAgentSnapshot {
  id: AgentId;
  position: Vec2;
  heading: number;
}

export interface RenderHazardSnapshot {
  id: string;
  position: Vec2;
  radius: number;
}

export interface ArenaSnapshot {
  tick: number;
  timeSeconds: number;
  agents: RenderAgentSnapshot[];
  foods: FoodState[];
  hazards: RenderHazardSnapshot[];
}
