export interface ArenaConfig {
  /** Fixed simulation step. 1/30 second; callers may not supply render delta time. */
  fixedDeltaSeconds: number;
  halfWidth: number;
  halfDepth: number;
  agentRadius: number;
  foodRadius: number;
  hazardRadius: number;
  foodCount: number;
  hazardCount: number;
  spawnInset: number;
  maxSpeed: number;
  acceleration: number;
  turnRate: number;
  rollingDrag: number;
  brakeDrag: number;
  movementScorePerUnit: number;
  foodScore: number;
  hazardPenalty: number;
  sensorRange: number;
}

/**
 * Practical browser-POC resource ceilings. Placement compares every new entity
 * with existing blockers, so these bounds cap both allocation and O(n²) work.
 */
export const ARENA_CONFIG_LIMITS = Object.freeze({
  maxFoodCount: 64,
  maxHazardCount: 32,
  maxEntityCount: 96,
  maxFixedDeltaSeconds: 1,
  maxArenaHalfExtent: 10_000,
  maxEntityRadius: 1_000,
  maxSpawnInset: 10_000,
  maxSpeed: 1_000,
  maxAcceleration: 10_000,
  maxTurnRate: 1_000,
  maxDrag: 10_000,
  maxScoreRate: 1_000_000,
  maxSensorRange: 100_000
});

/**
 * Deterministic model constants. World units are arbitrary metres-like units,
 * velocities are units/second, and headings use radians (0 points along +z).
 */
export const ARENA_CONFIG: Readonly<ArenaConfig> = Object.freeze({
  fixedDeltaSeconds: 1 / 30,
  halfWidth: 12,
  halfDepth: 8,
  agentRadius: 0.35,
  foodRadius: 0.25,
  hazardRadius: 0.6,
  foodCount: 4,
  hazardCount: 2,
  spawnInset: 1,
  maxSpeed: 6,
  acceleration: 9,
  turnRate: Math.PI,
  rollingDrag: 0.7,
  brakeDrag: 8,
  movementScorePerUnit: 0.1,
  foodScore: 10,
  hazardPenalty: 2,
  sensorRange: 24
});

const CONFIG_KEYS = Object.keys(ARENA_CONFIG) as (keyof ArenaConfig)[];

const invalid = (message: string): never => {
  throw new Error(`Invalid arena config: ${message}`);
};

/** Validate and detach caller-owned configuration before it enters world state. */
export const retainArenaConfig = (input: Readonly<ArenaConfig>): Readonly<ArenaConfig> => {
  const config = { ...input };
  const positive: (keyof ArenaConfig)[] = [
    'fixedDeltaSeconds',
    'halfWidth',
    'halfDepth',
    'agentRadius',
    'foodRadius',
    'hazardRadius',
    'maxSpeed',
    'sensorRange'
  ];
  for (const key of positive) {
    if (!Number.isFinite(config[key]) || config[key] <= 0) invalid(`${key} must be finite and positive`);
  }
  for (const key of ['foodCount', 'hazardCount'] as const) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 0) {
      invalid(`${key} must be an integral non-negative value`);
    }
  }
  if (
    config.foodCount > ARENA_CONFIG_LIMITS.maxFoodCount ||
    config.hazardCount > ARENA_CONFIG_LIMITS.maxHazardCount ||
    config.foodCount + config.hazardCount > ARENA_CONFIG_LIMITS.maxEntityCount
  ) {
    invalid('entity count exceeds the practical browser limit');
  }
  for (const key of [
    'spawnInset',
    'acceleration',
    'turnRate',
    'rollingDrag',
    'brakeDrag',
    'movementScorePerUnit',
    'foodScore',
    'hazardPenalty'
  ] as const) {
    if (!Number.isFinite(config[key]) || config[key] < 0) invalid(`${key} must be finite and non-negative`);
  }
  const boundedFields: ReadonlyArray<readonly [keyof ArenaConfig, number]> = [
    ['fixedDeltaSeconds', ARENA_CONFIG_LIMITS.maxFixedDeltaSeconds],
    ['halfWidth', ARENA_CONFIG_LIMITS.maxArenaHalfExtent],
    ['halfDepth', ARENA_CONFIG_LIMITS.maxArenaHalfExtent],
    ['agentRadius', ARENA_CONFIG_LIMITS.maxEntityRadius],
    ['foodRadius', ARENA_CONFIG_LIMITS.maxEntityRadius],
    ['hazardRadius', ARENA_CONFIG_LIMITS.maxEntityRadius],
    ['spawnInset', ARENA_CONFIG_LIMITS.maxSpawnInset],
    ['maxSpeed', ARENA_CONFIG_LIMITS.maxSpeed],
    ['acceleration', ARENA_CONFIG_LIMITS.maxAcceleration],
    ['turnRate', ARENA_CONFIG_LIMITS.maxTurnRate],
    ['rollingDrag', ARENA_CONFIG_LIMITS.maxDrag],
    ['brakeDrag', ARENA_CONFIG_LIMITS.maxDrag],
    ['movementScorePerUnit', ARENA_CONFIG_LIMITS.maxScoreRate],
    ['foodScore', ARENA_CONFIG_LIMITS.maxScoreRate],
    ['hazardPenalty', ARENA_CONFIG_LIMITS.maxScoreRate],
    ['sensorRange', ARENA_CONFIG_LIMITS.maxSensorRange]
  ];
  for (const [key, maximum] of boundedFields) {
    if (config[key] > maximum) invalid(`${key} exceeds the practical arithmetic limit`);
  }
  // Explicitly guard every compound used by one physics tick, even if limits change later.
  for (const [label, value] of [
    ['acceleration step', config.acceleration * config.fixedDeltaSeconds],
    ['turn step', config.turnRate * config.fixedDeltaSeconds],
    [
      'drag step',
      (config.rollingDrag + config.brakeDrag) * config.fixedDeltaSeconds
    ],
    ['maximum movement step', config.maxSpeed * config.fixedDeltaSeconds],
    ['maximum movement score step', config.maxSpeed * config.fixedDeltaSeconds * config.movementScorePerUnit]
  ] as const) {
    if (!Number.isFinite(value)) invalid(`${label} must remain finite`);
  }
  if (config.halfWidth <= config.agentRadius * 4 || config.halfDepth <= config.agentRadius) {
    invalid('dimensions cannot fit the two agent starts');
  }
  for (const radius of [config.foodRadius, config.hazardRadius]) {
    if (config.spawnInset + radius >= config.halfWidth || config.spawnInset + radius >= config.halfDepth) {
      invalid('dimensions and spawnInset cannot fit an entity');
    }
  }
  const diskArea =
    Math.PI *
    (2 * config.agentRadius ** 2 +
      config.foodCount * config.foodRadius ** 2 +
      config.hazardCount * config.hazardRadius ** 2);
  if (diskArea > 4 * config.halfWidth * config.halfDepth * 0.7) {
    invalid('entity counts and radii exceed bounded placement capacity');
  }
  return Object.freeze(config);
};

export const arenaConfigsEqual = (
  left: Readonly<ArenaConfig>,
  right: Readonly<ArenaConfig>
): boolean => CONFIG_KEYS.every((key) => Object.is(left[key], right[key]));

const canonicalNumber = (value: number): string => (Object.is(value, -0) ? '-0' : String(value));

/** Plain-data, versioned identity for the exact validated simulation semantics. */
export const createArenaConfigFingerprint = (config: Readonly<ArenaConfig>): string =>
  `arena-config-v1|${CONFIG_KEYS.map((key) => `${key}=${canonicalNumber(config[key])}`).join('|')}`;

/** Revalidate deserialized/cast world config and its retained identity. */
export const validateRetainedArenaConfig = (
  config: Readonly<ArenaConfig>,
  fingerprint: string
): Readonly<ArenaConfig> => {
  try {
    const validated = retainArenaConfig(config);
    if (fingerprint !== createArenaConfigFingerprint(validated)) {
      throw new Error('fingerprint mismatch');
    }
    return validated;
  } catch {
    throw new Error('Invalid retained arena config identity or contents');
  }
};

/** Use retained world config; a legacy explicit override must be exactly coherent. */
export const resolveArenaConfig = (
  retained: Readonly<ArenaConfig>,
  supplied?: Readonly<ArenaConfig>
): Readonly<ArenaConfig> => {
  if (supplied && !arenaConfigsEqual(retained, supplied)) {
    throw new Error('Arena config override must match the world config');
  }
  return retained;
};
