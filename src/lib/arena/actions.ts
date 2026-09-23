import type { ActionInput, DecodedAction } from './types';

const finiteOrZero = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : 0;

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(maximum, Math.max(minimum, value));

const isActionArray = (input: ActionInput | undefined): input is readonly number[] =>
  Array.isArray(input);

/**
 * Canonical output-population order: the single source of truth for what
 * `decodeAction`'s array form (and every neural graph's `outputPopulationIndex`,
 * see `docs/graph-format.md` and `src/lib/connectome/model.ts`) means by
 * population 0/1/2. Referenced by name elsewhere instead of restating the
 * indices, so a future reordering is a one-line change here rather than a
 * multi-file prose convention to keep in sync by hand.
 */
export const OUTPUT_POPULATION = { thrust: 0, yaw: 1, brake: 2 } as const;

/**
 * Shared decoder for every experimental mode. Array order matches
 * `OUTPUT_POPULATION`: thrust, yaw, brake. Invalid/missing values become
 * zero before applying declared ranges.
 */
export const decodeAction = (input: ActionInput | undefined): DecodedAction => {
  const thrust = isActionArray(input) ? input[OUTPUT_POPULATION.thrust] : input?.thrust;
  const yaw = isActionArray(input) ? input[OUTPUT_POPULATION.yaw] : input?.yaw;
  const brake = isActionArray(input) ? input[OUTPUT_POPULATION.brake] : input?.brake;

  return {
    thrust: clamp(finiteOrZero(thrust), -1, 1),
    yaw: clamp(finiteOrZero(yaw), -1, 1),
    brake: clamp(finiteOrZero(brake), 0, 1)
  };
};
