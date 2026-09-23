import type { ActionInput, DecodedAction } from './types';

const finiteOrZero = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : 0;

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(maximum, Math.max(minimum, value));

const isActionArray = (input: ActionInput | undefined): input is readonly number[] =>
  Array.isArray(input);

/**
 * Shared decoder for every experimental mode. Array order is thrust, yaw,
 * brake. Invalid/missing values become zero before applying declared ranges.
 */
export const decodeAction = (input: ActionInput | undefined): DecodedAction => {
  const thrust = isActionArray(input) ? input[0] : input?.thrust;
  const yaw = isActionArray(input) ? input[1] : input?.yaw;
  const brake = isActionArray(input) ? input[2] : input?.brake;

  return {
    thrust: clamp(finiteOrZero(thrust), -1, 1),
    yaw: clamp(finiteOrZero(yaw), -1, 1),
    brake: clamp(finiteOrZero(brake), 0, 1)
  };
};
