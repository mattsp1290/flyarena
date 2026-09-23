import { describe, expect, it } from 'vitest';
import { decodeAction, OUTPUT_POPULATION } from '../../src/lib/arena/actions';

describe('continuous action decoder', () => {
  it('maps every input shape to the same finite clamped contract', () => {
    expect(decodeAction([2, -4, 3])).toEqual({ thrust: 1, yaw: -1, brake: 1 });
    expect(decodeAction({ thrust: 2, yaw: -4, brake: 3 })).toEqual({
      thrust: 1,
      yaw: -1,
      brake: 1
    });
    expect(decodeAction([Number.NaN, Number.POSITIVE_INFINITY, -2])).toEqual({
      thrust: 0,
      yaw: 0,
      brake: 0
    });
    expect(decodeAction([])).toEqual({ thrust: 0, yaw: 0, brake: 0 });
  });

  it("ties OUTPUT_POPULATION's indices to decodeAction's array positions", () => {
    // Build an array form where every slot is unique and distinguishable,
    // then confirm each OUTPUT_POPULATION index reads the field it claims
    // to: a future reordering of either OUTPUT_POPULATION or decodeAction's
    // array destructuring alone (without updating the other) fails this.
    const features: number[] = [];
    features[OUTPUT_POPULATION.thrust] = 0.5;
    features[OUTPUT_POPULATION.yaw] = -0.25;
    features[OUTPUT_POPULATION.brake] = 0.75;

    const decoded = decodeAction(features);

    expect(decoded.thrust).toBeCloseTo(0.5);
    expect(decoded.yaw).toBeCloseTo(-0.25);
    expect(decoded.brake).toBeCloseTo(0.75);
  });
});
