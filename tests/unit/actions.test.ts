import { describe, expect, it } from 'vitest';
import { decodeAction } from '../../src/lib/arena/actions';

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
});
