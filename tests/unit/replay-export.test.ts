import { describe, expect, it } from 'vitest';
import { createExperimentReplayExport } from '../../src/lib/arena/replay';
import { createWorld, stepWorld } from '../../src/lib/arena/world';

/**
 * WP6 item 6: the replay download must contain configuration and score
 * traces only, never the connectome asset. This asserts the export shape
 * directly, both by field-name absence and by a full round-trip through
 * `JSON.stringify` (catching anything a typed array might smuggle through
 * that a plain key check would miss).
 */

const CONNECTOME_ONLY_KEYS = [
  'biologicalIds',
  'presynapticOffsets',
  'postsynapticIndices',
  'contactMagnitudes',
  'presynapticSigns',
  'inputChannelIndex',
  'inputWeight',
  'outputPopulationIndex',
  'outputWeight'
];

describe('createExperimentReplayExport', () => {
  const buildWorld = () => {
    let world = createWorld(123);
    for (let tick = 0; tick < 30; tick += 1) {
      world = stepWorld(world, { left: { thrust: 0.4, yaw: 0.1, brake: 0 }, right: { thrust: 0.3, yaw: -0.1, brake: 0 } });
    }
    return world;
  };

  it('contains configuration and score traces, and nothing connectome-shaped', () => {
    const world = buildWorld();
    const trace = [
      {
        tick: world.tick,
        timeSeconds: world.timeSeconds,
        agents: {
          left: { ...world.agents[0].score },
          right: { ...world.agents[1].score }
        }
      }
    ];

    const replay = createExperimentReplayExport(world, {
      topology: { left: 'biological', right: 'rewired' },
      substepsPerTick: 4,
      totalTicks: 2700,
      trace
    });

    expect(replay.seed).toBe(world.seed);
    expect(replay.configFingerprint).toBe(world.configFingerprint);
    expect(replay.topology).toEqual({ left: 'biological', right: 'rewired' });
    expect(replay.trace).toEqual(trace);
    expect(replay.finalSummary.ticks).toBe(30);

    const serialized = JSON.stringify(replay);
    for (const key of CONNECTOME_ONLY_KEYS) {
      expect(serialized.includes(key)).toBe(false);
    }
    // A typed array (e.g. a stray Float32Array reference) would either fail
    // to JSON.stringify meaningfully or serialize as an object with numeric
    // keys and no recognizable connectome field name — the field-name check
    // above is the meaningful guard, but also assert nothing bigger than a
    // trivial config/summary/trace payload made it in.
    expect(serialized.length).toBeLessThan(20_000);
  });

  it('never includes the connectome even when the world was produced from a real graph-driven run', () => {
    // Regression guard: even though `WorldState` itself never carries graph
    // data (see `src/lib/arena/types.ts`), assert the replay export's own
    // JSON shape at the top level matches exactly the declared schema —
    // catching an accidental future field addition that leaks graph state.
    const world = buildWorld();
    const replay = createExperimentReplayExport(world, {
      topology: { left: 'disconnected', right: 'biological' },
      substepsPerTick: 4,
      totalTicks: 100,
      trace: []
    });
    expect(Object.keys(replay).sort()).toEqual(
      [
        'schemaVersion',
        'seed',
        'configFingerprint',
        'topology',
        'graphBinarySha256',
        'substepsPerTick',
        'totalTicks',
        'finalSummary',
        'finalHash',
        'trace'
      ].sort()
    );
  });

  it('threads each arm’s manifest-verified graph binarySha256 through, and omits it (rather than a placeholder) for an arm with none — e.g. the runtime-derived disconnected control', () => {
    const world = buildWorld();
    const hash = 'a'.repeat(64);
    const replay = createExperimentReplayExport(world, {
      topology: { left: 'biological', right: 'disconnected' },
      graphBinarySha256: { left: hash },
      substepsPerTick: 4,
      totalTicks: 2700,
      trace: []
    });

    expect(replay.graphBinarySha256.left).toBe(hash);
    expect(replay.graphBinarySha256.right).toBeUndefined();

    // Round-trip through JSON (what the actual download does): the missing
    // 'right' hash must be dropped, never serialized as null/"".
    const parsed = JSON.parse(JSON.stringify(replay)) as typeof replay;
    expect(Object.keys(parsed.graphBinarySha256)).toEqual(['left']);
  });

  it('omits graphBinarySha256 entirely (an empty object, never undefined) when the caller supplies none', () => {
    const world = buildWorld();
    const replay = createExperimentReplayExport(world, {
      topology: { left: 'biological', right: 'rewired' },
      substepsPerTick: 4,
      totalTicks: 2700,
      trace: []
    });
    expect(replay.graphBinarySha256).toBeDefined();
    expect(Object.keys(JSON.parse(JSON.stringify(replay)).graphBinarySha256)).toEqual([]);
  });
});
