import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { parseGraphBinary, validateGraph } from '../../src/lib/connectome/format';
import { createWorld, stepWorld } from '../../src/lib/arena/world';
import {
  buildSeedTrace,
  DEFAULT_GRAPH_ID,
  graphIdFromPath,
  loadGraphArtifact,
  parseArgs,
  TRACE_SUBSTEPS
} from '../../scripts/training/export-traces';
import { createTraceGraph } from '../fixtures/trace-graph';

/**
 * Covers three thermo review findings against
 * `scripts/training/export-traces.ts` that `tests/unit/golden-traces.test.ts`
 * doesn't exercise (it only regenerates the committed default export):
 * `--graph` gzip support, the `--out` overwrite guard, and the opt-in
 * `includeWorld` per-tick world column.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');
const REAL_ARTIFACT = resolve(publicDataDir, 'malecns-arena-v1.bin.gz');

describe('graphIdFromPath', () => {
  // Round-3 thermo review (`thermo-maintainability` S3 / `thermo-architecture`
  // S3): `basename(path, extname(path))` alone strips only the last
  // extension, so `malecns-arena-v1.bin.gz` mangled to `graphId =
  // "malecns-arena-v1.bin"`. `graphIdFromPath` strips a trailing `.gz`
  // first, then the remaining extension.
  it('strips both .bin and a trailing .gz from a gzip artifact path', () => {
    expect(graphIdFromPath('public/data/malecns-arena-v1.bin.gz')).toBe('malecns-arena-v1');
  });

  it('strips just .bin from a non-gzip artifact path (no mangling regression)', () => {
    expect(graphIdFromPath('public/data/malecns-arena-v1.bin')).toBe('malecns-arena-v1');
  });

  it('matches on the real committed gzip artifact', () => {
    expect(graphIdFromPath(REAL_ARTIFACT)).toBe('malecns-arena-v1');
  });

  it('leaves a path with no recognized extension untouched (basename only)', () => {
    expect(graphIdFromPath('some/dir/plain-name')).toBe('plain-name');
  });
});

describe('loadGraphArtifact', () => {
  it('detects gzip by magic bytes (not filename) and parses/validates the real committed artifact', () => {
    // public/data/malecns-arena-v1.bin.gz is the one real graph artifact in
    // this repo (produced by scripts/data/compile.py's
    // binfmt.write_gzip_deterministic, a real gzip stream, not a raw .bin).
    // Before this fix, handing it to loadGraphArtifact failed with "bad
    // magic" because the raw gzip bytes were passed straight to
    // parseGraphBinary without decompressing first.
    const graph = loadGraphArtifact(REAL_ARTIFACT);
    expect(() => validateGraph(graph)).not.toThrow();
    expect(graph.metadata.neuronCount).toBeGreaterThan(0);

    // Cross-check against an independent decompress-then-parse path
    // (mirrors tests/unit/malecns-artifact.test.ts), so this isn't just
    // checking that loadGraphArtifact doesn't throw.
    const binary = gunzipSync(readFileSync(REAL_ARTIFACT));
    const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
    const direct = parseGraphBinary(arrayBuffer);

    expect(graph.metadata).toEqual(direct.metadata);
    expect(Array.from(graph.biologicalIds)).toEqual(Array.from(direct.biologicalIds));
  });
});

describe('parseArgs overwrite guard', () => {
  it('refuses --substeps with no --out at all', () => {
    expect(() => parseArgs(['--substeps', '7'])).toThrow(/--out/);
  });

  it('refuses --include-world with no --out at all', () => {
    expect(() => parseArgs(['--include-world'])).toThrow(/--out/);
  });

  it('refuses an explicit --out that resolves to the committed fixtures directory', () => {
    expect(() => parseArgs(['--substeps', '7', '--out', 'tests/fixtures/golden'])).toThrow(
      /committed/
    );
  });

  it('refuses an explicit --out that textually differs but resolves to the same directory', () => {
    expect(() =>
      parseArgs(['--include-world', '--out', 'tests/fixtures/../fixtures/golden'])
    ).toThrow(/committed/);
  });

  it('allows a non-default export to a genuinely different --out path', () => {
    expect(() => parseArgs(['--substeps', '7', '--out', 'training/runs/tmp'])).not.toThrow();
    expect(() => parseArgs(['--include-world', '--out', 'training/runs/tmp'])).not.toThrow();
  });

  it('allows the default (no flags) invocation, which targets the committed directory on purpose', () => {
    expect(() => parseArgs([])).not.toThrow();
  });

  describe('--arena-task', () => {
    it('rejects an unknown arena task id', () => {
      expect(() => parseArgs(['--arena-task', 'not-a-real-task'])).toThrow(/--arena-task must be one of/);
    });

    it('rejects --arena-task combined with --graph/--substeps/--include-world', () => {
      expect(() => parseArgs(['--arena-task', 'hazard-heavy', '--substeps', '7'])).toThrow(
        /--arena-task cannot be combined with/
      );
    });

    it('defaults --out to tests/fixtures/golden/tasks/<id> for a non-default task', () => {
      const args = parseArgs(['--arena-task', 'hazard-heavy']);
      expect(args.outDir.endsWith('tests/fixtures/golden/tasks/hazard-heavy')).toBe(true);
      expect(args.outDirExplicit).toBe(false);
    });

    it('--arena-task default resolves the same outDir/outDirExplicit as no --arena-task flag at all', () => {
      // arenaTask itself legitimately differs ("default" vs. undefined —
      // both are meaningful, recorded provenance elsewhere), but every
      // output-affecting field must be identical.
      const withDefault = parseArgs(['--arena-task', 'default']);
      const withNone = parseArgs([]);
      expect(withDefault.outDir).toBe(withNone.outDir);
      expect(withDefault.outDirExplicit).toBe(withNone.outDirExplicit);
      expect(withDefault.graphPath).toBe(withNone.graphPath);
      expect(withDefault.substeps).toBe(withNone.substeps);
      expect(withDefault.includeWorld).toBe(withNone.includeWorld);
    });

    // Regression (dual-review finding): an earlier version's overwrite guard
    // (`nonDefaultExport`) only ever looked at --graph/--substeps/
    // --include-world, so `--arena-task <id> --out tests/fixtures/golden`
    // sailed through unchecked and would have silently overwritten the real
    // committed default fixtures (same filenames: trace-graph.json,
    // trace-graph-seed-1.json) with task-variant data.
    it('refuses an explicit --out that resolves to the committed default fixtures directory', () => {
      expect(() => parseArgs(['--arena-task', 'hazard-heavy', '--out', 'tests/fixtures/golden'])).toThrow(
        /committed/
      );
    });

    it('allows an explicit --out to a genuinely different directory', () => {
      expect(() => parseArgs(['--arena-task', 'hazard-heavy', '--out', 'training/runs/tmp'])).not.toThrow();
    });
  });
});

describe('buildSeedTrace includeWorld', () => {
  it('is absent by default (committed-shape trace)', () => {
    const graph = createTraceGraph();
    const trace = buildSeedTrace(graph, DEFAULT_GRAPH_ID, 1, 3, TRACE_SUBSTEPS);
    expect(trace).not.toHaveProperty('worldAfter');
    expect(JSON.stringify(trace)).not.toContain('worldAfter');
  });

  it('records post-step world state that self-consistently matches an independent replay of initialWorld + actions through stepWorld', () => {
    const graph = createTraceGraph();
    const seed = 1;
    const ticks = 5;
    const trace = buildSeedTrace(graph, DEFAULT_GRAPH_ID, seed, ticks, TRACE_SUBSTEPS, {
      includeWorld: true
    });

    expect(trace.worldAfter).toBeDefined();
    expect(trace.worldAfter).toHaveLength(ticks);

    // Independent replay: starts fresh from createWorld(seed) (matching
    // buildSeedTrace's own starting point) and re-derives each tick's
    // post-step state directly from stepWorld's own output object, not by
    // calling this exporter's serializer again -- that would only prove the
    // serializer agrees with itself, not that it recorded the real state.
    let world = createWorld(seed);
    for (let tick = 0; tick < ticks; tick += 1) {
      const [thrust, yaw, brake] = trace.actions[tick];
      world = stepWorld(world, { left: [thrust, yaw, brake], right: [0, 0, 0] });

      const recorded = trace.worldAfter?.[tick];
      expect(recorded).toBeDefined();
      if (!recorded) continue;

      expect(recorded.tick).toBe(world.tick);
      expect(recorded.timeSeconds).toBe(world.timeSeconds);
      expect(recorded.rngState).toBe(world.rngState);
      expect(recorded.agentIds).toEqual(world.agents.map((agent) => agent.id));
      expect(recorded.agentPositions).toEqual(
        world.agents.map((agent) => [agent.position.x, agent.position.z])
      );
      expect(recorded.agentVelocities).toEqual(
        world.agents.map((agent) => [agent.velocity.x, agent.velocity.z])
      );
      expect(recorded.agentHeadings).toEqual(world.agents.map((agent) => agent.heading));
      expect(recorded.agentScores).toEqual(
        world.agents.map((agent) => [
          agent.score.foodPickups,
          agent.score.hazardContacts,
          agent.score.distanceTravelled,
          agent.score.movementScore
        ])
      );
      expect(recorded.agentActiveHazardIds).toEqual(
        world.agents.map((agent) => [...agent.activeHazardIds])
      );
      expect(recorded.foodPositions).toEqual(
        world.foods.map((food) => [food.position.x, food.position.z])
      );
      expect(recorded.foodRespawns).toEqual(world.foods.map((food) => food.respawns));
      expect(recorded.hazardPositions).toEqual(
        world.hazards.map((hazard) => [hazard.position.x, hazard.position.z])
      );
      expect(recorded.hazardVelocities).toEqual(
        world.hazards.map((hazard) => [hazard.velocity.x, hazard.velocity.z])
      );
    }
  });
});
