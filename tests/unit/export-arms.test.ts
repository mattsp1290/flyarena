import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

import { encodeGraphBinary, validateGraph } from '../../src/lib/connectome/format';
import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import {
  computeGraphIdentity,
  deserializeArmBundle,
  ExportArmsGateError,
  parseExportArmsArgs,
  runExportArms,
  type SerializedArmBundle
} from '../../scripts/training/export-arms';
import { createTraceGraph } from '../fixtures/trace-graph';

describe('parseExportArmsArgs', () => {
  it('rejects --fixture-rewire combined with --graph', () => {
    expect(() => parseExportArmsArgs(['--graph', 'foo.bin', '--fixture-rewire'])).toThrow(/cannot be combined/);
  });

  it('rejects --rewired without --graph', () => {
    expect(() => parseExportArmsArgs(['--rewired', 'foo.bin'])).toThrow(/requires --graph/);
  });

  it('parses defaults with no arguments (trace-graph mode)', () => {
    const args = parseExportArmsArgs([]);
    expect(args.graphPath).toBeUndefined();
    expect(args.rewiredPath).toBeUndefined();
    expect(args.fixtureRewire).toBe(false);
    expect(args.outDir).toBe('training/runs/arms');
  });
});

describe('computeGraphIdentity: graphId derivation', () => {
  // Same fix as export-traces.ts's graphIdFromPath (round-3 review S3):
  // computeGraphIdentity shares the same "strip .bin.gz, not just .gz"
  // requirement, via the shared graphIdFromPath helper.
  it('strips both .bin and a trailing .gz for a gzip artifact', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const graphPath = resolve(outDir, 'malecns-arena-v1.bin.gz');
      writeFileSync(graphPath, gzipSync(Buffer.from(encodeGraphBinary(createTraceGraph()))));
      const identity = computeGraphIdentity(graphPath);
      expect(identity.graphId).toBe('malecns-arena-v1');
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('strips just .bin for a non-gzip artifact', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const graphPath = resolve(outDir, 'malecns-arena-v1.bin');
      writeFileSync(graphPath, Buffer.from(encodeGraphBinary(createTraceGraph())));
      const identity = computeGraphIdentity(graphPath);
      expect(identity.graphId).toBe('malecns-arena-v1');
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });
});

describe('runExportArms (trace-graph mode)', () => {
  it('exports biological + disconnected with equal D when no rewired arm is requested', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const result = runExportArms({ outDir, fixtureRewire: false, fixtureRewireSeed: 0 });
      expect([...result.written].sort()).toEqual(['biological', 'disconnected']);

      const bio = JSON.parse(readFileSync(resolve(result.outDir, 'biological.json'), 'utf8')) as SerializedArmBundle;
      const disc = JSON.parse(readFileSync(resolve(result.outDir, 'disconnected.json'), 'utf8')) as SerializedArmBundle;
      expect(bio.D).toBe(disc.D);
      expect(disc.metadata.edgeCount).toBe(0);
      expect(disc.provenance.kind).toBe('disconnected-runtime-zero-edge');
      expect(bio.provenance.kind).toBe('biological-trace-graph-fixture');

      const bioGraph = deserializeArmBundle(bio);
      const discGraph = deserializeArmBundle(disc);
      expect(() => validateGraph(bioGraph)).not.toThrow();
      expect(() => validateGraph(discGraph)).not.toThrow();
      expect(outputNeuronIndices(bioGraph).length).toBe(bio.D);
      expect(outputNeuronIndices(discGraph).length).toBe(disc.D);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('exports all three arms with equal D when --fixture-rewire is set, preserving edge count and degrees', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const result = runExportArms({ outDir, fixtureRewire: true, fixtureRewireSeed: 42 });
      expect([...result.written].sort()).toEqual(['biological', 'disconnected', 'rewired']);

      const bundles = new Map(
        result.written.map((arm) => [
          arm,
          JSON.parse(readFileSync(resolve(result.outDir, `${arm}.json`), 'utf8')) as SerializedArmBundle
        ])
      );
      const dValues = new Set([...bundles.values()].map((bundle) => bundle.D));
      expect(dValues.size).toBe(1);

      const biological = bundles.get('biological')!;
      const rewired = bundles.get('rewired')!;
      expect(rewired.provenance.kind).toBe('rewired-fixture-only-swap');
      expect(rewired.metadata.edgeCount).toBe(biological.metadata.edgeCount);

      const rewiredGraph = deserializeArmBundle(rewired);
      expect(() => validateGraph(rewiredGraph)).not.toThrow();

      // Degree-preserving: every neuron's out-degree (row length) is
      // unchanged by the fixture-only swap.
      const biologicalGraph = deserializeArmBundle(biological);
      for (let neuron = 0; neuron < biologicalGraph.metadata.neuronCount; neuron += 1) {
        const originalOutDegree =
          biologicalGraph.presynapticOffsets[neuron + 1] - biologicalGraph.presynapticOffsets[neuron];
        const rewiredOutDegree = rewiredGraph.presynapticOffsets[neuron + 1] - rewiredGraph.presynapticOffsets[neuron];
        expect(rewiredOutDegree).toBe(originalOutDegree);
      }
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('writes bundles under a graphArtifactSha256-named directory matching computeGraphIdentity', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const identity = computeGraphIdentity(undefined);
      const result = runExportArms({ outDir, fixtureRewire: false, fixtureRewireSeed: 0 });
      expect(result.outDir).toBe(resolve(outDir, identity.graphArtifactSha256));
      expect(existsSync(result.outDir)).toBe(true);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('rejects a --rewired artifact whose node set does not match --graph (D equal, but a different graph)', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const biological = createTraceGraph();
      // Same D, same edge/degree structure, but a different node identity
      // (biologicalIds shifted) — exactly the "compiled from a different
      // graph with the same output-neuron count" case the D-only gate
      // could not catch.
      const differentGraph = {
        ...biological,
        biologicalIds: BigUint64Array.from(biological.biologicalIds, (id) => id + 1000n)
      };
      expect(outputNeuronIndices(differentGraph).length).toBe(outputNeuronIndices(biological).length);

      const bioPath = resolve(outDir, 'biological.bin');
      const rewiredPath = resolve(outDir, 'different.bin');
      writeFileSync(bioPath, Buffer.from(encodeGraphBinary(biological)));
      writeFileSync(rewiredPath, Buffer.from(encodeGraphBinary(differentGraph)));

      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(ExportArmsGateError);
      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(/node set/);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('removes a stale rewired.json left from a previous export when re-exporting without --rewired/--fixture-rewire', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const first = runExportArms({ outDir, fixtureRewire: true, fixtureRewireSeed: 7 });
      const rewiredPath = resolve(first.outDir, 'rewired.json');
      expect(existsSync(rewiredPath)).toBe(true);

      const second = runExportArms({ outDir, fixtureRewire: false, fixtureRewireSeed: 0 });
      expect(second.outDir).toBe(first.outDir);
      expect([...second.written].sort()).toEqual(['biological', 'disconnected']);
      expect(existsSync(rewiredPath)).toBe(false);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });
});

describe('runExportArms: assertMatchingNodeSet gate (round-2 R2-I1 regression coverage)', () => {
  // R2-I1 (round-2 action item, `reviews/feat-l127-arm-export-evaluator-round2-.../04-action-items.md`)
  // asked for a `leakRate`-changed and an `inputWeight`-scaled `--rewired`
  // artifact, each asserted to throw `ExportArmsGateError`; the thermo-nuclear
  // round-3 review (`thermo-architecture`, I2) found the underlying gate fix
  // (`export-arms.ts:328-355`) correct but these regression tests still
  // missing. This block covers every non-`edgeCount` branch
  // `assertMatchingNodeSet` checks: metadata (via `leakRate`), `inputWeight`,
  // `presynapticSigns`, and `outputWeight` — not just the pre-existing
  // "different graph, shifted `biologicalIds`" case above, which exercises a
  // different comparison (`biologicalIds`, not these four).
  const writeGraphBin = (outDir: string, fileName: string, graph: ReturnType<typeof createTraceGraph>): string => {
    const path = resolve(outDir, fileName);
    writeFileSync(path, Buffer.from(encodeGraphBinary(graph)));
    return path;
  };

  it('throws when the "rewired" artifact has a different leakRate (metadata mismatch)', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const biological = createTraceGraph();
      const mutated = { ...biological, metadata: { ...biological.metadata, leakRate: biological.metadata.leakRate + 0.1 } };
      expect(() => validateGraph(mutated)).not.toThrow(); // the mutation alone must stay a valid graph

      const bioPath = writeGraphBin(outDir, 'biological.bin', biological);
      const rewiredPath = writeGraphBin(outDir, 'leak-rate-mutated.bin', mutated);

      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(ExportArmsGateError);
      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(/metadata/);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('throws when the "rewired" artifact\'s inputWeight is scaled', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const biological = createTraceGraph();
      const mutated = { ...biological, inputWeight: Float32Array.from(biological.inputWeight, (w) => w * 2) };
      expect(() => validateGraph(mutated)).not.toThrow();
      // Sanity: the fixture actually has nonzero input weights, so scaling
      // by 2 is a real, detectable change, not a no-op on all-zero data.
      expect(Array.from(mutated.inputWeight).some((w) => w !== 0)).toBe(true);

      const bioPath = writeGraphBin(outDir, 'biological.bin', biological);
      const rewiredPath = writeGraphBin(outDir, 'input-weight-scaled.bin', mutated);

      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(ExportArmsGateError);
      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(/inputWeight/);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('throws when the "rewired" artifact has one flipped presynapticSign', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const biological = createTraceGraph();
      const flippedSigns = Int8Array.from(biological.presynapticSigns);
      flippedSigns[0] = flippedSigns[0] === 1 ? -1 : 1;
      const mutated = { ...biological, presynapticSigns: flippedSigns };
      expect(() => validateGraph(mutated)).not.toThrow(); // -1/1 stays a valid sign

      const bioPath = writeGraphBin(outDir, 'biological.bin', biological);
      const rewiredPath = writeGraphBin(outDir, 'sign-flipped.bin', mutated);

      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(ExportArmsGateError);
      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(/presynapticSigns/);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('throws when the "rewired" artifact\'s outputWeight is changed', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const biological = createTraceGraph();
      const mutatedOutputWeight = Float32Array.from(biological.outputWeight);
      mutatedOutputWeight[18] += 0.5; // neuron 18 is an output neuron in this fixture (see trace-graph.ts)
      const mutated = { ...biological, outputWeight: mutatedOutputWeight };
      expect(() => validateGraph(mutated)).not.toThrow();

      const bioPath = writeGraphBin(outDir, 'biological.bin', biological);
      const rewiredPath = writeGraphBin(outDir, 'output-weight-mutated.bin', mutated);

      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(ExportArmsGateError);
      expect(() =>
        runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir })
      ).toThrow(/outputWeight/);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it('does not throw for an unmutated pair (same graph as both --graph and --rewired)', () => {
    const outDir = mkdtempSync(join(tmpdir(), 'export-arms-'));
    try {
      const biological = createTraceGraph();
      const bioPath = writeGraphBin(outDir, 'biological.bin', biological);
      // A fresh, independently-encoded copy of the exact same graph as the
      // "rewired" artifact: every field this gate checks is equal, so the
      // gate must not throw a false positive.
      const rewiredPath = writeGraphBin(outDir, 'unmutated-copy.bin', createTraceGraph());

      const result = runExportArms({ graphPath: bioPath, rewiredPath, fixtureRewire: false, fixtureRewireSeed: 0, outDir });
      expect([...result.written].sort()).toEqual(['biological', 'disconnected', 'rewired']);
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });
});
