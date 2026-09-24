// @vitest-environment node
import { afterAll, beforeAll, expect, it } from 'vitest';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import type { ExportDocument } from '../../src/lib/counterfactual/types';

let directory: string;
let document: ExportDocument;
const cli = (...args: string[]) => spawnSync(process.execPath,
  ['--import', 'tsx', 'scripts/experiments/counterfactual.ts', ...args],
  { encoding: 'utf8', timeout: 15_000 });

beforeAll(async () => {
  directory = await mkdtemp(join(tmpdir(), 'flyarena-causal-cli-'));
  const output = join(directory, 'original.json');
  const result = cli('--seeds', '4', '--warmup', '30', '--horizon', '30', '--output', output);
  expect(result.status, result.stderr).toBe(0);
  document = JSON.parse(await readFile(output, 'utf8'));
});
afterAll(async () => { if (directory) await rm(directory, { recursive: true, force: true }); });

it('exactly regenerates a real-graph Node export', () => {
  const result = cli('--verify', join(directory, 'original.json'));
  expect(result.status, result.stderr).toBe(0);
  expect(JSON.parse(result.stdout).status).toBe('exact reproduction');
});

const edits: [string, (copy: ExportDocument) => void][] = [
  ['graph', copy => { copy.evidence.graph.binarySha256 = '0'.repeat(64); }],
  ['target', copy => { copy.evidence.target.bodyIds[0] = '999'; }],
  ['seeds', copy => { copy.evidence.seeds[0]++; }],
  ['version', copy => { copy.evidence.schemaVersion = 2 as 1; }],
  ['replay', copy => { const frame = copy.evidence.results[0].branches.baseline.frames[1]; frame.snapshot = { ...frame.snapshot, tick: frame.snapshot.tick + 1 }; }],
  ['outcome', copy => { copy.evidence.results[0].branches.lesion.outcome.movementScore += 1; }]
];
it.each(edits)('rejects altered %s evidence with a nonzero CLI exit', async (name, edit) => {
  const copy = structuredClone(document);
  edit(copy);
  const path = join(directory, `${name}.json`);
  await writeFile(path, JSON.stringify(copy));
  const result = cli('--verify', path);
  expect(result.error).toBeUndefined();
  expect(result.status).toBe(1);
  expect(result.stdout + result.stderr).toMatch(/mismatch|Unsupported evidence version|not reproduced/);
});
