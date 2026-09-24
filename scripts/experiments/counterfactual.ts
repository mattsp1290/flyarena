import { readFile, stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { verifyAndDecompressArtifact, type ArenaManifest } from '../../src/lib/experiment/assets';
import { parseGraphBinary } from '../../src/lib/connectome/format';
import { prepareGraph } from '../../src/lib/counterfactual/targets';
import { evidenceHeader, runExperiment } from '../../src/lib/counterfactual/engine';
import { compareEvidence, readEvidenceRequest, serializeEvidence } from '../../src/lib/counterfactual/evidence';
import { DEFAULT_REQUEST, validateRequest, type ExportDocument } from '../../src/lib/counterfactual/types';

const { values } = parseArgs({ options: {
  verify: { type: 'string' }, 'compare-numerical': { type: 'string' }, output: { type: 'string' },
  data: { type: 'string', default: 'public/data' }, seed: { type: 'string' }, seeds: { type: 'string' },
  warmup: { type: 'string' }, horizon: { type: 'string' }, topology: { type: 'string' }, target: { type: 'string' }
} });
if (values.verify && values['compare-numerical']) throw new Error('Choose exact verification or numerical comparison');
const inputPath = values.verify ?? values['compare-numerical'];
if (!inputPath && !values.output) throw new Error('Use --output FILE to retain the evidence, or --verify FILE');
if (inputPath && ['seed', 'seeds', 'warmup', 'horizon', 'topology', 'target'].some(k => values[k as keyof typeof values] !== undefined)) {
  throw new Error('Verification settings come only from the export');
}
let input: ReturnType<typeof readEvidenceRequest> | undefined;
if (inputPath) {
  if ((await stat(inputPath)).size > 16 * 1024 ** 2) throw new Error('Export exceeds 16 MiB');
  input = readEvidenceRequest(JSON.parse(await readFile(inputPath, 'utf8')));
}
const request = input?.request ?? validateRequest({
  ...DEFAULT_REQUEST,
  seed: values.seed === undefined ? DEFAULT_REQUEST.seed : Number(values.seed),
  seedCount: values.seeds === undefined ? DEFAULT_REQUEST.seedCount : Number(values.seeds),
  warmup: values.warmup === undefined ? DEFAULT_REQUEST.warmup : Number(values.warmup),
  horizon: values.horizon === undefined ? DEFAULT_REQUEST.horizon : Number(values.horizon),
  topology: values.topology ?? DEFAULT_REQUEST.topology, target: values.target ?? DEFAULT_REQUEST.target
});
const data = resolve(values.data);
const manifest: ArenaManifest = JSON.parse(await readFile(resolve(data, 'malecns-arena-v1.manifest.json'), 'utf8'));
async function artifact(name: string) {
  if (!/^[a-zA-Z0-9._-]+$/.test(name)) throw new Error('Invalid artifact filename');
  const buffer = await readFile(resolve(data, name));
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}
const [biological, rewired] = await Promise.all([
  artifact(manifest.artifact).then(b => verifyAndDecompressArtifact(b, manifest)),
  artifact(manifest.rewiredArms.seed0.artifact).then(b => verifyAndDecompressArtifact(b, manifest.rewiredArms.seed0))
]);
const parsedBiological = parseGraphBinary(biological.slice(0));
const prepared = await prepareGraph({ manifest, biological, rewired, parsedBiological }, request.topology);
if (input) {
  // Fail before simulation on mismatched model identity, targets, or seeds as well as after it on outcomes.
  const header = evidenceHeader(prepared, request);
  const actual = Object.fromEntries(Object.keys(header).map(key => [key, input!.evidence[key]]));
  if (!compareEvidence(header, actual).matches) throw new Error('Export model, graph, target or seed identity mismatch');
}
const started = performance.now();
const evidence = runExperiment(prepared, request);
if (input) {
  const comparison = compareEvidence(evidence, input.evidence, Boolean(values['compare-numerical']));
  console.log(JSON.stringify({ status: comparison.matches ? comparison.exact ? 'exact reproduction' : 'numerically close; not exact reproduction' : 'not reproduced', ...comparison }, null, 2));
  if (!comparison.matches) process.exitCode = 1;
}
if (values.output) {
  const document: ExportDocument = { evidence, runtime: { producer: `Node ${process.version}`, platform: `${process.platform}/${process.arch}`, elapsedMs: performance.now() - started } };
  await writeFile(values.output, serializeEvidence(document));
  console.log(`Saved ${values.output}`);
}
