import { loadLocalAtlas } from '../atlas/files';
import { resolveController } from '../../src/lib/atlas/controller';
import { AUTHORED, type Decoder } from '../../src/lib/counterfactual/decoder';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { loadLocalAssets } from './local-assets';
import { prepareGraph } from '../../src/lib/counterfactual/targets';
import { evidenceHeader, runExperiment } from '../../src/lib/counterfactual/engine';
import { compareEvidence, readEvidenceRequest, serializeEvidence } from '../../src/lib/counterfactual/evidence';
import { DEFAULT_REQUEST, validateRequest, type ExportDocument } from '../../src/lib/counterfactual/types';

const { values } = parseArgs({ options: {
  controller: { type: 'string' }, verify: { type: 'string' }, 'compare-numerical': { type: 'string' }, output: { type: 'string' },
  data: { type: 'string', default: 'public/data' }, seed: { type: 'string' }, seeds: { type: 'string' },
  warmup: { type: 'string' }, horizon: { type: 'string' }, topology: { type: 'string' }, target: { type: 'string' }
} });
if (values.verify && values['compare-numerical']) throw new Error('Choose exact verification or numerical comparison');
const inputPath = values.verify ?? values['compare-numerical'];
if (!inputPath && !values.output) throw new Error('Use --output FILE to retain the evidence, or --verify FILE');
if (inputPath && ['controller', 'seed', 'seeds', 'warmup', 'horizon', 'topology', 'target'].some(k => values[k as keyof typeof values] !== undefined)) {
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
const assets = await loadLocalAssets(data);
const prepared = await prepareGraph(assets, request.topology);
let decoder: Decoder = AUTHORED;
const exportedController = input?.evidence.decoder === 'atlas-trained' ? input.evidence.controller : undefined;
if (exportedController !== undefined || values.controller !== undefined) {
  if (exportedController !== undefined && (!exportedController || typeof exportedController !== 'object')) throw new Error('Invalid exported controller');
  const identity = exportedController as Record<string, unknown> | undefined;
  const id = identity ? identity.id : Number(values.controller);
  if (typeof id !== 'number' || !Number.isInteger(id)) throw new Error('Invalid controller ID');
  const loaded = await loadLocalAtlas(data);
  decoder = { decoder: 'atlas-trained', controller: await resolveController(loaded, prepared, id) };
}

if (input) {
  // Fail before simulation on mismatched model identity, targets, or seeds as well as after it on outcomes.
  const header = evidenceHeader(prepared, request, decoder);
  const actual = Object.fromEntries(Object.keys(header).map(key => [key, input!.evidence[key]]));
  if (!compareEvidence(header, actual).matches) throw new Error('Export model, graph, target or seed identity mismatch');
}
const started = performance.now();
const evidence = runExperiment(prepared, request, decoder);
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
