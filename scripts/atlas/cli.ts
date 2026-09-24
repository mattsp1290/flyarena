import { readFileSync, statSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { parseArgs } from 'node:util';
import { isDeepStrictEqual } from 'node:util';
import { publishAtlas } from './publish';
import { validateAtlas } from '../../src/lib/atlas/validation';
import { ATLAS_FILE, MAX_ATLAS_BYTES, average } from '../../src/lib/atlas/types';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';

const { values } = parseArgs({
  options: {
    input: { type: 'string' },
    output: { type: 'string' },
    verify: { type: 'string' },
    diagnostic: { type: 'boolean' },
    data: { type: 'string', default: 'public/data' }
  }
});
if (Boolean(values.input) === Boolean(values.verify))
  throw new Error('Choose --input SEARCH.json or --verify ATLAS.json');
if (values.verify && (values.output || values.diagnostic))
  throw new Error('Verification cannot publish or bypass diversity');
const path = values.input ?? values.verify!;
if (statSync(path).size > MAX_ATLAS_BYTES) throw new Error('Input exceeds 8 MiB');
const input: unknown = JSON.parse(readFileSync(path, 'utf8'));
const previous = values.verify ? validateAtlas(input) : undefined;
const atlas = await publishAtlas(
  previous ? previous.source : input,
  values.data,
  !values.diagnostic
);
if (previous) {
  if (!isDeepStrictEqual(previous, atlas))
    throw new Error('Atlas does not reproduce exactly on this host');
  console.log(
    `Exact reproduction: ${atlas.cells.length} controllers, all discovery metrics, held-out controls, and replay frames`
  );
} else if (values.diagnostic) {
  console.log(
    JSON.stringify(
      atlas.evaluations.map((e) => {
        const gpu = atlas.source.candidates.find((c) => c.id === e.id)!;
        return {
          id: e.id,
          qualityError: average(e.discovery.map((m) => m.movementScore)) - gpu.quality,
          coverageError: average(e.discovery.map((m) => m.coverage)) - gpu.coverage,
          turningError: average(e.discovery.map((m) => m.turning)) - gpu.turning
        };
      }),
      null,
      2
    )
  );
} else {
  const output = resolve(values.output ?? `${values.data}/${ATLAS_FILE}`);
  if (!output.endsWith('/' + ATLAS_FILE)) throw new Error(`Output filename must be ${ATLAS_FILE}`);
  const bytes = JSON.stringify(atlas) + '\n';
  if (Buffer.byteLength(bytes) > MAX_ATLAS_BYTES) throw new Error('Published atlas exceeds 8 MiB');
  const manifest = {
    schemaVersion: 1,
    artifact: ATLAS_FILE,
    sha256: sha256Hex(bytes),
    bytes: Buffer.byteLength(bytes)
  };
  atomicWriteFileSync(output, bytes);
  atomicWriteFileSync(
    resolve(dirname(output), 'behavior-atlas-v1.manifest.json'),
    JSON.stringify(manifest, null, 2) + '\n'
  );
  console.log(
    `Published ${atlas.cells.length} controllers (${manifest.bytes} bytes, ${manifest.sha256})`
  );
}
