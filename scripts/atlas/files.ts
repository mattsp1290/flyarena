import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { readAtlasManifest, verifyAtlasBytes } from '../../src/lib/atlas/assets';
import { MAX_ATLAS_BYTES, ATLAS_FILE } from '../../src/lib/atlas/types';

export async function loadLocalAtlas(data: string) {
  const manifestPath = resolve(data, 'behavior-atlas-v1.manifest.json'),
    artifactPath = resolve(data, ATLAS_FILE);
  if ((await stat(manifestPath)).size > 4096 || (await stat(artifactPath)).size > MAX_ATLAS_BYTES)
    throw new Error('Atlas file exceeds bounds');
  const manifest = readAtlasManifest(JSON.parse(await readFile(manifestPath, 'utf8')));
  const bytes = await readFile(artifactPath);
  return verifyAtlasBytes(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    manifest
  );
}
