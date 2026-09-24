import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { verifyAndDecompressArtifact, type ArenaManifest } from '../../src/lib/experiment/assets';
import { parseGraphBinary } from '../../src/lib/connectome/format';

/** Local counterpart to loadArenaArtifacts, sharing its integrity gates. */
export async function loadLocalAssets(data: string) {
  const manifest: ArenaManifest = JSON.parse(
    await readFile(resolve(data, 'malecns-arena-v1.manifest.json'), 'utf8')
  );
  async function artifact(name: string) {
    if (!/^[a-zA-Z0-9._-]+$/.test(name)) throw new Error('Invalid artifact filename');
    const buffer = await readFile(resolve(data, name));
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  }
  const [biological, rewired] = await Promise.all([
    artifact(manifest.artifact).then((bytes) => verifyAndDecompressArtifact(bytes, manifest)),
    artifact(manifest.rewiredArms.seed0.artifact).then((bytes) =>
      verifyAndDecompressArtifact(bytes, manifest.rewiredArms.seed0)
    )
  ]);
  return { manifest, biological, rewired, parsedBiological: parseGraphBinary(biological.slice(0)) };
}
