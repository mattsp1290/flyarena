import { sha256Hex } from '../experiment/assets';
import { ATLAS_FILE, MAX_ATLAS_BYTES, type LoadedAtlas } from './types';
import { hash, object, validateAtlas } from './validation';

async function bounded(response: Response, limit: number): Promise<ArrayBuffer> {
  if (!response.ok) throw new Error(`Atlas request failed (${response.status})`);
  if (!response.body) throw new Error('Missing atlas body');
  const reader = response.body.getReader(),
    chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) throw new Error('Atlas exceeds size limit');
      chunks.push(value);
    }
  } finally {
    await reader.cancel();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes.buffer;
}
export interface AtlasManifest {
  schemaVersion: 1;
  artifact: string;
  sha256: string;
  bytes: number;
}
export function readAtlasManifest(input: unknown, expectedHash?: string): AtlasManifest {
  const manifest = object(input),
    sha256 = hash(manifest.sha256);
  if (
    manifest.schemaVersion !== 1 ||
    manifest.artifact !== ATLAS_FILE ||
    typeof manifest.bytes !== 'number' ||
    !Number.isInteger(manifest.bytes) ||
    manifest.bytes < 1 ||
    manifest.bytes > MAX_ATLAS_BYTES
  )
    throw new Error('Invalid atlas manifest');
  if (expectedHash !== undefined && sha256 !== expectedHash)
    throw new Error('Atlas changed since selection; reload and select a controller again');
  return { schemaVersion: 1, artifact: ATLAS_FILE, sha256, bytes: manifest.bytes };
}
export async function verifyAtlasBytes(
  bytes: ArrayBuffer,
  manifest: AtlasManifest
): Promise<LoadedAtlas> {
  if (
    bytes.byteLength !== manifest.bytes ||
    bytes.byteLength > MAX_ATLAS_BYTES ||
    (await sha256Hex(bytes)) !== manifest.sha256
  )
    throw new Error('Atlas integrity check failed');
  return {
    atlas: validateAtlas(JSON.parse(new TextDecoder().decode(bytes))),
    sha256: manifest.sha256
  };
}
export async function loadAtlas(
  base = `${import.meta.env.BASE_URL}data`,
  expectedHash?: string
): Promise<LoadedAtlas> {
  const manifest = readAtlasManifest(
    JSON.parse(
      new TextDecoder().decode(
        await bounded(await fetch(`${base}/behavior-atlas-v1.manifest.json`), 4096)
      )
    ),
    expectedHash
  );
  return verifyAtlasBytes(
    await bounded(await fetch(`${base}/${ATLAS_FILE}`), MAX_ATLAS_BYTES),
    manifest
  );
}
