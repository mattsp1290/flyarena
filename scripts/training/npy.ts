import { readFileSync, writeFileSync } from 'node:fs';

/**
 * Minimal NumPy `.npy` reader/writer, scoped to exactly what
 * `scripts/training/evaluate.ts` needs: a 1-D little-endian float array
 * (`numpy.save(path, theta_final)` on a `float32` or `float64` vector, the
 * natural output of `training/`'s not-yet-built CEM trainer, WP3). This is
 * not a general NPY implementation: no Fortran-order, multi-dimensional, or
 * big-endian support, because the run-directory contract this evaluator
 * consumes (`evaluate.ts`'s `RunConfig` doc comment) never needs them.
 *
 * Format (see https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html):
 * 6-byte magic `\x93NUMPY`, 1-byte major version, 1-byte minor version, a
 * header-length field (2 bytes LE for v1.x, 4 bytes LE for v2.0+), then an
 * ASCII Python-dict-literal header padded with spaces (and a trailing `\n`)
 * so the whole prologue is a multiple of 64 bytes, then raw array bytes.
 */

const NPY_MAGIC = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]); // "\x93NUMPY"

/** Read a 1-D little-endian float32 (`<f4`) or float64 (`<f8`) `.npy` array. */
export const readNpyFloat32Array = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  if (buffer.length < 10 || !buffer.subarray(0, 6).equals(NPY_MAGIC)) {
    throw new Error(`Invalid .npy file at ${path}: bad magic bytes`);
  }
  const majorVersion = buffer.readUInt8(6);
  let headerLength: number;
  let headerStart: number;
  if (majorVersion === 1) {
    headerLength = buffer.readUInt16LE(8);
    headerStart = 10;
  } else {
    headerLength = buffer.readUInt32LE(8);
    headerStart = 12;
  }
  if (buffer.length < headerStart + headerLength) {
    throw new Error(`Invalid .npy file at ${path}: truncated header`);
  }
  const header = buffer.toString('latin1', headerStart, headerStart + headerLength);

  const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  const fortranMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error(`Invalid .npy file at ${path}: unparseable header "${header}"`);
  }
  if (fortranMatch[1] !== 'False') {
    throw new Error(`Unsupported .npy file at ${path}: fortran_order must be False (1-D arrays only)`);
  }
  const shapeNumbers = shapeMatch[1]
    .split(',')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0)
    .map(Number);
  if (shapeNumbers.length !== 1) {
    throw new Error(
      `Unsupported .npy file at ${path}: expected a 1-D array, got shape (${shapeMatch[1]})`
    );
  }
  const elementCount = shapeNumbers[0];
  const descr = descrMatch[1];
  const dataStart = headerStart + headerLength;
  const dataBytes = buffer.subarray(dataStart);

  if (descr === '<f4') {
    if (dataBytes.length < elementCount * 4) throw new Error(`Truncated .npy data at ${path}`);
    const out = new Float32Array(elementCount);
    for (let index = 0; index < elementCount; index += 1) out[index] = dataBytes.readFloatLE(index * 4);
    return out;
  }
  if (descr === '<f8') {
    if (dataBytes.length < elementCount * 8) throw new Error(`Truncated .npy data at ${path}`);
    const out = new Float32Array(elementCount);
    for (let index = 0; index < elementCount; index += 1) out[index] = dataBytes.readDoubleLE(index * 8);
    return out;
  }
  throw new Error(
    `Unsupported .npy dtype "${descr}" at ${path}: expected little-endian float32 ('<f4') or ` +
      "float64 ('<f8')"
  );
};

/**
 * Write a 1-D float32 (`<f4`) `.npy` array. Used only to build this
 * repository's own tiny synthetic run-dir test fixture
 * (`tests/fixtures/trained-readout-run.ts`) — never by product code, which
 * only ever reads `theta_final.npy` written by the (separate) Python
 * trainer.
 */
export const writeNpyFloat32Array = (path: string, values: Float32Array): void => {
  const shapeLiteral = `(${values.length},)`;
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shapeLiteral}, }`;
  const prologueFixedBytes = 6 + 1 + 1 + 2; // magic + major + minor + v1.0 header-length field
  const unpaddedTotal = prologueFixedBytes + header.length + 1; // +1 for the trailing '\n'
  const remainder = unpaddedTotal % 64;
  const padLength = remainder === 0 ? 0 : 64 - remainder;
  const headerBytes = Buffer.from(header + ' '.repeat(padLength) + '\n', 'latin1');

  const out = Buffer.alloc(prologueFixedBytes + headerBytes.length + values.length * 4);
  NPY_MAGIC.copy(out, 0);
  out.writeUInt8(1, 6); // major version
  out.writeUInt8(0, 7); // minor version
  out.writeUInt16LE(headerBytes.length, 8);
  headerBytes.copy(out, 10);
  let offset = 10 + headerBytes.length;
  for (let index = 0; index < values.length; index += 1) {
    out.writeFloatLE(values[index], offset);
    offset += 4;
  }
  writeFileSync(path, out);
};
