import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { readNpyFloat32Array, writeNpyFloat32Array } from '../../scripts/training/npy';

/**
 * Direct coverage for `scripts/training/npy.ts`'s hand-rolled `.npy`
 * reader/writer — previously exercised only indirectly, through
 * `writeNpyFloat32Array` -> `readNpyFloat32Array` round-trips inside
 * `tests/fixtures/trained-readout-run.ts` (used by `evaluate.test.ts`).
 * Round-2's `thermo-correctness` S1/S6 and round-3's `thermo-maintainability`
 * S2 both flagged this gap: a swapped byte offset or an off-by-one in
 * `headerStart`/`dataStart` is easy to introduce and easy to miss in review
 * without a dedicated regression test.
 *
 * `tests/fixtures/npy/real-numpy-{f4,f8}.npy` were written once by real
 * `numpy.save` (via `uv run --with numpy python3 -c "..."`, no separate
 * requirements file needed) and committed so this test does not depend on
 * `uv`/`numpy` being available in CI. To regenerate them:
 *
 * ```python
 * import numpy as np
 * np.save('tests/fixtures/npy/real-numpy-f4.npy', np.array([1.5, -2.25, 0.0, 3.0, 42.125], dtype='<f4'))
 * np.save('tests/fixtures/npy/real-numpy-f8.npy', np.array([1.5, -2.25, 0.0, 3.0, 42.125, 100.0625], dtype='<f8'))
 * ```
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const REAL_NUMPY_F4_PATH = resolve(HERE, '../fixtures/npy/real-numpy-f4.npy');
const REAL_NUMPY_F8_PATH = resolve(HERE, '../fixtures/npy/real-numpy-f8.npy');

describe('readNpyFloat32Array: real numpy.save fixtures', () => {
  it('decodes a real numpy.save <f4 array to the exact expected values', () => {
    const values = readNpyFloat32Array(REAL_NUMPY_F4_PATH);
    expect(Array.from(values)).toEqual([1.5, -2.25, 0, 3, 42.125]);
  });

  it('decodes a real numpy.save <f8 array to the exact expected values', () => {
    const values = readNpyFloat32Array(REAL_NUMPY_F8_PATH);
    expect(Array.from(values)).toEqual([1.5, -2.25, 0, 3, 42.125, 100.0625]);
  });
});

describe('writeNpyFloat32Array / readNpyFloat32Array: round-trip', () => {
  const roundTrip = (values: readonly number[]): number[] => {
    const dir = mkdtempSync(join(tmpdir(), 'npy-roundtrip-'));
    try {
      const path = resolve(dir, 'values.npy');
      writeNpyFloat32Array(path, Float32Array.from(values));
      return Array.from(readNpyFloat32Array(path));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  it.each([0, 1, 5, 6, 7, 100, 257])('round-trips a length-%d array exactly', (length) => {
    const values = Array.from({ length }, (_, i) => (i - length / 2) * 0.125);
    expect(roundTrip(values)).toEqual(Array.from(Float32Array.from(values)));
  });

  it('round-trips negative, zero, and fractional float32 values exactly', () => {
    const values = [-1.5, 0, 0.25, -0.000001, 123456.75, -42];
    expect(roundTrip(values)).toEqual(Array.from(Float32Array.from(values)));
  });

  it('writes a header prologue that is a multiple of 64 bytes, for lengths crossing the boundary', () => {
    for (const length of [0, 1, 5, 6, 7, 100, 257]) {
      const dir = mkdtempSync(join(tmpdir(), 'npy-header-'));
      try {
        const path = resolve(dir, 'values.npy');
        writeNpyFloat32Array(path, new Float32Array(length));
        const buffer = readFileSync(path);
        // v1.0 header-length field is 2 bytes LE at offset 8; prologue is
        // 10 + headerLength (magic + version + header-length field + header).
        const headerLength = buffer.readUInt16LE(8);
        expect((10 + headerLength) % 64).toBe(0);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
  });
});

describe('readNpyFloat32Array: rejection paths', () => {
  const writeAndRead = (dir: string, name: string, bytes: Buffer): Float32Array => {
    const path = resolve(dir, name);
    writeFileSync(path, bytes);
    return readNpyFloat32Array(path);
  };

  it('rejects bad magic bytes', () => {
    const dir = mkdtempSync(join(tmpdir(), 'npy-reject-'));
    try {
      const bytes = readFileSync(REAL_NUMPY_F4_PATH);
      const corrupted = Buffer.from(bytes);
      corrupted[0] = 0x00; // clobber the leading \x93 of the magic
      expect(() => writeAndRead(dir, 'bad-magic.npy', corrupted)).toThrow(/bad magic/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects a truncated header', () => {
    const dir = mkdtempSync(join(tmpdir(), 'npy-reject-'));
    try {
      const bytes = readFileSync(REAL_NUMPY_F4_PATH);
      // Keep the 10-byte prologue (magic + version + header-length field,
      // which claims a header far longer than what follows) but drop
      // everything after it.
      const truncated = bytes.subarray(0, 10);
      expect(() => writeAndRead(dir, 'truncated-header.npy', truncated)).toThrow(/truncated header/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects an unsupported dtype (<i4)', () => {
    const dir = mkdtempSync(join(tmpdir(), 'npy-reject-'));
    try {
      const bytes = readFileSync(REAL_NUMPY_F4_PATH);
      const mutated = Buffer.from(bytes.toString('latin1').replace("'<f4'", "'<i4'"), 'latin1');
      expect(() => writeAndRead(dir, 'bad-dtype.npy', mutated)).toThrow(/Unsupported \.npy dtype/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects fortran_order: True', () => {
    const dir = mkdtempSync(join(tmpdir(), 'npy-reject-'));
    try {
      const bytes = readFileSync(REAL_NUMPY_F4_PATH);
      const mutated = Buffer.from(
        bytes.toString('latin1').replace("'fortran_order': False", "'fortran_order': True"),
        'latin1'
      );
      expect(() => writeAndRead(dir, 'fortran-order.npy', mutated)).toThrow(/fortran_order must be False/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects a non-1-D shape', () => {
    const dir = mkdtempSync(join(tmpdir(), 'npy-reject-'));
    try {
      const bytes = readFileSync(REAL_NUMPY_F4_PATH);
      const mutated = Buffer.from(bytes.toString('latin1').replace('(5,)', '(5, 1)'), 'latin1');
      expect(() => writeAndRead(dir, 'bad-shape.npy', mutated)).toThrow(/expected a 1-D array/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
