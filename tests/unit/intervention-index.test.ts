import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { copyInterventionIndex } from '../../scripts/attribution/intervention-index';

/**
 * Coverage for `scripts/attribution/intervention-index.ts` (split out of
 * `archive-readouts.ts` -- a thermo-maintainability review finding).
 *
 * `copyInterventionIndex` validates with `graph-list-index.ts`'s own
 * `readGraphListIndex` (a thermo-maintainability review finding: the
 * previous version only checked `sourceSha256`/non-empty `entries`, missing
 * entry-shape, reserved-id, duplicate-id, and path-traversal validation that
 * `readGraphListIndex` already provides) -- several tests below exercise
 * exactly the checks that ad hoc version was missing.
 */

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'intervention-index-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const validPayload = {
  sourceArtifact: 'public/data/malecns-arena-v1.bin.gz',
  sourceSha256: 'a'.repeat(64),
  entries: [{ id: 'P', path: 'graphs/P.bin.gz', gzipSha256: 'b'.repeat(64), binarySha256: 'c'.repeat(64) }]
};

describe('copyInterventionIndex', () => {
  it('copies a well-formed graph-list index.json', () => {
    const src = resolve(root, 'index.json');
    const out = resolve(root, 'archive', 'intervention-index-v1.json');
    writeFileSync(src, JSON.stringify(validPayload));

    copyInterventionIndex(src, out);
    expect(JSON.parse(readFileSync(out, 'utf8'))).toEqual(validPayload);
  });

  it('copies non-canonical (pretty-printed) input byte-for-byte, not re-encoded', () => {
    const src = resolve(root, 'pretty-index.json');
    const out = resolve(root, 'archive', 'pretty-index-out.json');
    const text = `${JSON.stringify(validPayload, null, 2)}\n`;
    writeFileSync(src, text);

    copyInterventionIndex(src, out);
    expect(readFileSync(out)).toEqual(readFileSync(src));
  });

  it('refuses a file with no entries', () => {
    const src = resolve(root, 'bad-index.json');
    writeFileSync(src, JSON.stringify({ sourceArtifact: 'x', sourceSha256: 'a'.repeat(64), entries: [] }));
    expect(() => copyInterventionIndex(src, resolve(root, 'out.json'))).toThrow(/has no entries/);
  });

  it('refuses a file missing sourceArtifact/sourceSha256', () => {
    const src = resolve(root, 'no-source.json');
    writeFileSync(src, JSON.stringify({ entries: validPayload.entries }));
    expect(() => copyInterventionIndex(src, resolve(root, 'out.json'))).toThrow(/missing sourceArtifact\/sourceSha256/);
  });

  it('refuses an entry using a reserved graph id ("biological"/"disconnected")', () => {
    const src = resolve(root, 'reserved-id.json');
    writeFileSync(
      src,
      JSON.stringify({
        ...validPayload,
        entries: [{ id: 'biological', path: 'graphs/biological.bin.gz', gzipSha256: 'b'.repeat(64), binarySha256: 'c'.repeat(64) }]
      })
    );
    expect(() => copyInterventionIndex(src, resolve(root, 'out.json'))).toThrow(/reserved graph id/);
  });

  it('refuses a duplicate id across entries', () => {
    const src = resolve(root, 'dup-id.json');
    writeFileSync(
      src,
      JSON.stringify({
        ...validPayload,
        entries: [validPayload.entries[0], { ...validPayload.entries[0] }]
      })
    );
    expect(() => copyInterventionIndex(src, resolve(root, 'out.json'))).toThrow(/more than once/);
  });

  it('refuses a path that escapes the index\'s own directory', () => {
    const src = resolve(root, 'traversal.json');
    writeFileSync(
      src,
      JSON.stringify({
        ...validPayload,
        entries: [{ id: 'P', path: '../outside.bin.gz', gzipSha256: 'b'.repeat(64), binarySha256: 'c'.repeat(64) }]
      })
    );
    expect(() => copyInterventionIndex(src, resolve(root, 'out.json'))).toThrow(/path outside index\.json's own directory/);
  });
});
