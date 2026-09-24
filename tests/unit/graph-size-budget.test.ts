import { readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/**
 * WP7 performance gate: the checked-in compressed graph payload must fit the
 * plan's declared repository/runtime budget ("<= 5 MB compressed" —
 * `.agents/plans/flyarena-plan-9yhp-3d-connectome-arena-poc/sections/implementation.md`'s
 * "Chosen architecture" data-budget note). "Compressed" means every gzip
 * artifact the client actually downloads over the wire to run one
 * experiment — the biological artifact plus every rewired-control arm
 * declared in the manifest — not just the biological artifact alone.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const FIVE_MEGABYTES = 5 * 1024 * 1024;

interface Manifest {
  artifact: string;
  gzipBytes: number;
  rewiredArms: Record<string, { artifact: string; gzipBytes: number }>;
}

describe('compressed graph size budget (<= 5 MB, plan "Chosen architecture" data budget)', () => {
  const manifest = JSON.parse(
    readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
  ) as Manifest;

  it('the biological artifact alone fits well within the 5 MB budget', () => {
    const bytes = statSync(resolve(publicDataDir, manifest.artifact)).size;
    expect(bytes).toBe(manifest.gzipBytes);
    expect(bytes).toBeLessThanOrEqual(FIVE_MEGABYTES);
  });

  it('every rewired-control artifact fits well within the 5 MB budget', () => {
    for (const entry of Object.values(manifest.rewiredArms)) {
      const bytes = statSync(resolve(publicDataDir, entry.artifact)).size;
      expect(bytes).toBe(entry.gzipBytes);
      expect(bytes).toBeLessThanOrEqual(FIVE_MEGABYTES);
    }
  });

  it('the total compressed payload for one full experiment (biological + every rewired arm) fits within the 5 MB budget', () => {
    const rewiredTotal = Object.values(manifest.rewiredArms).reduce((sum, entry) => sum + entry.gzipBytes, 0);
    const totalGzipBytes = manifest.gzipBytes + rewiredTotal;
    // eslint-disable-next-line no-console -- perf-gate visibility: report the measured figure alongside the assertion, not just pass/fail.
    console.log(
      `[perf] total compressed graph payload: ${totalGzipBytes} bytes (${(totalGzipBytes / (1024 * 1024)).toFixed(3)} MB) of a 5 MB budget`
    );
    expect(totalGzipBytes).toBeLessThanOrEqual(FIVE_MEGABYTES);
  });
});
