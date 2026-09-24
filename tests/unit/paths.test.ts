import { describe, expect, it } from 'vitest';
import { publicAssetUrl } from '../../src/lib/paths';

/**
 * `publicAssetUrl` is the single seam every runtime-fetched public asset
 * URL (graph manifest/artifacts, ledger provenance links) must go through
 * instead of a hard-coded leading slash, so the app keeps working whether
 * it's served from the origin root or from a non-root deployment path like
 * `/fly/` (see `src/lib/paths.ts`'s doc comment and
 * `.agents/deployment.md`). `base` is exercised directly via the parameter
 * here rather than by mocking `import.meta.env`, which stays exercised
 * separately by the default-argument behavior asserted below (Vitest/jsdom
 * resolves `import.meta.env.BASE_URL` to `'/'`, matching every other
 * root-base test in this suite).
 */
describe('publicAssetUrl', () => {
  it('joins a relative path onto a root base ("/") with exactly one slash', () => {
    expect(publicAssetUrl('data', '/')).toBe('/data');
    expect(publicAssetUrl('data/malecns-arena-v1.manifest.json', '/')).toBe('/data/malecns-arena-v1.manifest.json');
  });

  it('joins a relative path onto a non-root base ("/fly/"), matching production', () => {
    expect(publicAssetUrl('data', '/fly/')).toBe('/fly/data');
    expect(publicAssetUrl('data/malecns-arena-v1.manifest.json', '/fly/')).toBe(
      '/fly/data/malecns-arena-v1.manifest.json'
    );
  });

  it('tolerates a leading slash on the relative path without producing a double slash', () => {
    expect(publicAssetUrl('/data', '/')).toBe('/data');
    expect(publicAssetUrl('/data', '/fly/')).toBe('/fly/data');
  });

  it('tolerates a base with no trailing slash (defensive; Vite itself always supplies one)', () => {
    expect(publicAssetUrl('data', '/fly')).toBe('/fly/data');
  });

  it('defaults to import.meta.env.BASE_URL when no base is given (\'/\' under Vitest/jsdom)', () => {
    expect(publicAssetUrl('data')).toBe('/data');
  });
});
