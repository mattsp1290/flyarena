import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import FindingsPanel from '../../src/lib/ui/FindingsPanel.svelte';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadRewiringNull, type RewiringNullArtifact, type RewiringNullLoadResult } from '../../src/lib/experiment/rewiringNull';
import { loadNullExplanation, type NullExplanationArtifact, type NullExplanationLoadResult } from '../../src/lib/experiment/nullExplanation';
import {
  loadPathwayInterventions,
  type PathwayInterventionsArtifact,
  type PathwayInterventionsLoadResult
} from '../../src/lib/experiment/pathwayInterventions';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP1 of `.agents/plans/findings-tour`: component coverage for
 * `FindingsPanel.svelte`'s ARIA stepper — collapsed by default,
 * `aria-expanded` toggles, `aria-current="step"` moves with Next/Previous
 * with focus landing on the new step's heading, the live region's text
 * updates, and the missing/unavailable/invalid status texts render per
 * step. Fixture data is loaded through the real loaders (not a bare
 * `JSON.parse`), for the same "raw JSON's `trained.perSeed` differs from
 * the validated `trained.perSeedCategory` shape" reason
 * `tests/unit/findings-steps.test.ts` documents.
 */

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as ArenaManifest;

let realRewiringNull: RewiringNullArtifact;
let realNullExplanation: NullExplanationArtifact;
let realPathwayInterventions: PathwayInterventionsArtifact;

beforeAll(async () => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  const rewiringNullResult = await loadRewiringNull(manifest, '/data');
  const nullExplanationResult = await loadNullExplanation(manifest, '/data');
  const pathwayInterventionsResult = await loadPathwayInterventions(manifest, '/data');
  if (rewiringNullResult.status !== 'ok') throw new Error(`Fixture setup: rewiringNull is "${rewiringNullResult.status}"`);
  if (nullExplanationResult.status !== 'ok') throw new Error(`Fixture setup: nullExplanation is "${nullExplanationResult.status}"`);
  if (pathwayInterventionsResult.status !== 'ok') {
    throw new Error(`Fixture setup: pathwayInterventions is "${pathwayInterventionsResult.status}"`);
  }
  realRewiringNull = rewiringNullResult.data;
  realNullExplanation = nullExplanationResult.data;
  realPathwayInterventions = pathwayInterventionsResult.data;
  vi.unstubAllGlobals();
});

const okProps = () => ({
  manifest,
  rewiringNull: { status: 'ok', data: realRewiringNull } as RewiringNullLoadResult,
  nullExplanation: { status: 'ok', data: realNullExplanation } as NullExplanationLoadResult,
  pathwayInterventions: { status: 'ok', data: realPathwayInterventions } as PathwayInterventionsLoadResult
});

describe('FindingsPanel', () => {
  it('is collapsed by default, with an accessible toggle', () => {
    render(FindingsPanel, okProps());
    const toggle = screen.getByRole('button', { name: /^expand$/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
  });

  it('expanding shows the seven-step list and flips aria-expanded', async () => {
    const { container } = render(FindingsPanel, okProps());
    const toggle = screen.getByRole('button', { name: /^expand$/i });
    await fireEvent.click(toggle);
    expect(screen.getByRole('button', { name: /^collapse$/i })).toHaveAttribute('aria-expanded', 'true');
    // Scoped to the top-level step `<li>`s specifically -- `getAllByRole('listitem')`
    // would also pick up each step's own nested `<ul class="provenance">` items.
    const items = container.querySelectorAll('ol.steps > li.step');
    expect(items).toHaveLength(7);
  });

  it('the header names the model framing constraint verbatim', async () => {
    render(FindingsPanel, okProps());
    expect(
      screen.getByRole('heading', { name: 'Findings under this model — not claims about the real fly' })
    ).toBeInTheDocument();
  });

  it('step 1 has aria-current="step" by default once expanded, and the live region announces "Step 1 of 7"', async () => {
    const { container } = render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    const items = container.querySelectorAll('ol.steps > li.step');
    expect(items[0]).toHaveAttribute('aria-current', 'step');
    for (const item of Array.from(items).slice(1)) expect(item).not.toHaveAttribute('aria-current');
    expect(screen.getByText('Step 1 of 7')).toBeInTheDocument();
  });

  it('Next moves aria-current to step 2, moves focus to its heading, and updates the live region', async () => {
    const { container } = render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await fireEvent.click(screen.getByRole('button', { name: /^next$/i }));
    const items = container.querySelectorAll('ol.steps > li.step');
    expect(items[0]).not.toHaveAttribute('aria-current');
    expect(items[1]).toHaveAttribute('aria-current', 'step');
    expect(screen.getByText('Step 2 of 7')).toBeInTheDocument();
    const heading = screen.getByRole('heading', { name: /2\. mirrored decoder/i });
    expect(heading).toHaveFocus();
  });

  it('Previous is disabled on step 1 and Next is disabled on step 7', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getByRole('button', { name: /^previous$/i })).toBeDisabled();
    for (let i = 0; i < 6; i += 1) {
      await fireEvent.click(screen.getByRole('button', { name: /^next$/i }));
    }
    expect(screen.getByText('Step 7 of 7')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^next$/i })).toBeDisabled();
  });

  it('Previous moves focus back to the prior step\'s heading', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await fireEvent.click(screen.getByRole('button', { name: /^next$/i }));
    await fireEvent.click(screen.getByRole('button', { name: /^previous$/i }));
    const heading = screen.getByRole('heading', { name: /1\. rewiring null/i });
    expect(heading).toHaveFocus();
    expect(screen.getByText('Step 1 of 7')).toBeInTheDocument();
  });

  it('step 7 (behavior repertoire) always shows "Not yet published" in this WP, with a link to the atlas', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getByText('Not yet published')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /behavior atlas/i })).toHaveAttribute('href', '#atlas');
  });

  it('an "unavailable" rewiringNull shows "Could not be loaded" with the honest reason, degrading only its dependent steps', async () => {
    render(FindingsPanel, {
      ...okProps(),
      rewiringNull: { status: 'unavailable', reason: 'network hiccup' } as RewiringNullLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    // Steps 1, 2, and 5 all depend on `rewiringNull` (step 2's mirrored-
    // decoder baseline needs it too) and degrade to the same honest status
    // line; steps 3, 4, and 6 do not depend on it and stay unaffected.
    expect(screen.getAllByText('Could not be loaded: network hiccup').length).toBe(3);
  });

  it('an "invalid" nullExplanation shows "Failed verification" with the honest reason', async () => {
    render(FindingsPanel, {
      ...okProps(),
      nullExplanation: { status: 'invalid', reason: 'sha256 mismatch' } as NullExplanationLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getAllByText(/Failed verification: sha256 mismatch/).length).toBeGreaterThan(0);
  });

  it('a "missing" pathwayInterventions shows "Not yet published" for steps 4 and 6', async () => {
    render(FindingsPanel, {
      ...okProps(),
      pathwayInterventions: {
        status: 'missing',
        reason: 'The manifest has no pathwayInterventions artifact entry.'
      } as PathwayInterventionsLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getAllByText('Not yet published').length).toBeGreaterThanOrEqual(2);
  });

  it('every provenance entry links the pinned JSON and the report, with the manifest sha256 prefix', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    const prefix = manifest.rewiringNull?.sha256.slice(0, 12);
    expect(screen.getAllByText(new RegExp(`sha256 ${prefix}`)).length).toBeGreaterThan(0);
    const jsonLinks = screen.getAllByRole('link', { name: 'Pinned JSON' });
    expect(jsonLinks.some((link) => link.getAttribute('href') === `/data/${manifest.rewiringNull?.artifact}`)).toBe(true);
    const reportLinks = screen.getAllByRole('link', { name: 'Report' });
    expect(
      reportLinks.some(
        (link) => link.getAttribute('href') === 'https://github.com/mattsp1290/flyarena/blob/main/docs/rewiring-null-report.md'
      )
    ).toBe(true);
  });
});
