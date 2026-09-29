import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { FINDING_SECTIONS, groupSteps } from '../../src/lib/findings/sections';
import { buildFindingSteps, type BuildFindingStepsInputs } from '../../src/lib/findings/steps';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadRewiringNull, type RewiringNullArtifact } from '../../src/lib/experiment/rewiringNull';
import { loadNullExplanation, type NullExplanationArtifact } from '../../src/lib/experiment/nullExplanation';
import { loadPathwayInterventions, type PathwayInterventionsArtifact } from '../../src/lib/experiment/pathwayInterventions';
import { loadRepertoireNull, type RepertoireNullArtifact } from '../../src/lib/experiment/repertoireNull';
import { loadTaskGenerality, type TaskGeneralityArtifact } from '../../src/lib/experiment/taskGenerality';
import { loadSelectionRobustness, type SelectionRobustnessArtifact } from '../../src/lib/experiment/selectionRobustness';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP1 of `.agents/plans/consolidated-release` (`01-findings-sections.md`):
 * unit coverage for `src/lib/findings/sections.ts`. The gate this plan
 * actually cares about: every id `buildFindingSteps` returns against the
 * real, committed `main` fixtures must map to a *named* `FINDING_SECTIONS`
 * entry, never silently landing in the "Other findings" fallback -- a new
 * step whose id nobody added to `FINDING_SECTIONS` fails this test instead
 * of vanishing into an unlabeled catch-all (`00-overview.md`'s "a unit test
 * fails if that section is non-empty on main").
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');
const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as ArenaManifest;

let realInputs: BuildFindingStepsInputs;

beforeAll(async () => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  const rewiringNull = await loadRewiringNull(manifest, '/data');
  const nullExplanation = await loadNullExplanation(manifest, '/data');
  const pathwayInterventions = await loadPathwayInterventions(manifest, '/data');
  const repertoireNull = await loadRepertoireNull(manifest, '/data');
  const taskGenerality = await loadTaskGenerality(manifest, '/data');
  const selectionRobustness = await loadSelectionRobustness(manifest, '/data');
  if (rewiringNull.status !== 'ok') throw new Error(`Fixture setup: rewiringNull is "${rewiringNull.status}"`);
  if (nullExplanation.status !== 'ok') throw new Error(`Fixture setup: nullExplanation is "${nullExplanation.status}"`);
  if (pathwayInterventions.status !== 'ok') throw new Error(`Fixture setup: pathwayInterventions is "${pathwayInterventions.status}"`);
  if (repertoireNull.status !== 'ok') throw new Error(`Fixture setup: repertoireNull is "${repertoireNull.status}"`);
  if (taskGenerality.status !== 'ok') throw new Error(`Fixture setup: taskGenerality is "${taskGenerality.status}"`);
  if (selectionRobustness.status !== 'ok') throw new Error(`Fixture setup: selectionRobustness is "${selectionRobustness.status}"`);
  realInputs = {
    manifest,
    dataBaseUrl: '/data',
    rewiringNull: { status: 'ok', data: rewiringNull.data as RewiringNullArtifact },
    nullExplanation: { status: 'ok', data: nullExplanation.data as NullExplanationArtifact },
    pathwayInterventions: { status: 'ok', data: pathwayInterventions.data as PathwayInterventionsArtifact },
    repertoireNull: { status: 'ok', data: repertoireNull.data as RepertoireNullArtifact },
    taskGenerality: { status: 'ok', data: taskGenerality.data as TaskGeneralityArtifact },
    selectionRobustness: { status: 'ok', data: selectionRobustness.data as SelectionRobustnessArtifact }
  };
  vi.unstubAllGlobals();
});

afterEach(() => vi.unstubAllGlobals());

describe('groupSteps against the real committed steps.ts fixtures', () => {
  it('classifies every step id from buildFindingSteps into a named section -- "Other findings" is empty', () => {
    const steps = buildFindingSteps(realInputs);
    const groups = groupSteps(steps);

    const otherGroup = groups.find((group) => group.section.id === 'other');
    expect(otherGroup).toBeUndefined();

    const groupedIds = groups.flatMap((group) => group.steps.map((step) => step.id));
    expect(new Set(groupedIds)).toEqual(new Set(steps.map((step) => step.id)));
    expect(groupedIds).toHaveLength(steps.length);
  });

  it('keeps FINDING_SECTIONS order, and each section keeps its own declared stepIds order', () => {
    const steps = buildFindingSteps(realInputs);
    const groups = groupSteps(steps);

    const nonEmptySectionIds = FINDING_SECTIONS.filter((section) =>
      section.stepIds.some((id) => steps.some((step) => step.id === id))
    ).map((section) => section.id);
    expect(groups.map((group) => group.section.id)).toEqual(nonEmptySectionIds);

    for (const section of FINDING_SECTIONS) {
      const group = groups.find((candidate) => candidate.section.id === section.id);
      if (!group) continue;
      const expectedOrder = section.stepIds.filter((id) => steps.some((step) => step.id === id));
      expect(group.steps.map((step) => step.id)).toEqual(expectedOrder);
    }
  });

  it('omits a section with none of its declared ids present, rather than rendering it empty', () => {
    const groups = groupSteps([{ id: 'rewiring-null' }]);
    expect(groups).toHaveLength(1);
    expect(groups[0].section.id).toBe('wiring-special');
    expect(groups.some((group) => group.section.id === 'scores-low')).toBe(false);
  });

  it('an id with no FINDING_SECTIONS entry falls into "Other findings", appended last', () => {
    const groups = groupSteps([{ id: 'rewiring-null' }, { id: 'some-unclassified-future-step' }, { id: 'intervention' }]);
    const other = groups.at(-1);
    expect(other?.section).toEqual({ id: 'other', title: 'Other findings' });
    expect(other?.steps.map((step) => step.id)).toEqual(['some-unclassified-future-step']);
  });

  it('preserves the input order of multiple unclassified ids inside "Other findings"', () => {
    const groups = groupSteps([{ id: 'zzz-unknown' }, { id: 'aaa-unknown' }]);
    expect(groups).toHaveLength(1);
    expect(groups[0].section.id).toBe('other');
    expect(groups[0].steps.map((step) => step.id)).toEqual(['zzz-unknown', 'aaa-unknown']);
  });

  it('returns no groups at all for an empty step list', () => {
    expect(groupSteps([])).toEqual([]);
  });

  it('a step id present twice in the input renders only once, at its first-declared location, never duplicated', () => {
    // Guards against a future FINDING_SECTIONS edit that lists a step id in
    // two sections, or twice within one section's own stepIds (both beans
    // in flight, `flyarena-9mls`/`flyarena-hd0j`, will independently edit
    // this file per the "merge-order rule") -- a duplicate would otherwise
    // produce a duplicate Svelte `{#each (step.id)}` key and a duplicate
    // DOM heading id in `FindingsPanel.svelte`.
    const groups = groupSteps([{ id: 'rewiring-null' }, { id: 'mirrored-decoder' }, { id: 'rewiring-null' }]);
    const allIds = groups.flatMap((group) => group.steps.map((step) => step.id));
    expect(allIds).toEqual(['rewiring-null', 'mirrored-decoder']);
  });
});
