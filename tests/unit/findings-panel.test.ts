import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/svelte';
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
import { loadRepertoireNull, type RepertoireNullArtifact, type RepertoireNullLoadResult } from '../../src/lib/experiment/repertoireNull';
import { loadTaskGenerality, type TaskGeneralityArtifact, type TaskGeneralityLoadResult } from '../../src/lib/experiment/taskGenerality';
import {
  loadSelectionRobustness,
  type SelectionRobustnessArtifact,
  type SelectionRobustnessLoadResult
} from '../../src/lib/experiment/selectionRobustness';
import {
  loadReadoutAttribution,
  type ReadoutAttributionArtifact,
  type ReadoutAttributionLoadResult
} from '../../src/lib/experiment/readoutAttribution';
import { buildFindingSteps } from '../../src/lib/findings/steps';
import { groupSteps } from '../../src/lib/findings/sections';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP1 of `.agents/plans/findings-tour`, grouped into question sections by
 * WP1 of `.agents/plans/consolidated-release` (`01-findings-sections.md`):
 * component coverage for `FindingsPanel.svelte`'s ARIA stepper/accordion —
 * collapsed by default, `aria-expanded` toggles, each question section is
 * its own `<h3>`-wrapped `<button>` accordion toggle (open by default,
 * `aria-expanded`/`aria-controls`), `aria-current="step"` moves with
 * Next/Previous across sections with focus landing on the new step's
 * heading (expanding a collapsed section first), the live region's text
 * updates with the current section's title, and the missing/unavailable/
 * invalid status texts render per step. Fixture data is loaded through the
 * real loaders (not a bare `JSON.parse`), for the same "raw JSON's
 * `trained.perSeed` differs from the validated `trained.perSeedCategory`
 * shape" reason `tests/unit/findings-steps.test.ts` documents.
 *
 * Grouping reorders steps for display (`01-findings-sections.md`'s own
 * "recompute every position-based assertion against the flattened grouped
 * order"): assertions here select by `data-step-id` rather than position
 * wherever practical, and the few genuinely position-based assertions (the
 * first/last step in the flattened walk) derive their expectations from
 * `groupSteps(buildFindingSteps(...))` directly rather than a hard-coded
 * literal, so they can never silently go stale when a step lands or moves.
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
let realRepertoireNull: RepertoireNullArtifact;
let realTaskGenerality: TaskGeneralityArtifact;
let realSelectionRobustness: SelectionRobustnessArtifact;
let realReadoutAttribution: ReadoutAttributionArtifact;

beforeAll(async () => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  const rewiringNullResult = await loadRewiringNull(manifest, '/data');
  const nullExplanationResult = await loadNullExplanation(manifest, '/data');
  const pathwayInterventionsResult = await loadPathwayInterventions(manifest, '/data');
  const repertoireNullResult = await loadRepertoireNull(manifest, '/data');
  const taskGeneralityResult = await loadTaskGenerality(manifest, '/data');
  const selectionRobustnessResult = await loadSelectionRobustness(manifest, '/data');
  const readoutAttributionResult = await loadReadoutAttribution(manifest, '/data');
  if (rewiringNullResult.status !== 'ok') throw new Error(`Fixture setup: rewiringNull is "${rewiringNullResult.status}"`);
  if (nullExplanationResult.status !== 'ok') throw new Error(`Fixture setup: nullExplanation is "${nullExplanationResult.status}"`);
  if (pathwayInterventionsResult.status !== 'ok') {
    throw new Error(`Fixture setup: pathwayInterventions is "${pathwayInterventionsResult.status}"`);
  }
  if (repertoireNullResult.status !== 'ok') throw new Error(`Fixture setup: repertoireNull is "${repertoireNullResult.status}"`);
  if (taskGeneralityResult.status !== 'ok') throw new Error(`Fixture setup: taskGenerality is "${taskGeneralityResult.status}"`);
  if (selectionRobustnessResult.status !== 'ok') {
    throw new Error(`Fixture setup: selectionRobustness is "${selectionRobustnessResult.status}"`);
  }
  if (readoutAttributionResult.status !== 'ok') {
    throw new Error(`Fixture setup: readoutAttribution is "${readoutAttributionResult.status}"`);
  }
  realRewiringNull = rewiringNullResult.data;
  realNullExplanation = nullExplanationResult.data;
  realPathwayInterventions = pathwayInterventionsResult.data;
  realRepertoireNull = repertoireNullResult.data;
  realTaskGenerality = taskGeneralityResult.data;
  realSelectionRobustness = selectionRobustnessResult.data;
  realReadoutAttribution = readoutAttributionResult.data;
  vi.unstubAllGlobals();
});

const okProps = () => ({
  manifest,
  rewiringNull: { status: 'ok', data: realRewiringNull } as RewiringNullLoadResult,
  nullExplanation: { status: 'ok', data: realNullExplanation } as NullExplanationLoadResult,
  pathwayInterventions: { status: 'ok', data: realPathwayInterventions } as PathwayInterventionsLoadResult,
  repertoireNull: { status: 'ok', data: realRepertoireNull } as RepertoireNullLoadResult,
  taskGenerality: { status: 'ok', data: realTaskGenerality } as TaskGeneralityLoadResult,
  selectionRobustness: { status: 'ok', data: realSelectionRobustness } as SelectionRobustnessLoadResult,
  readoutAttribution: { status: 'ok', data: realReadoutAttribution } as ReadoutAttributionLoadResult
});

/**
 * The exact flattened, grouped order `FindingsPanel.svelte` renders the
 * real fixtures in -- one `{ id, sectionTitle }` entry per step, computed
 * from the same `buildFindingSteps`/`groupSteps` the component itself
 * calls, never a hand-copied literal. Every position-based assertion below
 * reads its expectation from this, so it tracks the real grouped order
 * automatically if a future bean adds or reclassifies a step.
 */
const expectedFlatSteps = (): readonly { id: string; sectionTitle: string }[] =>
  groupSteps(buildFindingSteps({ ...okProps(), dataBaseUrl: '/data' })).flatMap((group) =>
    group.steps.map((step) => ({ id: step.id, sectionTitle: group.section.title }))
  );

const stepLocator = (id: string): HTMLElement => document.querySelector(`li.step[data-step-id="${id}"]`) as HTMLElement;

describe('FindingsPanel', () => {
  it('is collapsed by default, with an accessible toggle', () => {
    render(FindingsPanel, okProps());
    const toggle = screen.getByRole('button', { name: /^expand$/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
  });

  it('expanding renders every step, each carrying data-step-id matching its own step id, and flips aria-expanded', async () => {
    render(FindingsPanel, okProps());
    const toggle = screen.getByRole('button', { name: /^expand$/i });
    await fireEvent.click(toggle);
    expect(screen.getByRole('button', { name: /^collapse$/i })).toHaveAttribute('aria-expanded', 'true');
    const expected = expectedFlatSteps();
    const items = document.querySelectorAll('ol.steps > li.step');
    expect(items).toHaveLength(expected.length);
    for (const { id } of expected) {
      const item = stepLocator(id);
      expect(item).toBeInTheDocument();
      expect(item).toHaveAttribute('data-step-id', id);
    }
  });

  it('the header names the model framing constraint verbatim', async () => {
    render(FindingsPanel, okProps());
    expect(
      screen.getByRole('heading', { name: 'Findings under this model — not claims about the real fly' })
    ).toBeInTheDocument();
  });

  it('renders one accordion section per non-empty group, open by default, each with a distinct accessible toggle name and aria-controls pointing at its own <ol>', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));

    const expectedSectionTitles = [...new Set(expectedFlatSteps().map((step) => step.sectionTitle))];
    expect(expectedSectionTitles.length).toBeGreaterThan(0);

    for (const title of expectedSectionTitles) {
      const sectionToggle = screen.getByRole('button', { name: title });
      expect(sectionToggle).toHaveAttribute('aria-expanded', 'true');
      const controlsId = sectionToggle.getAttribute('aria-controls');
      expect(controlsId).toBeTruthy();
      const controlled = document.getElementById(controlsId as string);
      expect(controlled).toBeInTheDocument();
      expect(controlled?.tagName).toBe('OL');
    }

    // "Other findings" never appears while every shipped step id is classified.
    expect(screen.queryByRole('button', { name: 'Other findings' })).not.toBeInTheDocument();
  });

  it('collapsing a section hides its steps and flips aria-expanded to false; expanding it again restores them', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));

    const firstSectionTitle = expectedFlatSteps()[0].sectionTitle;
    const sectionToggle = screen.getByRole('button', { name: firstSectionTitle });
    const controlsId = sectionToggle.getAttribute('aria-controls') as string;

    await fireEvent.click(sectionToggle);
    expect(sectionToggle).toHaveAttribute('aria-expanded', 'false');
    expect(document.getElementById(controlsId)).not.toBeInTheDocument();
    // (accessibility review, Important) `aria-controls` must never point at
    // an id that doesn't exist in the document -- omitted entirely while
    // collapsed, not left pointing at the now-unmounted `<ol>`.
    expect(sectionToggle).not.toHaveAttribute('aria-controls');

    await fireEvent.click(sectionToggle);
    expect(sectionToggle).toHaveAttribute('aria-expanded', 'true');
    expect(document.getElementById(controlsId)).toBeInTheDocument();
    expect(sectionToggle).toHaveAttribute('aria-controls', controlsId);
  });

  it('collapsing the section that owns the current step relocates the cursor to the nearest visible step, so Previous/Next never disappear', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));

    const expected = expectedFlatSteps();
    const firstSectionTitle = expected[0].sectionTitle;
    const firstSectionToggle = screen.getByRole('button', { name: firstSectionTitle });

    // (dual review, Critical/Important) Collapsing the section holding
    // `currentIndex` (the first step, by default) used to unmount the only
    // Previous/Next controls in the panel, with no other way to move the
    // stepper. Collapsing it now relocates the cursor to the nearest step
    // that stays visible, so exactly one Previous and one Next button keep
    // existing somewhere in the panel.
    await fireEvent.click(firstSectionToggle);
    expect(firstSectionToggle).toHaveAttribute('aria-expanded', 'false');

    expect(screen.getByRole('button', { name: /^previous$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^next$/i })).toBeInTheDocument();

    // The new current step is the first one outside the now-collapsed
    // first section.
    const relocatedTo = expected.find((entry) => entry.sectionTitle !== firstSectionTitle);
    if (!relocatedTo) throw new Error('Fixture has only one section; cannot test relocation');
    expect(stepLocator(relocatedTo.id)).toHaveAttribute('aria-current', 'step');
    for (const { id, sectionTitle } of expected) {
      if (sectionTitle === firstSectionTitle) continue;
      if (id === relocatedTo.id) continue;
      expect(stepLocator(id)).not.toHaveAttribute('aria-current');
    }
  });

  it('collapsing every section in sequence never reaches zero visible steps: the last remaining open section refuses to collapse, and the live region stays accurate', async () => {
    // (thermo accessibility review, Important I1) An entirely ordinary
    // Tab+Enter/Space accordion walk -- collapsing all four sections, left
    // to right -- used to leave zero visible steps, zero Previous/Next
    // controls anywhere in the panel, and a stale "Step N of M in
    // <section>" live-region announcement describing content that no
    // longer exists (a WCAG 4.1.3 Status Messages violation). The last
    // collapse is now a no-op instead: at least one section, and therefore
    // exactly one current step with working Previous/Next, always remains.
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));

    const sectionTitles = [...new Set(expectedFlatSteps().map((step) => step.sectionTitle))];
    expect(sectionTitles.length).toBeGreaterThan(1);

    for (const title of sectionTitles.slice(0, -1)) {
      await fireEvent.click(screen.getByRole('button', { name: title }));
    }

    // Every section but the last is now collapsed; the last one still open
    // holds the (possibly relocated) current step.
    const lastTitle = sectionTitles[sectionTitles.length - 1];
    const lastToggle = screen.getByRole('button', { name: lastTitle });
    expect(lastToggle).toHaveAttribute('aria-expanded', 'true');

    const liveRegionTextBeforeFinalClick = screen.getByText(/^Step \d+ of \d+ in /).textContent;

    // Attempting to collapse the last remaining open section is a no-op.
    await fireEvent.click(lastToggle);
    expect(lastToggle).toHaveAttribute('aria-expanded', 'true');

    // At least one visible step (the last section's own steps), with
    // working Previous/Next, still exists.
    const visibleSteps = document.querySelectorAll('ol.steps > li.step');
    expect(visibleSteps.length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /^previous$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^next$/i })).toBeInTheDocument();

    // The live region still describes the one visible step accurately --
    // unchanged by the refused (no-op) collapse attempt, never stale.
    expect(screen.getByText(liveRegionTextBeforeFinalClick as string)).toBeInTheDocument();
  });

  it('aria-disabled appears only on the last open section\'s toggle (never the native disabled attribute), and clears reactively once another section reopens', async () => {
    // (coordinator follow-up to thermo accessibility review I1) The APG's
    // own "always one panel expanded" accordion variant names
    // `aria-disabled="true"` for exactly this situation -- a toggle whose
    // own collapse is refused. Verified here as its own regression,
    // independent of the "collapsing every section" scenario above: the
    // attribute must track section state reactively (present only while
    // this really is the last section with a visible step), and the button
    // must stay in the tab order throughout (real `disabled` would remove
    // it).
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));

    const sectionTitles = [...new Set(expectedFlatSteps().map((step) => step.sectionTitle))];
    expect(sectionTitles.length).toBeGreaterThan(1);

    // Every section starts open -- none is aria-disabled yet.
    for (const title of sectionTitles) {
      expect(screen.getByRole('button', { name: title })).not.toHaveAttribute('aria-disabled');
    }

    // Collapse every section but the last.
    for (const title of sectionTitles.slice(0, -1)) {
      await fireEvent.click(screen.getByRole('button', { name: title }));
    }

    const lastTitle = sectionTitles[sectionTitles.length - 1];
    const lastToggle = screen.getByRole('button', { name: lastTitle });
    expect(lastToggle).toHaveAttribute('aria-disabled', 'true');
    // Still a real, focusable, enabled button -- aria-disabled communicates
    // the refusal to assistive tech without removing it from the tab order.
    expect(lastToggle).not.toBeDisabled();
    lastToggle.focus();
    expect(lastToggle).toHaveFocus();

    // Reopening an earlier section clears aria-disabled from the last
    // toggle (its own collapse would no longer leave nothing visible), and
    // the reopened section's own toggle is not disabled either.
    const firstCollapsedTitle = sectionTitles[0];
    const reopenedToggle = screen.getByRole('button', { name: firstCollapsedTitle });
    await fireEvent.click(reopenedToggle);
    expect(lastToggle).not.toHaveAttribute('aria-disabled');
    expect(reopenedToggle).not.toHaveAttribute('aria-disabled');
  });

  it('the first step has aria-current="step" by default once expanded, and the live region announces it, with its section', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    const [first, ...rest] = expectedFlatSteps();
    expect(stepLocator(first.id)).toHaveAttribute('aria-current', 'step');
    for (const { id } of rest) expect(stepLocator(id)).not.toHaveAttribute('aria-current');
    expect(screen.getByText(`Step 1 of ${expectedFlatSteps().length} in ${first.sectionTitle}`)).toBeInTheDocument();
  });

  it('Next moves aria-current to the second step in the grouped order, moves focus to its heading, and updates the live region', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await fireEvent.click(screen.getByRole('button', { name: /^next$/i }));
    const [first, second] = expectedFlatSteps();
    expect(stepLocator(first.id)).not.toHaveAttribute('aria-current');
    expect(stepLocator(second.id)).toHaveAttribute('aria-current', 'step');
    expect(screen.getByText(`Step 2 of ${expectedFlatSteps().length} in ${second.sectionTitle}`)).toBeInTheDocument();
    const heading = within(stepLocator(second.id)).getByRole('heading', { level: 4 });
    expect(heading).toHaveFocus();
  });

  it('Previous is disabled on the first step and Next is disabled on the last step of the grouped order', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    const expected = expectedFlatSteps();
    expect(screen.getByRole('button', { name: /^previous$/i })).toBeDisabled();
    for (let i = 0; i < expected.length - 1; i += 1) {
      await fireEvent.click(screen.getByRole('button', { name: /^next$/i }));
    }
    const last = expected[expected.length - 1];
    expect(screen.getByText(`Step ${expected.length} of ${expected.length} in ${last.sectionTitle}`)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^next$/i })).toBeDisabled();
  });

  it("Previous moves focus back to the prior step's heading", async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await fireEvent.click(screen.getByRole('button', { name: /^next$/i }));
    await fireEvent.click(screen.getByRole('button', { name: /^previous$/i }));
    const [first] = expectedFlatSteps();
    const heading = within(stepLocator(first.id)).getByRole('heading', { level: 4 });
    expect(heading).toHaveFocus();
    expect(screen.getByText(`Step 1 of ${expectedFlatSteps().length} in ${first.sectionTitle}`)).toBeInTheDocument();
  });

  it('Next into a step whose section was collapsed expands that section, so the new current step is actually visible', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    const expected = expectedFlatSteps();
    const secondSectionFirstStep = expected.find((step) => step.sectionTitle !== expected[0].sectionTitle);
    if (!secondSectionFirstStep) throw new Error('Fixture has only one section; cannot test cross-section navigation');

    // Collapse the second section before walking into it.
    const secondSectionToggle = screen.getByRole('button', { name: secondSectionFirstStep.sectionTitle });
    await fireEvent.click(secondSectionToggle);
    expect(secondSectionToggle).toHaveAttribute('aria-expanded', 'false');

    // Walk Next until the current step's section changes.
    let current = expected[0];
    while (current.sectionTitle === expected[0].sectionTitle) {
      await fireEvent.click(screen.getByRole('button', { name: /^next$/i }));
      const index = expected.findIndex((step) => step.id === current.id) + 1;
      current = expected[index];
    }

    expect(secondSectionToggle).toHaveAttribute('aria-expanded', 'true');
    expect(stepLocator(current.id)).toBeInTheDocument();
    expect(stepLocator(current.id)).toHaveAttribute('aria-current', 'step');
  });

  it('the task-generality step states the real overall verdicts in a short summary, plus a per-task <ul> list', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    const step = stepLocator('task-generality');
    const sentence = step.querySelector('p.sentence') as HTMLElement;
    expect(sentence.textContent).toMatch(new RegExp(`authored: ${realTaskGenerality.overall.authored.verdict}`));
    expect(sentence.textContent).toMatch(new RegExp(`trained: ${realTaskGenerality.overall.trained.verdict}`));
    expect(sentence.textContent).toMatch(/under this model\.$/);
    // Thermo review (Important, both reviewers): per-task detail moved out
    // of the sentence into its own screen-reader-friendly <ul>/<li> list.
    const perTaskItems = step.querySelectorAll('ul.detail-list > li');
    expect(perTaskItems).toHaveLength(realTaskGenerality.tasks.length);
    const itemTexts = Array.from(perTaskItems).map((li) => li.textContent ?? '');
    for (const task of realTaskGenerality.tasks) {
      expect(itemTexts.some((text) => text.includes(task.id))).toBe(true);
    }
  });

  it('the selection-robustness step states the real overall verdicts in a short summary, plus a per-selection <ul> list', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    const step = stepLocator('selection-robustness');
    const sentence = step.querySelector('p.sentence') as HTMLElement;
    expect(sentence.textContent).toMatch(new RegExp(`robust to size is ${realSelectionRobustness.overall.robustToSize.verdict}`));
    expect(sentence.textContent).toMatch(new RegExp(`robust to method is ${realSelectionRobustness.overall.robustToMethod.verdict}`));
    expect(sentence.textContent).toMatch(new RegExp(`the channel-mapping result is ${realSelectionRobustness.overall.mapping.verdict}`));
    expect(sentence.textContent).toMatch(/under this model\.$/);
    const perSelectionItems = step.querySelectorAll('ul.detail-list > li');
    expect(perSelectionItems).toHaveLength(realSelectionRobustness.selections.length);
    const itemTexts = Array.from(perSelectionItems).map((li) => li.textContent ?? '');
    for (const selection of realSelectionRobustness.selections) {
      expect(itemTexts.some((text) => text.includes(selection.id))).toBe(true);
    }
  });

  it('the selection-robustness step shows "Not yet published" when the artifact is missing', async () => {
    render(FindingsPanel, {
      ...okProps(),
      selectionRobustness: {
        status: 'missing',
        reason: 'The manifest has no selectionRobustness artifact entry.'
      } as SelectionRobustnessLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getByText('Not yet published')).toBeInTheDocument();
  });

  it('the task-generality step shows "Not yet published" when the task-generality artifact is missing', async () => {
    render(FindingsPanel, {
      ...okProps(),
      taskGenerality: { status: 'missing', reason: 'The manifest has no taskGenerality artifact entry.' } as TaskGeneralityLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getByText('Not yet published')).toBeInTheDocument();
  });

  it('the behavior-repertoire step states the real category and links to the atlas', async () => {
    render(FindingsPanel, okProps());
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getByText(new RegExp(`biological occupies ${realRepertoireNull.primary.bio.occupied} of 36`))).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /behavior atlas/i })).toHaveAttribute('href', '#atlas');
  });

  it('the behavior-repertoire step shows "Not yet published" when the repertoire-null artifact is missing', async () => {
    render(FindingsPanel, {
      ...okProps(),
      repertoireNull: {
        status: 'missing',
        reason: 'The manifest has no behaviorRepertoireNull artifact entry.'
      } as RepertoireNullLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(screen.getByText('Not yet published')).toBeInTheDocument();
    expect(screen.getByText(/Once published, this step will compare/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /behavior atlas/i })).toHaveAttribute('href', '#atlas');
  });

  it('an "unavailable" rewiringNull shows "Could not be loaded" with the honest reason, degrading only its dependent steps', async () => {
    render(FindingsPanel, {
      ...okProps(),
      rewiringNull: { status: 'unavailable', reason: 'network hiccup' } as RewiringNullLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    // The rewiring-null, mirrored-decoder, and trained-null steps all
    // depend on rewiringNull (mirrored-decoder's baseline needs it too) and
    // degrade to the same honest status line; explanation, intervention,
    // and trained-interventions do not depend on it and stay unaffected.
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

  it('a "missing" pathwayInterventions shows "Not yet published" for the intervention and trained-interventions steps', async () => {
    render(FindingsPanel, {
      ...okProps(),
      pathwayInterventions: {
        status: 'missing',
        reason: 'The manifest has no pathwayInterventions artifact entry.'
      } as PathwayInterventionsLoadResult
    });
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(stepLocator('intervention')).toHaveTextContent('Not yet published');
    expect(stepLocator('trained-interventions')).toHaveTextContent('Not yet published');
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
