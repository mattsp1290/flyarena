<script lang="ts">
  import { tick } from 'svelte';
  import type { ArenaManifest } from '../experiment/assets';
  import type { RewiringNullLoadResult } from '../experiment/rewiringNull';
  import type { NullExplanationLoadResult } from '../experiment/nullExplanation';
  import type { PathwayInterventionsLoadResult } from '../experiment/pathwayInterventions';
  import type { RepertoireNullLoadResult } from '../experiment/repertoireNull';
  import type { TaskGeneralityLoadResult } from '../experiment/taskGenerality';
  import type { SelectionRobustnessLoadResult } from '../experiment/selectionRobustness';
  import type { ReadoutAttributionLoadResult } from '../experiment/readoutAttribution';
  import { buildFindingSteps, findingStepStatusLabel, type FindingStep } from '../findings/steps';
  import { groupSteps } from '../findings/sections';
  import { hasAnyVisibleStep, nearestVisibleIndex } from '../findings/navigation';

  /**
   * WP1 of `.agents/plans/findings-tour` (`01-findings-panel.md`), grouped
   * into question sections by WP1 of `.agents/plans/consolidated-release`
   * (`01-findings-sections.md`): a collapsible ARIA stepper that walks a
   * visitor through the evidence chain, each step templated from an already
   * fetched/sha256-verified/shape-validated artifact
   * (`src/lib/findings/steps.ts#buildFindingSteps`) — never a hard-coded
   * number. Framing (user constraint, no biological claims): the header
   * reads "Findings under this model — not claims about the real fly," and
   * every templated sentence names its decoder condition and ends with
   * "under this model."
   *
   * Collapsed by default, like `ActivityPanel.svelte`'s own "expand to see
   * more" convention — it never blocks the arena or Start. Placed directly
   * above `LedgerPanel` in `App.svelte`'s sidebar; reuses the controller's
   * already-mirrored `$state` load results (props from `App.svelte`), the
   * same "one fetch and verification per artifact, no re-fetching inside
   * the panel" discipline `LedgerPanel`/`NullExplanationNote` already
   * follow.
   *
   * Once the panel itself is expanded, the steps are grouped into
   * collapsible question sections (`../findings/sections.ts#groupSteps`),
   * each open by default (`01-findings-sections.md`'s "matches today's
   * behavior where all steps are visible once the panel is expanded"),
   * following the WAI-ARIA APG accordion pattern: an `<h3>` wraps each
   * section's own toggle `<button>`, with `aria-expanded`/`aria-controls`
   * pointing at that section's `<ol>`. Every one of the `<li>` steps is
   * always rendered with its own full content (sentence, provenance) once
   * its section is open — nothing is hidden behind a wizard-style
   * single-step view, so the tampering/degradation coverage in
   * `tests/e2e/findings.spec.ts` can assert every step's state at once, and
   * a screen-reader user is never forced through Next clicks to reach a
   * later step's content.
   *
   * "Current step" (`aria-current="step"`) is a keyboard-walking cursor
   * only, over the flattened grouped order (section order, then each
   * section's own declared step order — never `steps.ts`'s own array
   * order, which no longer matches display): Previous/Next move which step
   * has it, expanding that step's section first if it is collapsed, and
   * move DOM focus to that step's own heading. A polite live region
   * announces "Step N of M in <section title>" (`flatSteps.length`, never a
   * hard-coded literal) on every move — the accessible way to walk the
   * chain in order without requiring a screen-reader user to tab through
   * every earlier step's worth of links first.
   *
   * Collapsing a section always leaves at least one step visible
   * (`toggleSection`, using `../findings/navigation.ts#hasAnyVisibleStep`):
   * collapsing the section that owns the current step relocates the cursor
   * to the nearest step that stays visible, and collapsing the last
   * remaining visible section is a no-op (thermo accessibility review,
   * Important I1 — see `toggleSection`'s own doc comment for why this, not
   * an "all sections collapsed" empty state, is the APG-conformant choice).
   * That toggle also carries `aria-disabled="true"` (never the native
   * `disabled` attribute, which would drop it from the tab order) while its
   * own collapse would be refused, per the APG's own "always one panel
   * expanded" variant, which names exactly this attribute for exactly this
   * situation (`wouldRefuseCollapse`) — so the no-op is announced, not
   * silent, and clears the moment another section opens.
   */

  interface Props {
    manifest: ArenaManifest | undefined;
    rewiringNull: RewiringNullLoadResult | undefined;
    nullExplanation: NullExplanationLoadResult | undefined;
    pathwayInterventions: PathwayInterventionsLoadResult | undefined;
    /** WP3 of `.agents/plans/repertoire-null`, wired per `findings-tour`'s own `01-findings-panel.md` ("optional `repertoireNull`" input). `undefined` while `App.svelte`'s repertoire-null load has not yet resolved. */
    repertoireNull: RepertoireNullLoadResult | undefined;
    /** WP4 of `.agents/plans/task-generality`. `undefined` while `App.svelte`'s task-generality load has not yet resolved. */
    taskGenerality: TaskGeneralityLoadResult | undefined;
    /** WP3 of `.agents/plans/selection-robustness`. `undefined` while `App.svelte`'s selection-robustness load has not yet resolved. */
    selectionRobustness: SelectionRobustnessLoadResult | undefined;
    /** WP3 of `.agents/plans/readout-attribution`. `undefined` while `App.svelte`'s readout-attribution load has not yet resolved. */
    readoutAttribution: ReadoutAttributionLoadResult | undefined;
  }

  let {
    manifest,
    rewiringNull,
    nullExplanation,
    pathwayInterventions,
    repertoireNull,
    taskGenerality,
    selectionRobustness,
    readoutAttribution
  }: Props = $props();

  let expanded = $state(false);
  let currentIndex = $state(0);
  let headingEls = $state<(HTMLElement | undefined)[]>([]);
  // Section ids the visitor has collapsed. Empty by default: every section
  // starts open (`01-findings-sections.md`'s "open by default"). Membership
  // in this set, not a positive "open" flag, so a still-unseen section
  // (e.g. one a future bean adds) also defaults to open with no extra code.
  let collapsedSectionIds = $state(new Set<string>());

  // Mirrors `ExperimentController#initialize()`'s own `dataBaseUrl`
  // (`${import.meta.env.BASE_URL}data`) — the same base every loader this
  // panel's steps cite already fetched against, so a provenance link
  // resolves under the app's real deployment base path (including `/fly/`
  // — see `tests/e2e/subpath.spec.ts`).
  const dataBaseUrl = `${import.meta.env.BASE_URL}data`;

  const steps = $derived<readonly FindingStep[]>(
    buildFindingSteps({
      manifest,
      dataBaseUrl,
      rewiringNull,
      nullExplanation,
      pathwayInterventions,
      repertoireNull,
      taskGenerality,
      selectionRobustness,
      readoutAttribution
    })
  );

  // The steps grouped into question sections (`../findings/sections.ts`).
  // This is the panel's one flattened display order from here on --
  // `steps.ts`'s own array order no longer determines position on screen.
  const groups = $derived(groupSteps(steps));

  // Each entry pairs a step with the id/title of the section that contains
  // it, in the same flattened order the `<ol>`s render -- the single source
  // `goTo`/the live region/the per-step heading number all read from, so
  // none of them can drift from what is actually on screen.
  const flatSteps = $derived(
    groups.flatMap((group) => group.steps.map((step) => ({ step, sectionId: group.section.id, sectionTitle: group.section.title })))
  );
  const stepIndexById = $derived(new Map(flatSteps.map((entry, index) => [entry.step.id, index] as const)));
  const currentSectionTitle = $derived(flatSteps[currentIndex]?.sectionTitle ?? '');

  const toggle = (): void => {
    expanded = !expanded;
  };

  const isSectionOpen = (sectionId: string): boolean => !collapsedSectionIds.has(sectionId);

  /**
   * Whether *this* section's toggle would have its own collapse refused
   * right now -- the same `!hasAnyVisibleStep(flatSteps, next)` condition
   * `toggleSection`'s own no-op guard uses, computed per section so the
   * button can announce it (thermo accessibility review, follow-up to
   * Important I1). Only an already-open section's toggle can ever be
   * refused (collapsing a closed section is nonsensical -- its own click
   * always *expands*, which `toggleSection` never refuses), so a closed
   * section's toggle is never marked this way regardless of how many other
   * sections are collapsed.
   */
  const wouldRefuseCollapse = (sectionId: string): boolean => {
    if (!isSectionOpen(sectionId)) return false;
    const hypothetical = new Set(collapsedSectionIds);
    hypothetical.add(sectionId);
    return !hasAnyVisibleStep(flatSteps, hypothetical);
  };

  const toggleSection = (sectionId: string): void => {
    const collapsing = !collapsedSectionIds.has(sectionId);
    const next = new Set(collapsedSectionIds);
    if (collapsing) next.add(sectionId);
    else next.delete(sectionId);

    // (thermo accessibility review, Important I1) Collapsing every section
    // in sequence used to leave zero visible steps and zero Previous/Next
    // controls anywhere in the panel, with the live region stuck on its
    // last, now-stale "Step N of M in <section>" text -- a WCAG 4.1.3
    // (Status Messages) violation reachable by an entirely ordinary
    // Tab+Enter/Space accordion walk, not a keyboard trap (every toggle
    // stays reachable) but a real "the stepper silently stopped working"
    // defect. Fixed by refusing the collapse that would leave nothing
    // visible: the vanilla WAI-ARIA APG accordion pattern permits an
    // all-collapsed state for a pure content-disclosure widget, but this
    // component layers a second, custom contract on top -- exactly one
    // `aria-current="step"` cursor with a working Previous/Next pair --
    // that the APG pattern doesn't have and doesn't need to accommodate.
    // The APG's own "accordion with always one panel expanded" variant
    // exists for precisely this situation (a consumer that depends on a
    // panel always being visible), so keeping the last visible section open
    // is the pattern-conformant choice, not a deviation from it. No new UI
    // is required: the toggle simply has no effect when it is the last
    // section with a visible step, the same "can't clear the last required
    // selection" precedent many minimum-one widgets already use.
    if (collapsing && !hasAnyVisibleStep(flatSteps, next)) return;

    // (dual review, Important) Collapsing the section that owns the current
    // step would otherwise hide the only Previous/Next controls in the
    // panel (they render inside the current step's own `<li>`), stranding
    // keyboard/screen-reader navigation with no way to move until the
    // visitor deduces they must reopen that exact section. Relocate the
    // cursor to the nearest step that stays visible instead -- silently,
    // without moving DOM focus away from the toggle button the visitor
    // just activated (matching the WAI-ARIA APG accordion's own "focus
    // stays on the trigger" convention); the live region still announces
    // the new current step/section on the next mutation. The guard above
    // guarantees `nearestVisibleIndex` never returns `undefined` here.
    if (collapsing && flatSteps[currentIndex]?.sectionId === sectionId) {
      const relocated = nearestVisibleIndex(flatSteps, currentIndex, next);
      if (relocated !== undefined) currentIndex = relocated;
    }

    collapsedSectionIds = next;
  };

  const expandSection = (sectionId: string): void => {
    if (!collapsedSectionIds.has(sectionId)) return;
    const next = new Set(collapsedSectionIds);
    next.delete(sectionId);
    collapsedSectionIds = next;
  };

  const focusHeading = async (index: number): Promise<void> => {
    await tick();
    headingEls[index]?.focus();
  };

  const goTo = (index: number): void => {
    if (index < 0 || index >= flatSteps.length) return;
    currentIndex = index;
    // Moving into a collapsed section expands it first (`01-findings-sections.md`),
    // so the step's heading is actually visible/reachable when focus lands there.
    const target = flatSteps[index];
    if (target) expandSection(target.sectionId);
    void focusHeading(index);
  };

  const goPrevious = (): void => goTo(currentIndex - 1);
  const goNext = (): void => goTo(currentIndex + 1);

  const conditionLabel = (condition: FindingStep['condition']): string =>
    condition === 'authored' ? 'Authored decoder' : condition === 'trained' ? 'Trained decoder' : 'Both decoders';
</script>

<section class="panel findings" aria-labelledby="findings-heading">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Evidence chain</p>
      <h2 id="findings-heading">Findings under this model — not claims about the real fly</h2>
    </div>
    <button type="button" onclick={toggle} aria-expanded={expanded}>
      {expanded ? 'Collapse' : 'Expand'}
    </button>
  </div>

  {#if !expanded}
    <p class="reason">Expand to walk the evidence chain: {flatSteps.length} steps, each templated from a verified artifact.</p>
  {/if}

  <!-- (thermo review, maintainability Suggestion) Mounted unconditionally
       (present but empty before expansion), not inside `{#if expanded}` —
       some screen readers do not reliably announce a live region's
       *initial* content when the region and its first text arrive in the
       same DOM update; they announce only a later mutation of an
       already-present node. The first "Step 1 of N" text on expand is then
       a real mutation of an existing node, not a simultaneous insertion. -->
  <div aria-live="polite" class="sr-only">
    {expanded ? `Step ${currentIndex + 1} of ${flatSteps.length} in ${currentSectionTitle}` : ''}
  </div>

  {#if expanded}
    <p class="disclaimer">
      Every step below names its decoder condition (authored or trained) and states its result "under this
      model" — a descriptive finding about this synthetic experiment, never a claim about the real fly. The
      authored decoder is a fixed, hand-written mapping, not biology and not trained.
    </p>

    {#each groups as group (group.section.id)}
      {@const sectionOpen = isSectionOpen(group.section.id)}
      {@const sectionStepsId = `finding-section-${group.section.id}-steps`}
      {@const collapseRefused = wouldRefuseCollapse(group.section.id)}
      <div class="finding-section">
        <h3 class="section-toggle-heading">
          <button
            type="button"
            class="section-toggle"
            onclick={() => toggleSection(group.section.id)}
            aria-expanded={sectionOpen}
            aria-controls={sectionOpen ? sectionStepsId : undefined}
            aria-disabled={collapseRefused ? 'true' : undefined}
          >
            <span class="section-disclosure" aria-hidden="true">{sectionOpen ? '▾' : '▸'}</span>
            {group.section.title}
          </button>
        </h3>

        {#if sectionOpen}
          <ol class="steps" id={sectionStepsId} data-section-id={group.section.id}>
            {#each group.steps as step (step.id)}
              {@const index = stepIndexById.get(step.id) ?? 0}
              <li
                data-step-id={step.id}
                aria-current={index === currentIndex ? 'step' : undefined}
                class="step"
                class:current={index === currentIndex}
              >
                <h4
                  id={`finding-step-${step.id}-heading`}
                  bind:this={headingEls[index]}
                  tabindex="-1"
                >
                  {index + 1}. {step.title}
                  <span class="condition">{conditionLabel(step.condition)}</span>
                </h4>

                {#if step.status === 'ok' && step.sentence}
                  <p class="sentence">{step.sentence}</p>
                {:else if step.status !== 'ok'}
                  <p class="status-line">{findingStepStatusLabel(step.status)}{step.reason ? `: ${step.reason}` : ''}</p>
                {/if}

                {#if step.status === 'ok' && step.details && step.details.length > 0}
                  <!-- WP4 fix pass (thermo review, Important, both reviewers): the
                       task-generality step's per-task detail used to be folded
                       into `sentence` itself, producing one ~1050-character
                       run-on sentence with no navigable internal structure. Each
                       item is now its own `<li>` -- a natural stop for both a
                       sighted skim and a screen-reader user walking the step
                       (this list sits inside the same `<li class="step">`, so it
                       reads immediately after the summary sentence, before the
                       provenance links). A thermo-maintainability review
                       (Important, I1) found this used to be two separately-typed
                       fields (`perTask`/`perSelection`) with two structurally
                       identical render blocks -- collapsed into one `details`
                       field/block here, since every study's own step builder
                       already produces fully-formatted `text` per item; the
                       panel doesn't need to know which study it came from. -->
                  <ul class="detail-list">
                    {#each step.details as item (item.id)}
                      <li><strong>{item.id}</strong> — {item.text}</li>
                    {/each}
                  </ul>
                {/if}

                {#if step.id === 'behavior-repertoire'}
                  <p class="see-also">
                    {#if step.status === 'ok'}
                      <!-- A dual review (Important) caught this claiming content
                           the atlas view does not show -- `Atlas.svelte` renders
                           only the one-line comparison and a "Full report" link;
                           the occupancy map and per-graph audit table exist only
                           in `docs/behavior-repertoire-null-report.md` (see
                           `repertoireNull.ts`'s own doc comment: that detail is
                           "intentionally not carried into this browser-side
                           shape"). -->
                      See the one-line comparison in the <a href="#atlas">behavior atlas</a>; the occupancy map and
                      per-graph audit table are in the full report (linked below).
                    {:else if step.status === 'missing' || step.status === 'loading'}
                      Once published, this step will compare the measured topology's behavior repertoire against the
                      rewired null. See the <a href="#atlas">behavior atlas</a> in the meantime.
                    {:else}
                      <!-- `unavailable`/`invalid`: the artifact *is* published but
                           failed to load or verify -- the step's own status line
                           above already says so; this line only adds the atlas
                           link, not a "not yet published" claim that would be
                           false here (a maintainability review, Suggestion). -->
                      See the <a href="#atlas">behavior atlas</a> in the meantime.
                    {/if}
                  </p>
                {/if}

                {#if step.provenance.length > 0}
                  <ul class="provenance">
                    {#each step.provenance as entry (entry.label)}
                      <li>
                        <span class="provenance-label">{entry.label}</span>
                        <span class="provenance-hash">sha256 {entry.sha256Prefix}…</span>
                        <a href={entry.artifactPath} target="_blank" rel="noreferrer">Pinned JSON</a>
                        <a href={entry.reportPath} target="_blank" rel="noreferrer">Report</a>
                      </li>
                    {/each}
                  </ul>
                {/if}

                {#if index === currentIndex}
                  <!-- (dual review, Important; narrowed by thermo review,
                       maintainability Important — the prior comment here
                       overclaimed "the next Tab reaches Next directly," which a
                       real Playwright/Chromium Tab-key session showed is false
                       whenever the current step has its own provenance links:
                       Tab lands on "Pinned JSON"/"Report" first, then Next/
                       Previous) Rendered inside the current step, immediately
                       after its own content — not above the `<ol>` — so Tab no
                       longer walks through *every earlier step's* own links to
                       reach Next again, only the *current* step's own (a normal
                       "read the evidence, then act" order: 1-4 links, one pair
                       per provenance entry, not every step in the panel's worth
                       a control placed above every list would force). See
                       `tests/e2e/findings.spec.ts`'s real-Tab-order test for the
                       guarantee this comment actually makes. -->
                  <div class="stepper-controls">
                    <button type="button" onclick={goPrevious} disabled={currentIndex === 0}>Previous</button>
                    <button type="button" onclick={goNext} disabled={currentIndex === flatSteps.length - 1}>Next</button>
                  </div>
                {/if}
              </li>
            {/each}
          </ol>
        {/if}
      </div>
    {/each}
  {/if}
</section>

<style>
  .reason {
    margin: 0.9rem 0 0;
    color: #9aacc2;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  .disclaimer {
    margin: 0.9rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  .stepper-controls {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.6rem;
  }

  .finding-section {
    margin: 0.9rem 0 0;
  }

  .finding-section:first-of-type {
    margin-top: 0.7rem;
  }

  .section-toggle-heading {
    margin: 0;
  }

  .section-toggle {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    width: 100%;
    background: none;
    border: none;
    padding: 0.3rem 0;
    color: #edf4ff;
    font-size: 0.9rem;
    font-weight: 600;
    text-align: left;
    cursor: pointer;
  }

  .section-toggle:focus-visible {
    outline: 2px solid #7be5c5;
    outline-offset: 2px;
  }

  /* Still focusable (never the native `disabled` attribute -- see the
     component's own doc comment), just a visual echo of `aria-disabled`
     for a sighted user: the last section with a visible step can't be
     collapsed further. */
  .section-toggle[aria-disabled='true'] {
    cursor: default;
    opacity: 0.7;
  }

  .section-disclosure {
    color: #79d8d0;
    font-size: 0.75rem;
    width: 0.8rem;
    flex: none;
  }

  .steps {
    margin: 0.6rem 0 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 0.75rem;
  }

  .step {
    border: 1px solid #1d2b3a;
    border-radius: 0.5rem;
    padding: 0.7rem 0.85rem;
    background: rgb(121 216 208 / 4%);
  }

  .step.current {
    border-color: #3b665b;
    background: rgb(121 216 208 / 10%);
  }

  .step h4 {
    margin: 0 0 0.4rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #edf4ff;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    outline: none;
  }

  .step h4:focus-visible {
    outline: 2px solid #7be5c5;
    outline-offset: 2px;
  }

  .condition {
    font-size: 0.68rem;
    font-weight: 400;
    color: #9aacc2;
    border: 1px solid #2b4354;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
  }

  .sentence {
    margin: 0.4rem 0;
    color: #cbd8e7;
    font-size: 0.82rem;
    line-height: 1.5;
  }

  .status-line {
    margin: 0.4rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
    font-style: italic;
  }

  .see-also {
    margin: 0.4rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
  }

  .detail-list {
    margin: 0.4rem 0;
    padding-left: 1.1rem;
    display: grid;
    gap: 0.3rem;
  }

  .detail-list li {
    color: #cbd8e7;
    font-size: 0.78rem;
    line-height: 1.4;
  }

  .detail-list strong {
    color: #e8f0f8;
  }

  .see-also a {
    color: #79d8d0;
  }

  .provenance {
    margin: 0.5rem 0 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 0.3rem;
  }

  .provenance li {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: #9aacc2;
    border: none;
    padding: 0;
  }

  .provenance-hash {
    font-family: ui-monospace, 'SFMono-Regular', Menlo, monospace;
  }

  .provenance a {
    color: #79d8d0;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
