/**
 * WP1 of `.agents/plans/consolidated-release` (`01-findings-sections.md`):
 * the Findings panel's grouping metadata, kept as data next to the steps
 * (`./steps.ts`) rather than hard-coded in `../ui/FindingsPanel.svelte`'s
 * own markup, so the section map stays one testable source of truth
 * (`00-overview.md`'s "Key decisions").
 *
 * Every id in `FINDING_SECTIONS` is a real, already-shipped
 * `FindingStep['id']` from `./steps.ts` (grounded against the current
 * `buildFindingSteps` order rather than the plan's own worked table, which
 * names the still-unlanded selection-robustness and readout-attribution
 * steps only by description, not by id). Two beans that add Findings steps
 * are still in flight (`flyarena-9mls`, selection robustness; `flyarena-hd0j`,
 * readout attribution) — per the plan's own "merge-order rule"
 * (`01-findings-sections.md`'s Acceptance section), whichever lands after
 * this change adds its exact step id to the matching section's `stepIds` in
 * the same change. Until then, "Does it generalize?" and "What do trained
 * readouts do?" simply carry fewer ids than the plan's worked table shows —
 * `groupSteps` below never fails on a section with fewer entries than
 * expected, only on a step id with *no* entry anywhere.
 */

export interface FindingSection {
  readonly id: string;
  readonly title: string;
  readonly stepIds: readonly string[];
}

export const FINDING_SECTIONS: readonly FindingSection[] = [
  {
    id: 'wiring-special',
    title: 'Is the measured wiring special?',
    stepIds: ['rewiring-null', 'mirrored-decoder', 'behavior-repertoire']
  },
  {
    id: 'scores-low',
    title: 'Why does it score low?',
    stepIds: ['explanation', 'intervention']
  },
  {
    id: 'generalizes',
    title: 'Does it generalize?',
    // The selection-robustness step (`flyarena-9mls`) is not yet landed --
    // its id is added here, in the same change, by whichever lands second.
    stepIds: ['task-generality']
  },
  {
    id: 'trained-readouts',
    title: 'What do trained readouts do?',
    // The readout-attribution step (`flyarena-hd0j`) is not yet landed --
    // its id is added here, in the same change, by whichever lands second.
    stepIds: ['trained-null', 'trained-interventions']
  }
] as const;

/** A step id not classified into any `FINDING_SECTIONS` entry lands here (`00-overview.md`'s "a fifth section, 'Other findings'"), so a new step can never silently vanish from the panel unclassified. */
export const OTHER_SECTION: FindingSection = { id: 'other', title: 'Other findings', stepIds: [] };

export interface GroupedFindingSection<TStep extends { readonly id: string }> {
  readonly section: Pick<FindingSection, 'id' | 'title'>;
  readonly steps: readonly TStep[];
}

/**
 * Groups `steps` (any ordering) into `FINDING_SECTIONS`' declared order,
 * each section's own declared `stepIds` order internally, with the current
 * `steps.ts` array order preserved as the fallback order within `Other
 * findings`. A section with none of its declared ids present is omitted
 * entirely (`01-findings-sections.md`'s own "Empty sections are omitted"),
 * so a still-in-flight section (today, effectively none) never renders an
 * empty accordion shell. `Other findings` is appended last, and only when
 * non-empty -- on `main`, with every shipped step id classified above, it
 * is always empty and never rendered.
 */
export const groupSteps = <TStep extends { readonly id: string }>(
  steps: readonly TStep[]
): readonly GroupedFindingSection<TStep>[] => {
  const byId = new Map(steps.map((step) => [step.id, step] as const));
  const claimed = new Set<string>();
  const groups: GroupedFindingSection<TStep>[] = [];

  for (const section of FINDING_SECTIONS) {
    const sectionSteps: TStep[] = [];
    for (const id of section.stepIds) {
      // (dual review, Suggestion) Guards against a step id listed twice --
      // a duplicate within one section's own `stepIds` (a copy-paste typo),
      // or the same id accidentally added to two different sections (a real
      // near-term risk: the merge-order rule this file's own doc comment
      // describes means two different in-flight beans may each edit this
      // array independently). Without this, the id would be pushed more
      // than once, producing a duplicate Svelte `{#each ... (step.id)}` key
      // and a duplicate DOM `id="finding-step-<id>-heading"` -- first
      // declared location wins, every later duplicate is silently skipped
      // rather than rendered twice.
      if (claimed.has(id)) continue;
      const step = byId.get(id);
      if (!step) continue;
      sectionSteps.push(step);
      claimed.add(id);
    }
    if (sectionSteps.length > 0) {
      groups.push({ section: { id: section.id, title: section.title }, steps: sectionSteps });
    }
  }

  const otherSteps = steps.filter((step) => !claimed.has(step.id));
  if (otherSteps.length > 0) {
    groups.push({ section: { id: OTHER_SECTION.id, title: OTHER_SECTION.title }, steps: otherSteps });
  }

  return groups;
};
