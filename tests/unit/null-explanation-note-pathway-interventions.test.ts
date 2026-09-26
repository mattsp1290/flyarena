import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import NullExplanationNote from '../../src/lib/ui/NullExplanationNote.svelte';
import type { NullExplanationLoadResult } from '../../src/lib/experiment/nullExplanation';
import type { PathwayInterventionsLoadResult } from '../../src/lib/experiment/pathwayInterventions';

/**
 * WP4 of `.agents/plans/pathway-interventions`: component coverage for the
 * tested-outcome sentence and its "Intervention report" link, rendered by
 * `NullExplanationNote.svelte` under the explanation paragraph. Renders the
 * component directly (not through `LedgerPanel.svelte`) since this sentence
 * is this component's own concern.
 */

afterEach(() => cleanup());

const nullExplanationOk = {
  status: 'ok',
  data: {
    version: 1,
    sources: { rewiringNullSha256: 'a'.repeat(64) },
    variants: { flipBoth: { bioPercentile: 0 } },
    regime: { gatePassed: true },
    finding: {
      categories: ['linearPathway'],
      definitionSensitive: false,
      qualifyingMetrics: [{ kind: 'transfer', name: 'T:rightClearance->thrust', spearman: 0.467 }],
      regimeInvalid: false,
      summarySentence: 'Biological is associated with one qualifying metric.'
    }
  }
} as unknown as Extract<NullExplanationLoadResult, { status: 'ok' }>;

const renderNote = (pathwayInterventions: PathwayInterventionsLoadResult | undefined) =>
  render(NullExplanationNote, { nullExplanation: nullExplanationOk, baselinePercentile: 0, rewiringCount: 500, pathwayInterventions });

describe('NullExplanationNote: pathway-interventions tested-outcome sentence', () => {
  it('states the authored category, the channel-specific modifier, and the trained result, with a link to the intervention report', () => {
    const ok: PathwayInterventionsLoadResult = {
      status: 'ok',
      data: {
        version: 1,
        sources: { biologicalSha: 'c'.repeat(64), rewiringNullSha: 'a'.repeat(64), nullExplanationSha: 'b'.repeat(64) },
        authored: { category: 'pathway-supported', channelSpecific: true },
        trained: {
          trainedRobust: true,
          perSeedCategory: { '101': 'no-specific-effect', '202': 'no-specific-effect', '303': 'no-specific-effect' }
        }
      }
    };
    render(NullExplanationNote, { ...{ nullExplanation: nullExplanationOk, baselinePercentile: 0, rewiringCount: 500 }, pathwayInterventions: ok });

    expect(screen.getByText(/tested under this model/i)).toBeInTheDocument();
    expect(screen.getByText(/the pathway-supported category holds/i)).toBeInTheDocument();
    expect(screen.getByText(/the channel-specific modifier holds/i)).toBeInTheDocument();
    expect(screen.getByText(/P shows no advantage over either freshly-trained control arm/i)).toBeInTheDocument();
    expect(screen.getByText(/\(robust\)/i)).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /intervention report/i });
    expect(link).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/pathway-interventions-report.md'
    );
  });

  it('states edge-class-effect and the not-holding channel-specific modifier', () => {
    const ok: PathwayInterventionsLoadResult = {
      status: 'ok',
      data: {
        version: 1,
        sources: { biologicalSha: 'c'.repeat(64), rewiringNullSha: 'a'.repeat(64), nullExplanationSha: 'b'.repeat(64) },
        authored: { category: 'edge-class-effect', channelSpecific: false },
        trained: {
          trainedRobust: true,
          perSeedCategory: { '101': 'edge-class-effect', '202': 'edge-class-effect', '303': 'edge-class-effect' }
        }
      }
    };
    renderNote(ok);
    expect(screen.getByText(/the edge-class-effect category holds/i)).toBeInTheDocument();
    expect(screen.getByText(/the channel-specific modifier does not hold/i)).toBeInTheDocument();
    expect(screen.getByText(/P outperforms the freshly-trained unrestricted \(C\) arm but not the class-matched \(M\) arm/i)).toBeInTheDocument();
  });

  it('states the trained result is inconclusive when trainedRobust is false, without naming a specific trained category', () => {
    const ok: PathwayInterventionsLoadResult = {
      status: 'ok',
      data: {
        version: 1,
        sources: { biologicalSha: 'c'.repeat(64), rewiringNullSha: 'a'.repeat(64), nullExplanationSha: 'b'.repeat(64) },
        authored: { category: 'generic-rewiring-effect', channelSpecific: false },
        trained: {
          trainedRobust: false,
          perSeedCategory: { '101': 'no-specific-effect', '202': 'pathway-supported', '303': 'no-specific-effect' }
        }
      }
    };
    renderNote(ok);
    expect(screen.getByText(/the generic-rewiring-effect category holds/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        /the three trainer seeds do not agree on a category \(seed 101: no-specific-effect, seed 202: pathway-supported, seed 303: no-specific-effect; not robust\)/i
      )
    ).toBeInTheDocument();
  });

  it('hides the sentence and link entirely when pathwayInterventions is undefined (loading) or "missing"', () => {
    renderNote(undefined);
    expect(screen.queryByText(/tested under this model/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /intervention report/i })).not.toBeInTheDocument();

    renderNote({ status: 'missing', reason: 'The manifest has no pathwayInterventions artifact entry.' });
    expect(screen.queryByText(/tested under this model/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /intervention report/i })).not.toBeInTheDocument();
  });

  it('shows a "could not be loaded" message (not "failed verification") when unavailable', () => {
    renderNote({ status: 'unavailable', reason: 'network error (test)' });
    expect(screen.getByText(/intervention test could not be loaded/i)).toBeInTheDocument();
    expect(screen.getByText(/network error \(test\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/failed verification/i)).not.toBeInTheDocument();
  });

  it('shows a verification-failure line when invalid', () => {
    renderNote({ status: 'invalid', reason: 'pathway-interventions artifact sha256 mismatch' });
    expect(screen.getByText(/intervention test failed verification/i)).toBeInTheDocument();
    expect(screen.getByText(/sha256 mismatch/i)).toBeInTheDocument();
    expect(screen.queryByText(/tested under this model/i)).not.toBeInTheDocument();
  });
});
