// @vitest-environment node
import { describe, expect, it } from 'vitest';

import { parseAttribution, parseIndexGraphTransfer, TRANSFER_INPUT_CHANNELS, TRANSFER_OUTPUT_POPULATIONS } from '../../scripts/null/intervention-attribution';

/** Coverage for `scripts/null/intervention-attribution.ts` — WP1's `attribution.json`/`index.json` transfer-table parsing WP4 combines into the published artifact/report. */

const validTransfer3x8 = TRANSFER_OUTPUT_POPULATIONS.map(() => TRANSFER_INPUT_CHANNELS.map((_, i) => i * 0.001));

const validAttribution = {
  version: 1,
  P: {
    kind: 'P',
    swaps: 6,
    targetReached: true,
    stopReason: 'target_reached',
    finalRightClearanceThrust: 0.09,
    finalForwardClearanceThrust: 0.02,
    candidateSampleSeed: 20260925,
    steps: [{}, {}, {}, {}, {}, {}]
  },
  Q: {
    kind: 'Q',
    swaps: 5,
    targetReached: true,
    stopReason: 'target_reached',
    finalRightClearanceThrust: 0.08,
    finalForwardClearanceThrust: 0.019,
    candidateSampleSeed: 20260925,
    steps: [{}, {}, {}, {}, {}]
  },
  R: { applicable: false, edgeCount: 0, reason: 'no input-labeled -> thrust edges in the biological graph' },
  biological: {
    transfer: { full3x8: validTransfer3x8, rightClearanceThrust: 0.003, forwardClearanceThrust: 0.002 },
    sourceSha256: 'f'.repeat(64),
    explanationCrossCheck: {
      rightClearanceThrust: { computed: 0.003, published: 0.003 },
      forwardClearanceThrust: { computed: 0.002, published: 0.002 }
    }
  },
  nullRegeneration: {
    rightClearanceThrust: { p25: 0.01 },
    forwardClearanceThrust: { p25: 0.009 }
  },
  producer: {
    script: 'scripts/analysis/interventions.py',
    sourceSha256: 'a'.repeat(64),
    dependencies: ['scripts/analysis/interventions.py'],
    host: { arch: 'aarch64', python: '3.12.3' }
  }
};

describe('parseAttribution', () => {
  it('parses a well-formed attribution.json', () => {
    const parsed = parseAttribution(JSON.stringify(validAttribution), 'attribution.json');
    expect(parsed.P.swaps).toBe(6);
    expect(parsed.P.targetReached).toBe(true);
    expect(parsed.P.stepCount).toBe(6);
    expect(parsed.Q.swaps).toBe(5);
    expect(parsed.Q.stepCount).toBe(5);
    expect(parsed.R.applicable).toBe(false);
    expect(parsed.biological.transfer.full3x8).toEqual(validTransfer3x8);
    expect(parsed.producer.dependencies).toEqual(['scripts/analysis/interventions.py']);
  });

  it('throws when P.swaps is missing', () => {
    const { swaps: _s, ...rest } = validAttribution.P;
    const bad = { ...validAttribution, P: rest };
    expect(() => parseAttribution(JSON.stringify(bad), 'a.json')).toThrow(/swaps is not a non-negative integer/);
  });

  it('throws when biological.transfer.full3x8 has the wrong shape', () => {
    const bad = { ...validAttribution, biological: { ...validAttribution.biological, transfer: { ...validAttribution.biological.transfer, full3x8: [[1, 2, 3]] } } };
    expect(() => parseAttribution(JSON.stringify(bad), 'a.json')).toThrow(/full3x8 is missing or not a 3x8 table/);
  });

  it('throws when R.edgeCount is not a non-negative integer', () => {
    const bad = { ...validAttribution, R: { ...validAttribution.R, edgeCount: -1 } };
    expect(() => parseAttribution(JSON.stringify(bad), 'a.json')).toThrow(/edgeCount is not a non-negative integer/);
  });

  it('throws when producer.dependencies is missing', () => {
    const { dependencies: _d, ...restProducer } = validAttribution.producer;
    const bad = { ...validAttribution, producer: restProducer };
    expect(() => parseAttribution(JSON.stringify(bad), 'a.json')).toThrow(/producer is missing script\/sourceSha256\/dependencies/);
  });

  it('throws when nullRegeneration is missing', () => {
    const { nullRegeneration: _n, ...rest } = validAttribution;
    expect(() => parseAttribution(JSON.stringify(rest), 'a.json')).toThrow(/missing nullRegeneration/);
  });
});

describe('parseIndexGraphTransfer', () => {
  const indexText = JSON.stringify({
    entries: [
      { id: 'P', kind: 'P', swaps: 6, transfer: { full3x8: validTransfer3x8 } },
      { id: 'Q', kind: 'Q', swaps: 5, transfer: { full3x8: validTransfer3x8 } }
    ]
  });

  it('finds the requested id and returns its swaps/transfer', () => {
    const result = parseIndexGraphTransfer(indexText, 'index.json', 'P');
    expect(result.swaps).toBe(6);
    expect(result.transfer).toEqual(validTransfer3x8);
  });

  it('throws when the id is not found', () => {
    expect(() => parseIndexGraphTransfer(indexText, 'index.json', 'Z')).toThrow(/has no entry with id "Z"/);
  });

  it('throws when the entry has no transfer table', () => {
    const badText = JSON.stringify({ entries: [{ id: 'P', swaps: 6 }] });
    expect(() => parseIndexGraphTransfer(badText, 'index.json', 'P')).toThrow(/missing a 3x8 transfer\.full3x8 table/);
  });
});
