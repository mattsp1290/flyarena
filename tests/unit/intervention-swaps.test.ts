import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { extractInterventionSwaps } from '../../scripts/attribution/intervention-swaps';

/**
 * Coverage for `scripts/attribution/intervention-swaps.ts` (split out of
 * `archive-readouts.ts` -- a thermo-maintainability review finding).
 */

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'intervention-swaps-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const validAttributionPayload = {
  P: {
    kind: 'P',
    swaps: 1,
    steps: [
      {
        step: 0,
        accepted: true,
        addedEdge: { pre: 1, post: 2 },
        addedEdge2: { pre: 3, post: 4 },
        removedEdge: { pre: 5, post: 6 },
        removedEdge2: { pre: 7, post: 8 }
      },
      {
        step: 1,
        accepted: false,
        addedEdge: { pre: 100, post: 200 },
        addedEdge2: { pre: 100, post: 200 },
        removedEdge: { pre: 100, post: 200 },
        removedEdge2: { pre: 100, post: 200 }
      }
    ]
  },
  Q: {
    kind: 'Q',
    swaps: 1,
    steps: [
      {
        step: 0,
        accepted: true,
        addedEdge: { pre: 9, post: 10 },
        addedEdge2: { pre: 11, post: 12 },
        removedEdge: { pre: 13, post: 14 },
        removedEdge2: { pre: 15, post: 16 }
      }
    ]
  }
};

describe('extractInterventionSwaps', () => {
  it('extracts only accepted steps\' addedEdge/addedEdge2/removedEdge/removedEdge2 for P and Q', () => {
    const src = resolve(root, 'attribution.json');
    const out = resolve(root, 'archive', 'intervention-swaps-v1.json');
    writeFileSync(src, JSON.stringify(validAttributionPayload));

    extractInterventionSwaps(src, out);
    const written = JSON.parse(readFileSync(out, 'utf8'));
    expect(written.version).toBe(1);
    expect(written.sourceSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(written.swaps).toEqual([
      {
        id: 'P',
        addedEdges: [
          { pre: 1, post: 2 },
          { pre: 3, post: 4 }
        ],
        removedEdges: [
          { pre: 5, post: 6 },
          { pre: 7, post: 8 }
        ]
      },
      {
        id: 'Q',
        addedEdges: [
          { pre: 9, post: 10 },
          { pre: 11, post: 12 }
        ],
        removedEdges: [
          { pre: 13, post: 14 },
          { pre: 15, post: 16 }
        ]
      }
    ]);
  });

  it('refuses a file with no P entry', () => {
    const src = resolve(root, 'no-p.json');
    writeFileSync(src, JSON.stringify({ Q: { kind: 'Q', steps: [] } }));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/no "P" entry/);
  });

  it('refuses a file with no Q entry', () => {
    const src = resolve(root, 'no-q.json');
    writeFileSync(src, JSON.stringify({ P: { kind: 'P', swaps: 0, steps: [] } }));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/no "Q" entry/);
  });

  it('refuses when the accepted-step count disagrees with the entry\'s own "swaps" field', () => {
    const src = resolve(root, 'bad-swaps-count.json');
    const payload = {
      ...validAttributionPayload,
      P: { ...validAttributionPayload.P, swaps: 99 }
    };
    writeFileSync(src, JSON.stringify(payload));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/accepted step\(s\) but swaps=99/);
  });

  it('refuses a malformed edge', () => {
    const src = resolve(root, 'bad-edge.json');
    const payload = {
      ...validAttributionPayload,
      P: {
        kind: 'P',
        swaps: 1,
        steps: [
          {
            step: 0,
            accepted: true,
            addedEdge: { pre: 1 }, // missing "post"
            addedEdge2: { pre: 3, post: 4 },
            removedEdge: { pre: 5, post: 6 },
            removedEdge2: { pre: 7, post: 8 }
          }
        ]
      }
    };
    writeFileSync(src, JSON.stringify(payload));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/malformed edge/);
  });
});
