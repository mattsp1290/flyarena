import { describe, expect, it } from 'vitest';
import { compareEvidence, readEvidenceRequest, serializeEvidence } from '../../src/lib/counterfactual/evidence';
import { DEFAULT_REQUEST, MODEL_VERSION } from '../../src/lib/counterfactual/types';

describe('counterfactual evidence boundaries', () => {
  it('requires its own schema and bounded settings, rejecting other experiment exports', () => {
    expect(readEvidenceRequest({evidence:{schemaVersion:1,modelVersion:MODEL_VERSION,request:DEFAULT_REQUEST}}).request).toEqual(DEFAULT_REQUEST);
    for (const doc of [{schema_version:1}, {evidence:{schemaVersion:2,modelVersion:MODEL_VERSION}}, {evidence:{schemaVersion:1,modelVersion:MODEL_VERSION,request:{...DEFAULT_REQUEST,horizon:Infinity}}}]) expect(()=>readEvidenceRequest(doc)).toThrow();
    expect(()=>serializeEvidence({bad:NaN})).toThrow('nonfinite');
  });
  it('accepts exact structure regardless of key order and rejects any edited leaf by default', () => {
    expect(compareEvidence({a:1,b:2},{b:2,a:1}).exact).toBe(true);
    expect(compareEvidence({a:1},{a:1,b:2}).matches).toBe(false);
    expect(compareEvidence({results:[{difference:{movementScore:1}}]},{results:[{difference:{movementScore:1+1e-10}}]}).matches).toBe(false);
  });
  it('diagnoses tiny continuous errors explicitly, with strict discrete fields and a density budget', () => {
    const original = {results:Array.from({length:9},()=>({difference:{movementScore:1,foodPickups:1}}))};
    const copy = structuredClone(original); copy.results[0].difference.movementScore += 1e-9;
    const comparison = compareEvidence(original,copy,true);
    expect(comparison.matches).toBe(true);expect(comparison.exact).toBe(false);expect(comparison.inexactLeaves).toBe(1);
    copy.results[0].difference.foodPickups += 1e-9;
    expect(compareEvidence(original,copy,true).matches).toBe(false);
    copy.results[0].difference.foodPickups = 1;
    copy.results.forEach(r=>r.difference.movementScore+=1e-9);
    expect(compareEvidence(original,copy,true).matches).toBe(false);
    expect(compareEvidence({config:{halfWidth:12}},{config:{halfWidth:12.00000001}},true).matches).toBe(false);
    expect(compareEvidence({results:[{difference:{movementScore:1}}]},{results:[{difference:{movementScore:2}}]},true).matches).toBe(false);
  });
});
