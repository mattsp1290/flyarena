import type { Decoder } from './decoder';
import type { AgentId, AgentScore, ArenaSnapshot } from '../arena/types';
import type { ArenaConfig } from '../arena/config';
import type { GraphMode } from '../connectome/format';

export const MODEL_VERSION = 'arena-counterfactual-v1';
export const BRANCHES = ['baseline', 'sham', 'lesion'] as const;
export type Branch = typeof BRANCHES[number];
export const TARGET_LABELS = {
  'input-0': 'Food bearing input', 'input-1': 'Food distance input',
  'input-2': 'Hazard bearing input', 'input-3': 'Hazard distance input',
  'input-4': 'Forward clearance input', 'input-5': 'Left clearance input',
  'input-6': 'Right clearance input', 'input-7': 'Speed input',
  bridge: 'Bridge neurons', output: 'Descending output neurons'
} as const;
export type TargetId = keyof typeof TARGET_LABELS;
export interface Request {
  seed: number;
  seedCount: number;
  warmup: number;
  horizon: number;
  topology: GraphMode;
  target: TargetId;
}
export const DEFAULT_REQUEST: Request = {
  seed: 20260923, seedCount: 8, warmup: 120, horizon: 180, topology: 'biological', target: 'bridge'
};
export interface Target {
  id: TargetId;
  label: string;
  indices: number[];
  bodyIds: string[];
}
export interface GraphIdentity {
  topology: GraphMode;
  binarySha256: string;
  sourceBinarySha256: string;
  sourceArtifact: string;
  sourceDataset: string;
  license: string;
  neuronCount: number;
  edgeCount: number;
}
export interface Preparation {
  identity: GraphIdentity;
  targets: Target[];
}
export interface Frame {
  snapshot: ArenaSnapshot;
  scores: Record<AgentId, AgentScore>;
}
export interface BranchResult {
  outcome: AgentScore;
  frames: Frame[];
}
export interface SeedResult {
  seed: number;
  checkpoint: { tick: number; rngState: number; scores: Record<AgentId, AgentScore> };
  branches: Record<Branch, BranchResult>;
  difference: AgentScore;
  shamDifference: AgentScore;
}
export interface Interval { mean: number; low: number; high: number }
interface EvidenceBase {
  schemaVersion: 1;
  modelVersion: typeof MODEL_VERSION;
  request: Request;
  seeds: number[];
  graph: GraphIdentity;
  config: Readonly<ArenaConfig>;
  configFingerprint: string;
  substeps: number;
  opponent: 'zero-action';
  target: Target;
  provenance: { topology: string; grouping: string; dynamics: string; interpretation: string };
  results: SeedResult[];
  summary: {
    means: Record<Branch, AgentScore>;
    effect: Interval;
    shamEffect: Interval;
  };
}
export type EvidenceHeader = Omit<EvidenceBase, 'results' | 'summary'> & Decoder;
export type Evidence = EvidenceBase & Decoder;
export interface ExportDocument {
  evidence: Evidence;
  runtime: { producer: string; platform: string; elapsedMs: number };
}
export const SCORE_KEYS = ['foodPickups', 'hazardContacts', 'distanceTravelled', 'movementScore'] as const;

export function validateRequest(value: unknown): Request {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid experiment settings');
  const input = value as Record<string, unknown>;
  if (Object.keys(input).sort().join(',') !== Object.keys(DEFAULT_REQUEST).sort().join(',')) {
    throw new Error('Unexpected or missing experiment settings');
  }
  function integer(key: string, min: number, max: number): number {
    const n = input[key];
    if (typeof n !== 'number' || !Number.isInteger(n) || n < min || n > max) {
      throw new Error(`${key} must be an integer in [${min}, ${max}]`);
    }
    return n;
  }
  const topology = input.topology;
  if (topology !== 'biological' && topology !== 'rewired' && topology !== 'disconnected') throw new Error('Unknown topology');
  if (typeof input.target !== 'string' || !Object.hasOwn(TARGET_LABELS, input.target)) throw new Error('Unknown target');
  return { seed: integer('seed', 0, 0xffffffff), seedCount: integer('seedCount', 4, 16), warmup: integer('warmup', 0, 300), horizon: integer('horizon', 30, 300), topology, target: input.target as TargetId };
}
export const seedsFor = (request: Request): number[] =>
  Array.from({ length: request.seedCount }, (_, i) => (request.seed + i) >>> 0);
