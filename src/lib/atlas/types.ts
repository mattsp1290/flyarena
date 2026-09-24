import type { AgentScore } from '../arena/types';
import type { Frame } from '../counterfactual/types';

export const ATLAS_VERSION = 'behavior-atlas-v1';
export const ATLAS_FILE = 'behavior-atlas-v1.json';
export const MAX_ATLAS_BYTES = 8 * 1024 ** 2;
export const DISCOVERY_SEEDS = Array.from({ length: 8 }, (_, i) => 61001 + i);
export const HELDOUT_SEEDS = Array.from({ length: 12 }, (_, i) => 62001 + i);
export const COVERAGE_EDGES = [0, 0.05, 0.1, 0.2, 0.35, 0.6, 1];
export const TURN_EDGES = [-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1];
export const CONTROL_NAMES = ['biological', 'disconnected', 'silenced'] as const;
export type Control = (typeof CONTROL_NAMES)[number];
export interface Metrics extends AgentScore {
  coverage: number;
  turning: number;
}
export interface Candidate {
  id: number;
  theta: number[];
  quality: number;
  coverage: number;
  turning: number;
}
export interface SearchArtifact {
  schemaVersion: 1;
  modelVersion: typeof ATLAS_VERSION;
  options: { seed: number; population: number; generations: number; ticks: number };
  inputSize: number;
  hiddenSize: number;
  substeps: number;
  discoverySeeds: number[];
  heldoutSeeds: number[];
  coverageEdges: number[];
  turnEdges: number[];
  graphArtifactSha256: string;
  bundleSha256: string;
  bundle: Record<string, unknown>;
  candidates: Candidate[];
  history: { generation: number; occupied: number; bestQuality: number }[];
  searchPolicy: {
    initialStd: number;
    mutationScales: number[];
    freshFraction: number;
    weightBound: number;
    ties: string;
    rng: string;
  };
  runtime: {
    device: 'cpu' | 'cuda';
    deviceName: string;
    torch: string;
    cuda: string | null;
    seconds: number;
    peakTensorBytes: number;
    config: Record<string, number>;
  };
}
export interface Cell {
  id: number;
  cell: number;
  quality: number;
  coverage: number;
  turning: number;
  discovery: Metrics[];
  heldout: Record<Control, Metrics[]>;
  replay: Frame[];
}
export interface Atlas {
  schemaVersion: 1;
  modelVersion: typeof ATLAS_VERSION;
  source: SearchArtifact;
  graphBinarySha256: string;
  configFingerprint: string;
  outputNeuronIndices: number[];
  evaluations: { id: number; discovery: Metrics[] }[];
  cells: Cell[];
  authored: Metrics[];
}
export interface AtlasSelection {
  id: number;
  atlasSha256: string;
}
export interface LoadedAtlas {
  atlas: Atlas;
  sha256: string;
}

export function binIndex(value: number, edges: readonly number[]): number {
  if (!Number.isFinite(value) || value < edges[0] || value > edges.at(-1)!)
    throw new Error('Descriptor outside its finite range');
  return Math.min(edges.length - 2, edges.slice(1).filter((edge) => value >= edge).length);
}
export const cellFor = (coverage: number, turning: number) =>
  binIndex(turning, TURN_EDGES) * 6 + binIndex(coverage, COVERAGE_EDGES);
export const average = (values: readonly number[]) =>
  values.reduce((a, b) => a + b, 0) / values.length;
