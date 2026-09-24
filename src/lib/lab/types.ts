export interface Options {
  seed: number;
  device: "cpu" | "cuda";
  population: number;
  generations: number;
  training_seeds: number;
  heldout_seeds: number;
  ticks: number;
}
export interface Arm {
  name: string;
  scores: number[];
  mean_score: number;
  effect: { mean: number; low: number; high: number };
  pickups: number[];
  contacts: number[];
}
export interface Frame {
  tick: number;
  agents: number[][][];
  food: number[][][];
  hazard: number[];
}
export interface Result {
  schema_version: number;
  model_version: string;
  provenance: string;
  options: Options;
  training_seeds: number[];
  heldout_seeds: number[];
  history: number[];
  arms: Arm[];
  frames: Frame[];
  weights: Record<string, number[][]>;
  runtime: {
    device: string;
    device_name: string;
    torch: string;
    cuda: string | null;
    training_seconds: number;
    evaluation_seconds: number;
    peak_tensor_bytes: number | null;
    precision: string;
  };
}
export interface Job {
  id: string;
  status:
    | "queued"
    | "running"
    | "cancelling"
    | "cancelled"
    | "failed"
    | "completed";
  progress: {
    phase?: string;
    generation?: number;
    generations?: number;
    best_score?: number;
  };
  result: Result | null;
  error: string | null;
}
