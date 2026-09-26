import { afterEach, beforeEach, vi } from 'vitest';
import { ExperimentController, type ExperimentControllerCallbacks } from '../../src/lib/experiment/controller';
import type { AgentId } from '../../src/lib/arena/types';
import type { GraphMode } from '../../src/lib/connectome/format';
import type { ArenaManifest, TrainedReadoutLoadResult } from '../../src/lib/experiment/assets';
import type { RewiringNullLoadResult } from '../../src/lib/experiment/rewiringNull';
import type { NullExplanationLoadResult } from '../../src/lib/experiment/nullExplanation';
import type { PathwayInterventionsLoadResult } from '../../src/lib/experiment/pathwayInterventions';
import type { RepertoireNullLoadResult } from '../../src/lib/atlas/repertoire';
import type { ExperimentStatus } from '../../src/lib/experiment/state';
import type { DecoderKind } from '../../src/lib/worker/protocol';
import { createPublicDataFetch, FakeNeuralWorker } from '../helpers/fake-worker';

/**
 * Shared fixture/lifecycle setup for `ExperimentController` tests, factored
 * out so `tests/unit/experiment-controller.test.ts` and
 * `tests/unit/experiment-controller-decoder.test.ts` (split out of the
 * former when it crossed the thermo review's 1000-line threshold — see
 * `reviews/feat-nom6-trained-toggle-thermo-2026-09-24-766f09c/thermo-architecture/01-critical-and-important.md`)
 * do not each hand-roll their own copy of the same `beforeEach`/`afterEach`
 * wiring and callback recorder.
 */

export const TOTAL_TICKS = 30;
export const SEED = 12345;

export type TestControllerCallbacks = ExperimentControllerCallbacks & {
  statuses: ExperimentStatus[];
  errors: string[];
  topologyApplied: Array<[AgentId, GraphMode]>;
  switchCounts: Array<Readonly<Record<AgentId, number>>>;
  trainedReadoutStatuses: TrainedReadoutLoadResult[];
  rewiringNullResults: RewiringNullLoadResult[];
  nullExplanationResults: NullExplanationLoadResult[];
  pathwayInterventionsResults: PathwayInterventionsLoadResult[];
  repertoireNullResults: RepertoireNullLoadResult[];
  decodersApplied: DecoderKind[];
};

export const createCallbacks = (): TestControllerCallbacks => {
  const statuses: ExperimentStatus[] = [];
  const errors: string[] = [];
  const topologyApplied: Array<[AgentId, GraphMode]> = [];
  const switchCounts: Array<Readonly<Record<AgentId, number>>> = [];
  const trainedReadoutStatuses: TrainedReadoutLoadResult[] = [];
  const rewiringNullResults: RewiringNullLoadResult[] = [];
  const nullExplanationResults: NullExplanationLoadResult[] = [];
  const pathwayInterventionsResults: PathwayInterventionsLoadResult[] = [];
  const repertoireNullResults: RepertoireNullLoadResult[] = [];
  const decodersApplied: DecoderKind[] = [];
  return {
    statuses,
    errors,
    topologyApplied,
    switchCounts,
    trainedReadoutStatuses,
    rewiringNullResults,
    nullExplanationResults,
    pathwayInterventionsResults,
    repertoireNullResults,
    decodersApplied,
    onStatusChange: (status) => statuses.push(status),
    onTelemetry: vi.fn(),
    onError: (message) => errors.push(message),
    onManifest: vi.fn(),
    onTopologyApplied: (agentId, mode) => topologyApplied.push([agentId, mode]),
    onTopologySwitchCountChange: (counts) => switchCounts.push({ ...counts }),
    onTrainedReadoutStatus: (status) => trainedReadoutStatuses.push(status),
    onRewiringNull: (result) => rewiringNullResults.push(result),
    onNullExplanation: (result) => nullExplanationResults.push(result),
    onPathwayInterventions: (result) => pathwayInterventionsResults.push(result),
    onRepertoireNull: (result) => repertoireNullResults.push(result),
    onDecoderApplied: (decoder) => decodersApplied.push(decoder)
  };
};

export const createWorker = (): Worker => new FakeNeuralWorker() as unknown as Worker;

/**
 * Registers the shared `beforeEach`/`afterEach` pair (stub `fetch`/`Worker`
 * with the fake-worker helpers; dispose every tracked controller and
 * unstub globals after each test — see the pre-split file's original doc
 * comment for why the cleanup matters: an un-disposed controller's runner
 * can keep ticking in the background past its own test). Must be called at
 * the top level of a test file (module scope, outside any `describe`), the
 * same place `beforeEach`/`afterEach` would otherwise be called directly.
 * Returns `trackController`, which every test that constructs an
 * `ExperimentController` must call immediately after construction.
 */
export const useControllerTestLifecycle = (): {
  trackController: (controller: ExperimentController) => ExperimentController;
} => {
  const activeControllers: ExperimentController[] = [];
  const trackController = (controller: ExperimentController): ExperimentController => {
    activeControllers.push(controller);
    return controller;
  };

  beforeEach(() => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    vi.stubGlobal('Worker', FakeNeuralWorker as unknown as typeof Worker);
  });

  afterEach(() => {
    for (const controller of activeControllers) controller.dispose();
    activeControllers.length = 0;
    vi.unstubAllGlobals();
  });

  return { trackController };
};
