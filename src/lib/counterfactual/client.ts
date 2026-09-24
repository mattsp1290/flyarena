import type { AtlasSelection } from '../atlas/types';
import type { GraphMode } from '../connectome/format';
import type { ExportDocument, Preparation, Request } from './types';
import type { WorkerCommand, WorkerEvent } from './protocol';

export class ExperimentCancelled extends Error {
  constructor() { super('Experiment cancelled'); }
}
export const WORKER_TIMEOUTS = { preparation: 30_000, progress: 120_000, total: 300_000 };

/** A request owns its worker and timers; cancellation cannot affect a subsequent request. */
export class CounterfactualClient {
  private activeCancel: (() => void) | undefined;
  constructor(
    private readonly createWorker: () => Worker = () => new Worker(new URL('./counterfactual.worker.ts', import.meta.url), { type: 'module' }),
    private readonly baseUrl = `${import.meta.env.BASE_URL}data`,
    private readonly timeouts = WORKER_TIMEOUTS
  ) {}

  prepare(topology: GraphMode): Promise<Preparation> {
    return this.request({ type: 'prepare', baseUrl: this.baseUrl, topology }, event => event.type === 'prepared' ? event.preparation : undefined);
  }
  run(request: Request, progress: (completed: number, total: number) => void): Promise<ExportDocument> {
    return this.request({ type: 'run', baseUrl: this.baseUrl, request }, event => {
      if (event.type === 'progress') progress(event.completed, event.total);
      return event.type === 'complete' ? event.document : undefined;
    });
  }
  runAtlas(selection: AtlasSelection, request: Request, progress: (completed: number, total: number) => void): Promise<ExportDocument> {
    return this.request({ type: 'run-atlas', baseUrl: this.baseUrl, selection, request }, event => {
      if (event.type === 'progress') progress(event.completed, event.total);
      return event.type === 'complete' ? event.document : undefined;
    });
  }
  cancel() { this.activeCancel?.(); }

  private request<T>(command: WorkerCommand, select: (event: WorkerEvent) => T | undefined): Promise<T> {
    this.cancel();
    return new Promise((resolve, reject) => {
      let worker: Worker;
      try { worker = this.createWorker(); } catch (error) { reject(error); return; }
      let settled = false;
      let watchdog: ReturnType<typeof setTimeout>;
      const total = setTimeout(() => finish(new Error('Experiment exceeded the five-minute time limit')), this.timeouts.total);
      const finish = (error?: unknown, result?: T) => {
        if (settled) return;
        settled = true;
        clearTimeout(watchdog); clearTimeout(total);
        worker.onmessage = null; worker.onerror = null; worker.onmessageerror = null;
        worker.terminate();
        if (this.activeCancel === cancel) this.activeCancel = undefined;
        if (error) reject(error); else resolve(result!);
      };
      const cancel = () => finish(new ExperimentCancelled());
      this.activeCancel = cancel;
      const deadline = (ms: number) => {
        clearTimeout(watchdog);
        watchdog = setTimeout(() => finish(new Error('Experiment worker timed out; retry to start a fresh worker')), ms);
      };
      deadline(this.timeouts.preparation);
      worker.onmessage = (message: MessageEvent<WorkerEvent>) => {
        if (settled) return;
        const event = message.data;
        if (!event || !['prepared', 'progress', 'complete', 'error'].includes(event.type)) {
          finish(new Error('Invalid experiment worker response')); return;
        }
        if (event.type === 'error') { finish(new Error(event.message)); return; }
        deadline(this.timeouts.progress);
        try {
          const result = select(event);
          if (result !== undefined) finish(undefined, result);
        } catch (error) { finish(error); }
      };
      worker.onerror = event => { event.preventDefault(); finish(new Error('Experiment worker failed; retry to start a fresh worker')); };
      worker.onmessageerror = () => finish(new Error('Cannot decode experiment worker response'));
      try { worker.postMessage(command); } catch (error) { finish(error); }
    });
  }
}
