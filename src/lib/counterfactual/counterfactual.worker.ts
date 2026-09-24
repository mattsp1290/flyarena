import { loadPreparedGraph } from './targets';
import { experiment } from './engine';
import { serializeEvidence } from './evidence';
import { validateRequest, type Request } from './types';
import type { GraphMode } from '../connectome/format';
import type { WorkerCommand, WorkerEvent } from './protocol';

const send = (message: WorkerEvent) => self.postMessage(message);
let used = false;
self.onmessage = async (event: MessageEvent<WorkerCommand>) => {
  if (used) return; // Each Worker owns exactly one bounded request; cancellation terminates it.
  used = true;
  try {
    const command = event.data;
    if (command.type !== 'prepare' && command.type !== 'run') throw new Error('Invalid worker command');
    let request: Request | undefined;
    let topology: GraphMode;
    if (command.type === 'run') {
      request = validateRequest(command.request);
      topology = request.topology;
    } else topology = command.topology;
    const prepared = await loadPreparedGraph(command.baseUrl, topology);
    send({ type: 'prepared', preparation: { identity: prepared.identity, targets: prepared.targets } });
    if (!request) return;
    const started = performance.now();
    const iterator = experiment(prepared, request);
    let next = iterator.next();
    while (!next.done) {
      send({ type: 'progress', completed: next.value, total: request.seedCount });
      next = iterator.next();
    }
    const document = {
      evidence: next.value,
      runtime: { producer: navigator.userAgent, platform: 'browser; architecture unknown', elapsedMs: performance.now() - started }
    };
    serializeEvidence(document); // Never publish evidence with nonfinite arithmetic.
    send({ type: 'complete', document });
  } catch (error) {
    send({ type: 'error', message: error instanceof Error ? error.message : 'Counterfactual experiment failed' });
  }
};
