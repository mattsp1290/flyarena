import { loadAtlas } from '../atlas/assets';
import { resolveController } from '../atlas/controller';
import { AUTHORED, type Decoder } from './decoder';
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
    if (command.type !== 'prepare' && command.type !== 'run' && command.type !== 'run-atlas') throw new Error('Invalid worker command');
    let request: Request | undefined;
    let topology: GraphMode;
    if (command.type === 'run' || command.type === 'run-atlas') {
      request = validateRequest(command.request);
      topology = request.topology;
    } else topology = command.topology;
    const prepared = await loadPreparedGraph(command.baseUrl, topology);
    send({ type: 'prepared', preparation: { identity: prepared.identity, targets: prepared.targets } });
    if (!request) return;
    const started = performance.now();
    let decoder: Decoder = AUTHORED;
    if (command.type === 'run-atlas') {
      if (!command.selection || !/^[a-f0-9]{64}$/.test(command.selection.atlasSha256)) throw new Error('Missing selected atlas identity');
      const loaded = await loadAtlas(command.baseUrl, command.selection.atlasSha256);
      decoder = { decoder: 'atlas-trained', controller: await resolveController(loaded, prepared, command.selection.id) };
    }
    const iterator = experiment(prepared, request, decoder);
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
