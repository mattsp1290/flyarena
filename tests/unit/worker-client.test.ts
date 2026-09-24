import { describe, expect, it } from 'vitest';
import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { createWorkerClient, WorkerClientError } from '../../src/lib/worker/client';
import { WORKER_PROTOCOL_VERSION } from '../../src/lib/worker/protocol';
import { createRandomGraph } from '../fixtures/tiny-graph';
import { FakeNeuralWorker } from '../helpers/fake-worker';

const graph = createRandomGraph(0x9, { neuronCount: 6, inputChannelCount: 2, outputPopulationCount: 1 });
const buffer = encodeGraphBinary(graph);

describe('createWorkerClient', () => {
  it('round-trips init/step/reset/dispose against a live worker-shaped target, matching responses by requestId', async () => {
    const worker = new FakeNeuralWorker();
    const client = createWorkerClient(worker);

    const initResult = await client.init(buffer.slice(0));
    expect(initResult.ok).toBe(true);
    expect(initResult.neuronCount).toBe(graph.metadata.neuronCount);
    expect(initResult.edgeCount).toBe(graph.metadata.edgeCount);

    const stepResult = await client.step([0.1, -0.2], 2);
    expect(stepResult.ok).toBe(true);
    expect(stepResult.actionFeatures).toHaveLength(graph.metadata.outputPopulationCount);
    for (const value of stepResult.actionFeatures) expect(Number.isFinite(value)).toBe(true);

    const resetResult = await client.reset();
    expect(resetResult.ok).toBe(true);

    const disposeResult = await client.dispose();
    expect(disposeResult.ok).toBe(true);

    client.terminate();
    expect(worker.terminated).toBe(true);
  });

  it('init resolves with the Worker-echoed protocolVersion matching this bundle\'s WORKER_PROTOCOL_VERSION', async () => {
    const worker = new FakeNeuralWorker();
    const client = createWorkerClient(worker);
    const initResult = await client.init(buffer.slice(0));
    expect(initResult.protocolVersion).toBe(WORKER_PROTOCOL_VERSION);
  });

  it('init rejects with a structured protocol-version-mismatch error when the Worker echoes a different protocolVersion', async () => {
    // A minimal `WorkerLike` double (not `FakeNeuralWorker`, which always
    // echoes the real `WORKER_PROTOCOL_VERSION` via production
    // `handleWorkerRequest`) that responds to any `init` request with a
    // deliberately wrong `protocolVersion`, simulating a main thread and
    // Worker built from skewed bundles — the one scenario this check exists
    // to catch.
    const listeners = new Map<string, Set<(event: MessageEvent) => void>>();
    const worker = {
      postMessage: (message: { requestId: string; type: string }) => {
        if (message.type !== 'init') return;
        queueMicrotask(() => {
          const event = {
            data: {
              type: 'init',
              requestId: message.requestId,
              ok: true,
              neuronCount: 1,
              edgeCount: 0,
              inputChannelCount: 1,
              outputPopulationCount: 1,
              protocolVersion: WORKER_PROTOCOL_VERSION + 1
            }
          } as unknown as MessageEvent;
          for (const listener of listeners.get('message') ?? []) listener(event);
        });
      },
      addEventListener: (type: string, listener: (event: MessageEvent) => void) => {
        let set = listeners.get(type);
        if (!set) {
          set = new Set();
          listeners.set(type, set);
        }
        set.add(listener);
      },
      removeEventListener: (type: string, listener: (event: MessageEvent) => void) => {
        listeners.get(type)?.delete(listener);
      }
    };
    const client = createWorkerClient(worker as never);

    const initPromise = client.init(buffer.slice(0));
    await expect(initPromise).rejects.toThrow(/protocol version/i);
    await initPromise.catch((error: unknown) => {
      expect(error).toBeInstanceOf(WorkerClientError);
      expect((error as WorkerClientError).code).toBe('protocol-version-mismatch');
    });
  });

  it('setActivity round-trips a set-activity request, echoing enabled and toggling whether step responses carry rates', async () => {
    const worker = new FakeNeuralWorker();
    const client = createWorkerClient(worker);
    await client.init(buffer.slice(0));

    const disabledStep = await client.step([0.1, -0.2], 1);
    expect('rates' in disabledStep).toBe(false);

    const enableResult = await client.setActivity(true);
    expect(enableResult.ok).toBe(true);
    expect(enableResult.enabled).toBe(true);

    const enabledStep = await client.step([0.1, -0.2], 1);
    // `ArrayBuffer.isView` (not `toBeInstanceOf(Float32Array)`): the value
    // crosses `structuredClone` inside `FakeNeuralWorker`, which under
    // jsdom's Vitest environment can construct it in a different realm than
    // this test file's own `Float32Array` global — a real cross-realm
    // typed-array value that `instanceof` alone cannot see as one.
    expect(ArrayBuffer.isView(enabledStep.rates)).toBe(true);
    expect(enabledStep.rates).toHaveLength(graph.metadata.neuronCount);

    const disableResult = await client.setActivity(false);
    expect(disableResult.ok).toBe(true);
    expect(disableResult.enabled).toBe(false);

    const redisabledStep = await client.step([0.1, -0.2], 1);
    expect('rates' in redisabledStep).toBe(false);
  });

  it('setActivity rejects before init with not-initialized', async () => {
    const worker = new FakeNeuralWorker();
    const client = createWorkerClient(worker);
    await expect(client.setActivity(true)).rejects.toThrow(/not-initialized/);
  });

  it('rejects the returned promise with the structured error message on a Worker failure response', async () => {
    const worker = new FakeNeuralWorker();
    const client = createWorkerClient(worker);

    await expect(client.step([0, 0], 1)).rejects.toThrow(/not-initialized/);
  });

  it('matches concurrent in-flight requests to their own responses, not to each other', async () => {
    const worker = new FakeNeuralWorker();
    const client = createWorkerClient(worker);
    await client.init(buffer.slice(0));

    const [a, b, c] = await Promise.all([
      client.step([0.1, 0.1], 1),
      client.step([0.2, 0.2], 1),
      client.step([0.3, 0.3], 1)
    ]);
    expect(a.requestId).not.toBe(b.requestId);
    expect(b.requestId).not.toBe(c.requestId);
  });

  it('rejects every pending request when terminated', async () => {
    const worker = new FakeNeuralWorker();
    const client = createWorkerClient(worker);
    await client.init(buffer.slice(0));

    const pending = client.step([0, 0], 1);
    client.terminate();
    await expect(pending).rejects.toThrow(WorkerClientError);

    await expect(client.step([0, 0], 1)).rejects.toThrow('terminated');
  });

  it('resolves each in-flight request with its own response even when the worker delivers them out of order', async () => {
    // A distinct fake from FakeNeuralWorker: this test needs to control
    // delivery order directly rather than have responses land in
    // submission order, to prove routing is by requestId and not by
    // arrival order (three distinct requestIds alone would pass even for a
    // client that routed responses to the wrong callers, as long as
    // delivery happened to match submission order).
    const sentRequestIds: string[] = [];
    const listeners = new Set<(event: MessageEvent) => void>();
    const worker = {
      postMessage: (message: { requestId: string }) => {
        sentRequestIds.push(message.requestId);
      },
      addEventListener: (type: string, listener: (event: MessageEvent) => void) => {
        if (type === 'message') listeners.add(listener);
      },
      removeEventListener: (type: string, listener: (event: MessageEvent) => void) => {
        listeners.delete(listener);
      }
    };
    const client = createWorkerClient(worker as never);

    const p1 = client.reset();
    const p2 = client.reset();
    const p3 = client.reset();
    expect(sentRequestIds).toHaveLength(3);

    const deliver = (index: number): void => {
      const event = { data: { type: 'reset', requestId: sentRequestIds[index], ok: true } } as unknown as MessageEvent;
      for (const listener of listeners) listener(event);
    };
    // Reverse delivery order: the 3rd request's response arrives first.
    deliver(2);
    deliver(0);
    deliver(1);

    const [r1, r2, r3] = await Promise.all([p1, p2, p3]);
    expect(r1.requestId).toBe(sentRequestIds[0]);
    expect(r2.requestId).toBe(sentRequestIds[1]);
    expect(r3.requestId).toBe(sentRequestIds[2]);
  });

  it('marks the client failed after a Worker error event: every later request rejects immediately instead of hanging forever', async () => {
    const listeners = new Map<string, Set<(event: Event) => void>>();
    const worker = {
      postMessage: () => {},
      addEventListener: (type: string, listener: (event: Event) => void) => {
        let set = listeners.get(type);
        if (!set) {
          set = new Set();
          listeners.set(type, set);
        }
        set.add(listener);
      },
      removeEventListener: () => {}
    };
    const client = createWorkerClient(worker as never);

    const pendingBeforeError = client.step([0, 0], 1);
    const errorEvent = new ErrorEvent('error', { message: 'module evaluation failed' });
    for (const listener of listeners.get('error') ?? []) listener(errorEvent);

    await expect(pendingBeforeError).rejects.toThrow(/module evaluation failed/);
    await expect(client.step([0, 0], 1)).rejects.toThrow(/previously failed/);
  });
});
