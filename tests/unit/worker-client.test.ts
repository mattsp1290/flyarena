import { describe, expect, it } from 'vitest';
import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { createWorkerClient, WorkerClientError } from '../../src/lib/worker/client';
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
