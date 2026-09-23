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
});
