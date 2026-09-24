import { afterEach, describe, expect, it, vi } from 'vitest';
import { CounterfactualClient, ExperimentCancelled } from '../../src/lib/counterfactual/client';
import { DEFAULT_REQUEST } from '../../src/lib/counterfactual/types';
import type { WorkerEvent } from '../../src/lib/counterfactual/protocol';

class FakeWorker {
  onmessage: ((event: MessageEvent<WorkerEvent>)=>void)|null = null;
  onerror: ((event: ErrorEvent)=>void)|null = null;
  onmessageerror: (()=>void)|null = null;
  terminate = vi.fn();
  postMessage = vi.fn();
  emit(event:WorkerEvent) { this.onmessage?.({data:event} as MessageEvent<WorkerEvent>); }
}
const preparation = {identity:{topology:'biological',binarySha256:'a',sourceBinarySha256:'a',sourceArtifact:'a',sourceDataset:'fixture',license:'test',neuronCount:1,edgeCount:0},targets:[]} as const;
const make = () => {
  const workers:FakeWorker[]=[];
  const client = new CounterfactualClient(()=>{const w=new FakeWorker();workers.push(w);return w as unknown as Worker;},'/data',{preparation:30,progress:120,total:300});
  return {client,workers};
};
afterEach(()=>vi.useRealTimers());
describe('counterfactual worker lifetime',()=>{
  it('times out stalled preparation and can prepare successfully with a new worker',async()=>{
    vi.useFakeTimers();const {client,workers}=make();
    const pending=client.prepare('biological');const caught=expect(pending).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(31);await caught;
    expect(workers[0].terminate).toHaveBeenCalledOnce();
    const retry=client.prepare('biological');workers[1].emit({type:'prepared',preparation:{...preparation,targets:[]}});
    await expect(retry).resolves.toEqual(preparation);expect(workers[1].terminate).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  });
  it('times out silent execution, cancels instantly and ignores obsolete results',async()=>{
    vi.useFakeTimers();const {client,workers}=make();
    const run=client.run(DEFAULT_REQUEST,vi.fn());const timed=expect(run).rejects.toThrow('timed out');
    workers[0].emit({type:'prepared',preparation:{...preparation,targets:[]}});
    await vi.advanceTimersByTimeAsync(121);await timed;
    const next=client.run(DEFAULT_REQUEST,vi.fn());const cancelled=expect(next).rejects.toBeInstanceOf(ExperimentCancelled);
    const obsolete=workers[1].onmessage;
    client.cancel();await cancelled;
    obsolete?.({data:{type:'progress',completed:99,total:99}} as MessageEvent<WorkerEvent>);
    expect(workers[1].terminate).toHaveBeenCalledOnce();expect(vi.getTimerCount()).toBe(0);
  });
  it('reports worker errors and serialization failures without hanging',async()=>{
    const {client,workers}=make();
    const pending=client.prepare('biological');workers[0].onmessageerror?.();
    await expect(pending).rejects.toThrow('decode');
    expect(workers[0].terminate).toHaveBeenCalledOnce();
  });
});
