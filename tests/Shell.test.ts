import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import Shell from '../src/Shell.svelte';
import { createPublicDataFetch, FakeNeuralWorker } from './helpers/fake-worker';

afterEach(()=>{cleanup();vi.unstubAllGlobals();window.location.hash='';});
it('preserves job identity and reconnects after polling failure across view switches',async()=>{
  const assets=createPublicDataFetch();
  let polls=0;
  const fetch=vi.fn((url:RequestInfo|URL,init?:RequestInit)=>{
    if(String(url).includes('/data/'))return assets(url);
    if(init?.method==='POST')return Promise.resolve(new Response(JSON.stringify({id:'retained-job'}),{status:202}));
    if(init?.method==='GET'){
      polls++;
      if(polls===1)return Promise.reject(new Error('Offline'));
      return Promise.resolve(new Response(JSON.stringify({id:'retained-job',status:'cancelled',progress:{},result:null,error:null})));
    }
    return Promise.resolve(new Response('{}'));
  });
  vi.stubGlobal('fetch',fetch);vi.stubGlobal('Worker',FakeNeuralWorker);
  window.location.hash='dgx';
  const {container}=render(Shell);
  await screen.findByLabelText('Access token');
  await fireEvent.submit(container.querySelector('form')!);
  await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('Offline'));
  window.location.hash='arena';await fireEvent(window,new HashChangeEvent('hashchange'));
  await waitFor(()=>expect(screen.getByLabelText('Experiment status: ready')).toBeInTheDocument());
  expect(screen.queryByText('Reconnect to job')).not.toBeVisible();
  window.location.hash='dgx';await fireEvent(window,new HashChangeEvent('hashchange'));
  await fireEvent.click(await screen.findByRole('button',{name:'Reconnect to job'}));
  await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('cancelled'));
  expect(screen.getByTestId('lab-job-id')).toHaveTextContent('retained-job');
  expect(fetch.mock.calls.filter(([,init])=>init?.method==='POST')).toHaveLength(1);
});
