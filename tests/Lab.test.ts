import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Lab from '../src/lib/lab/Lab.svelte';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe('Lab lifecycle', () => {
  it('shows server rejection without retrying the submission', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'A job is already active' }), { status: 409 }));
    vi.stubGlobal('fetch', fetch);
    const { container } = render(Lab);
    await fireEvent.input(screen.getByLabelText('Access token'), { target: { value: 'local-test-token-only' } });
    await fireEvent.submit(container.querySelector('form')!);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('409: A job is already active'));
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: /Train & probe/ })).toBeEnabled();
  });
  it('retains job identity on polling failure and reconnects without resubmission', async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'known-job' }), { status: 202 }))
      .mockRejectedValueOnce(new TypeError('Offline'))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'known-job', status: 'cancelled', progress: {}, result: null, error: null })));
    vi.stubGlobal('fetch', fetch);
    const { container } = render(Lab);
    await fireEvent.submit(container.querySelector('form')!);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Offline'));
    expect(screen.getByRole('button', { name: /Train & probe/ })).toBeDisabled();
    await fireEvent.click(screen.getByRole('button', { name: 'Reconnect to job' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('cancelled'));
    expect(fetch.mock.calls.filter(([, options]) => options.method === 'POST')).toHaveLength(1);
  });
});

describe('Lab teardown', () => {
  it('cancels a known active job with a fresh signal and aborts polling on destruction', async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'active-job' }), { status: 202 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'active-job', status: 'running', progress: {} })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'active-job', status: 'cancelling', progress: {} })));
    vi.stubGlobal('fetch', fetch);
    const { container, unmount } = render(Lab);
    await fireEvent.submit(container.querySelector('form')!);
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('running'));
    const pollingSignal = fetch.mock.calls[1][1].signal;
    unmount();
    const cancellation = fetch.mock.calls.find(([, options]) => options.method === 'DELETE');
    expect(cancellation?.[0]).toContain('active-job');
    expect(cancellation?.[1].signal.aborted).toBe(false);
    expect(pollingSignal.aborted).toBe(true);
    await new Promise(resolve => setTimeout(resolve, 650));
    expect(fetch).toHaveBeenCalledTimes(3);
  });
});
