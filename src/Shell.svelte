<script lang="ts">
  import { onMount } from 'svelte';
  import Arena from './App.svelte';
  import type { GraphMode } from './lib/connectome/format';
  type View = 'arena' | 'counterfactual' | 'dgx' | 'atlas';
  const readView = (): View => {
    const hash = window.location.hash.slice(1);
    return hash === 'counterfactual' || hash === 'dgx' || hash === 'atlas' ? hash : 'arena';
  };
  let view = $state<View>(readView());
  let Workbench = $state<typeof import('./lib/counterfactual/Workbench.svelte').default>();
  let Lab = $state<typeof import('./lib/lab/Lab.svelte').default>();
  let Atlas = $state<typeof import('./lib/atlas/Atlas.svelte').default>();
  let error = $state('');
  let setup = $state<{seed:number; topology:GraphMode} | undefined>();
  let destroyed = false;
  let navigation = 0;
  async function route() {
    const current = ++navigation;
    view = readView();
    error = '';
    try {
      if (view === 'counterfactual' && !Workbench) {
        const component = (await import('./lib/counterfactual/Workbench.svelte')).default;
        if (!destroyed) Workbench = component;
      }
      if (view === 'atlas' && !Atlas) {
        const component = (await import('./lib/atlas/Atlas.svelte')).default;
        if (!destroyed) Atlas = component;
      }
      if (view === 'dgx' && !Lab) {
        const component = (await import('./lib/lab/Lab.svelte')).default;
        if (!destroyed) Lab = component;
      }
    } catch (e) {
      if (!destroyed && current === navigation) error = e instanceof Error ? e.message : 'Unable to load view';
    }
  }
  function probe(next: { seed:number; topology:GraphMode }) {
    setup = next; window.location.hash = 'counterfactual';
  }
  onMount(() => {
    void route(); window.addEventListener('hashchange', route);
    return () => { destroyed = true; window.removeEventListener('hashchange', route); };
  });
</script>
<svelte:head><title>FlyArena — {view === 'arena' ? '3D Connectome Arena' : view === 'counterfactual' ? 'Counterfactual workbench' : view === 'atlas' ? 'Behavior atlas' : 'DGX synthetic sandbox'}</title><meta name="description" content="Explore measured connectome topology, paired model interventions, and a separate GPU synthetic circuit sandbox." /></svelte:head>
<nav aria-label="Experiment views">
  <a href="#arena" aria-current={view === 'arena' ? 'page' : undefined}><span>01</span> Arena</a>
  <a href="#counterfactual" aria-current={view === 'counterfactual' ? 'page' : undefined}><span>02</span> Counterfactual workbench</a>
  <a href="#atlas" aria-current={view === 'atlas' ? 'page' : undefined}><span>03</span> Behavior atlas</a>
  <a href="#dgx" aria-current={view === 'dgx' ? 'page' : undefined}><span>04</span> DGX sandbox <small>optional backend</small></a>
</nav>
{#if error}<div class="load-error" role="alert">{error} <button onclick={route}>Retry loading view</button></div>{/if}
{#if view === 'arena'}<Arena onProbe={probe} />
{:else if view === 'counterfactual'}
  {#if Workbench}<Workbench {setup} />{:else if !error}<p class="loading" role="status">Loading workbench…</p>{/if}
{:else if view === 'atlas'}
  {#if Atlas}<Atlas />{:else if !error}<p class="loading" role="status">Loading behavior atlas…</p>{/if}
{:else if !Lab && !error}<p class="loading" role="status">Loading DGX sandbox…</p>{/if}
{#if Lab}
  <!-- Keep job identity and serial polling alive across navigation, including completion while hidden. -->
  <div hidden={view !== 'dgx'} inert={view !== 'dgx'}><Lab /></div>
{/if}
<style>
  nav { display:flex; flex-wrap:wrap; gap:.4rem; max-width:1500px; margin:auto; padding:1rem clamp(1rem,4vw,3rem); border-bottom:1px solid #223646; }
  a { display:flex; align-items:center; gap:.5rem; padding:.65rem .85rem; color:#a7bdcc; text-decoration:none; font-size:.85rem; border:1px solid transparent; border-radius:6px; }
  a[aria-current] { color:#b0f0da; border-color:#3b665b; background:#15332d66; } a span { font-size:.65rem; color:#73999c; } small { font-size:.6rem; color:#8aa2b0; }
  a:focus-visible { outline:2px solid #7be5c5; outline-offset:2px; } .load-error, .loading { padding:2rem; color:#c5d6e5; }
  [hidden] { display:none !important; }
  @media(max-width:600px) { nav { gap:.1rem; } a { padding:.55rem .5rem; font-size:.75rem; } small { display:none; } }
</style>
