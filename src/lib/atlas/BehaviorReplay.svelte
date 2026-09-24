<script lang="ts">
  import { onDestroy } from 'svelte';
  import type { Frame } from '../counterfactual/types';
  let { frames, seed }: { frames: Frame[]; seed: number } = $props();
  let index = $state(0), playing = $state(false);
  let timer: ReturnType<typeof setInterval> | undefined;
  const current = $derived(frames[index]);
  const x = (v: number) => 20 + (v + 12) * 15;
  const y = (v: number) => 20 + (8 - v) * 15;
  const path = $derived(frames.slice(0, index + 1).map(f => { const p = f.snapshot.agents[0].position; return `${x(p.x)},${y(p.z)}`; }).join(' '));
  function stop() { clearInterval(timer); playing = false; }
  function play() {
    if (playing) { stop(); return; }
    if (index === frames.length - 1) index = 0;
    playing = true;
    const period = Math.max(16, (frames[1].snapshot.timeSeconds - frames[0].snapshot.timeSeconds) * 1000);
    timer = setInterval(() => { if (index < frames.length - 1) index++; else stop(); }, period);
  }
  onDestroy(stop);
</script>
<div class="replay">
  <svg viewBox="0 0 400 280" role="img" aria-label="Discovered controller replay">
    <rect x="20" y="20" width="360" height="240" fill="#0a1420" stroke="#395062" />
    <polyline points={path} fill="none" stroke="#7be5c5" stroke-opacity=".6" stroke-width="2" />
    {#each current.snapshot.foods as food}<circle cx={x(food.position.x)} cy={y(food.position.z)} r="4" fill="#f1c870" />{/each}
    {#each current.snapshot.hazards as hazard}<circle cx={x(hazard.position.x)} cy={y(hazard.position.z)} r="9" fill="#d8768b" fill-opacity=".55" />{/each}
    {#each current.snapshot.agents as agent}
      <g transform={`translate(${x(agent.position.x)}, ${y(agent.position.z)}) rotate(${agent.heading * 180 / Math.PI})`}>
        <circle r="5" fill={agent.id === 'left' ? '#b1f3dc' : '#657488'} />
        <path d="M 0 -10 L -3 -3 L 3 -3 Z" fill="#fff" />
      </g>
    {/each}
  </svg>
  <div class="toolbar"><button onclick={play}>{playing ? 'Pause behavior replay' : 'Play behavior replay'}</button><span>{current.snapshot.timeSeconds.toFixed(1)} s · score {current.scores.left.movementScore.toFixed(2)}</span></div>
  <label>Behavior replay timeline<input aria-label="Behavior replay timeline" type="range" min="0" max={frames.length - 1} bind:value={index} oninput={stop} /></label>
  <p>Held-out seed {seed}, chosen before evaluation. Mint: controlled agent · gray: parked opponent · gold: food · rose: hazards.</p>
</div>
<style>
  svg { width:100%; display:block; max-height:340px; } .toolbar { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }
  p, label, span { color:#aabed0; font-size:.8rem; line-height:1.5; } label { display:grid; gap:.5rem; margin-top:1rem; }
  input { width:100%; accent-color:#7be5c5; } button { border:1px solid #52796e; background:#18342d; color:#d4f5e9; padding:.6rem .8rem; border-radius:6px; cursor:pointer; }
  button:focus-visible,input:focus-visible { outline:2px solid #f1c870; outline-offset:3px; }
</style>
