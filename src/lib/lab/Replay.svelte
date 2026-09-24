<script lang="ts">
  import type { Result } from './types';
  let { result, arm = 0 }: { result: Result; arm?: number } = $props();
  let seed = $state(0);
  let frameIndex = $state(0);
  let frame = $derived(result.frames[frameIndex] ?? result.frames[0]);
  const project = (value: number) => 20 + (value + 8) * 22.5;
  let points = $derived(result.frames.slice(0, frameIndex + 1).map(f =>
    `${project(f.agents[arm]![seed]![0]!)},${project(f.agents[arm]![seed]![1]!)}`).join(' '));
  let baseline = $derived(result.frames.slice(0, frameIndex + 1).map(f =>
    `${project(f.agents[0]![seed]![0]!)},${project(f.agents[0]![seed]![1]!)}`).join(' '));
</script>
<div class="replay-head">
  <div><p class="eyebrow">Held-out world · synthetic</p><h3>Follow the counterfactual</h3></div>
  <label>Replay seed
    <select aria-label="Replay seed" bind:value={seed}>
      {#each result.heldout_seeds as value, i}<option value={i}>{value}</option>{/each}
    </select>
  </label>
</div>
<svg viewBox="0 0 400 400" role="img" aria-label={`Replay of ${result.arms[arm]?.name} at tick ${frame.tick}`}>
  <defs><pattern id="grid" width="22.5" height="22.5" patternUnits="userSpaceOnUse"><path d="M 22.5 0 L 0 0 0 22.5" fill="none" stroke="#173142" stroke-width="0.6" /></pattern></defs>
  <rect x="20" y="20" width="360" height="360" rx="5" fill="url(#grid)" stroke="#365367" />
  <polyline points={baseline} fill="none" stroke="#a3adba" stroke-dasharray="4 4" stroke-width="1.5" />
  <polyline {points} fill="none" stroke="#77f0ca" stroke-width="2.5" />
  <circle cx={project(frame.food[arm]![seed]![0]!)} cy={project(frame.food[arm]![seed]![1]!)} r="7" fill="#f0c879" />
  <circle cx={project(frame.hazard[0]!)} cy={project(frame.hazard[1]!)} r="20.25" fill="#ec7e8940" stroke="#ec7e89" />
  <circle cx={project(frame.agents[arm]![seed]![0]!)} cy={project(frame.agents[arm]![seed]![1]!)} r="5" fill="#77f0ca" />
</svg>
<div class="legend"><span>● Selected arm</span><span class="muted">┄ Baseline</span><span class="food">● Food</span><span class="hazard">○ Hazard</span></div>
<label class="timeline">Replay tick {frame.tick} / {result.options.ticks}
  <input aria-label="Replay timeline" type="range" min="0" max={result.frames.length - 1} step="1" bind:value={frameIndex} />
</label>
<p class="subtle">Same starting conditions and hazard schedule. Paths diverge only through the intervention’s effect on the controller.</p>
