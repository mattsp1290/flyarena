<script lang="ts">
  import { onDestroy } from 'svelte';
  import type { Branch, Evidence } from './types';
  let { evidence, selectedSeed = 0 }: { evidence: Evidence; selectedSeed?: number } = $props();
  let frameIndex = $state(0);
  let comparison = $state<Branch>('lesion');
  let playing = $state(false);
  let timer: ReturnType<typeof setInterval> | undefined;
  let sample = $derived(evidence.results[selectedSeed]);
  let frames = $derived(sample.branches.baseline.frames);
  let offset = $derived(frames[frameIndex].snapshot.tick - evidence.request.warmup);
  const x = (v: number) => 16 + (v + evidence.config.halfWidth) / (2 * evidence.config.halfWidth) * 448;
  const y = (v: number) => 16 + (evidence.config.halfDepth - v) / (2 * evidence.config.halfDepth) * 300;
  const radius = (v: number) => v / (2 * evidence.config.halfWidth) * 448;
  const score = (branch: Branch, index: number) => sample.branches[branch].frames[index].scores.left.movementScore - sample.checkpoint.scores.left.movementScore;
  const trail = (branch: Branch) => sample.branches[branch].frames.slice(0, frameIndex + 1)
    .map(f => { const p = f.snapshot.agents.find(a => a.id === 'left')!.position; return `${x(p.x)},${y(p.z)}`; }).join(' ');
  let scale = $derived(Math.max(1e-9, ...frames.map((_, i) => Math.abs(score('lesion', i) - score('baseline', i)))));
  let curve = $derived(frames.map((_, i) => `${16 + i * 448 / (frames.length - 1)},${55 - (score('lesion', i) - score('baseline', i)) / scale * 38}`).join(' '));
  function stop() { clearInterval(timer); playing = false; }
  function play() {
    if (playing) { stop(); return; }
    if (frameIndex === frames.length - 1) frameIndex = 0;
    playing = true;
    timer = setInterval(() => {
      if (frameIndex >= frames.length - 1) stop(); else frameIndex++;
    }, 100);
  }
  onDestroy(stop);
</script>
<section class="paired-replay" aria-label="Paired world replay">
  <div class="replay-toolbar">
    <div><p class="eyebrow">Same past. Three possible futures.</p><h2>Inspect the fork</h2></div>
    <label>Compare baseline with
      <select bind:value={comparison}><option value="lesion">Silenced circuit</option><option value="sham">Sham control</option></select>
    </label>
  </div>
  <div class="worlds">
    {#each ['baseline', comparison] as name, index}
      {@const branch = name as Branch}
      {@const frame = sample.branches[branch].frames[frameIndex]}
      <figure>
        <figcaption><strong>{index === 0 ? 'Baseline' : comparison === 'lesion' ? 'Silenced circuit' : 'Sham control'}</strong>
          <span data-testid={`replay-score-${index}`}>Post-fork score {score(branch, frameIndex).toFixed(4)}</span></figcaption>
        <svg viewBox="0 0 480 332" role="img" aria-label={`${branch} world at tick ${frame.snapshot.tick}`}>
          <rect x="16" y="16" width="448" height="300" rx="8" fill="#08131e" stroke="#314a5b" />
          {#each [1, 2, 3, 4, 5, 6, 7] as n}<line x1={16 + n * 56} x2={16 + n * 56} y1="16" y2="316" stroke="#142735" />{/each}
          {#each [1, 2, 3, 4, 5] as n}<line x1="16" x2="464" y1={16 + n * 50} y2={16 + n * 50} stroke="#142735" />{/each}
          {#each frame.snapshot.foods as food}<circle cx={x(food.position.x)} cy={y(food.position.z)} r={radius(food.radius)} fill="#f0cb81" />{/each}
          {#each frame.snapshot.hazards as hazard}<circle cx={x(hazard.position.x)} cy={y(hazard.position.z)} r={radius(hazard.radius)} fill="#ee819447" stroke="#ee8194" />{/each}
          <polyline points={trail(branch)} fill="none" stroke={index === 0 ? '#b4c8d8' : '#7be5c5'} stroke-width="2" />
          {#each frame.snapshot.agents as agent}
            <g transform={`translate(${x(agent.position.x)},${y(agent.position.z)}) rotate(${agent.heading * 180 / Math.PI})`}>
              <circle r="7" fill={agent.id === 'right' ? '#485567' : index === 0 ? '#d9e3eb' : '#7be5c5'} />
              <path d="M 0 -13 L -4 -4 L 4 -4 Z" fill={agent.id === 'right' ? '#7b8490' : '#f6fbff'} />
            </g>
          {/each}
        </svg>
      </figure>
    {/each}
  </div>
  <div class="replay-toolbar">
    <button onclick={play}>{playing ? 'Pause replay' : 'Play replay'}</button>
    <span>Fork tick {evidence.request.warmup} → +{offset} ticks · {(offset * evidence.config.fixedDeltaSeconds).toFixed(2)} simulated seconds</span>
    <button onclick={() => { stop(); frameIndex = 0; }}>Back to fork</button>
  </div>
  <label class="timeline">Paired replay timeline
    <input aria-label="Paired replay timeline" type="range" min="0" max={frames.length - 1} step="1" bind:value={frameIndex} oninput={stop} />
  </label>
  <svg viewBox="0 0 480 110" role="img" aria-label="Silenced minus baseline score through time" class="difference-plot">
    <line x1="16" y1="55" x2="464" y2="55" stroke="#516578" stroke-dasharray="4 4" />
    <polyline points={curve} fill="none" stroke="#7be5c5" stroke-width="2" />
    <line x1={16 + frameIndex * 448 / (frames.length - 1)} x2={16 + frameIndex * 448 / (frames.length - 1)} y1="12" y2="96" stroke="#f0cb81" />
    <text x="16" y="108" fill="#adbecf" font-size="10">Silenced − baseline score · range ±{scale.toPrecision(3)} · dashed line = zero</text>
  </svg>
  <p>Gold: food · rose: hazard · gray agent: zero-action opponent. Each panel shows its own world. Food placement can diverge after different contacts; only the complete state at the fork is identical.</p>
</section>
<style>
  .paired-replay { min-width: 0; } .replay-toolbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
  .replay-toolbar span, p { color:#9eafc2; font-size:.8rem; line-height:1.6; }
  .worlds { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1.5rem 0; }
  figure { margin:0; min-width:0; } figcaption { display:flex; flex-direction:column; gap:.5rem; padding:0 .5rem; font-size:.85rem; } figcaption span { color:#b1c3d1; font-variant-numeric:tabular-nums; }
  svg { width:100%; display:block; } .difference-plot { max-height:160px; margin-top:1rem; }
  label { display:grid; gap:.5rem; font-size:.8rem; color:#b8cad8; } .timeline { margin-top:1rem; }
  input { width:100%; accent-color:#7be5c5; } select, button { color:#d9e9ef; background:#142534; border:1px solid #365064; padding:.6rem .8rem; border-radius:6px; font:inherit; }
  button { cursor:pointer; } button:focus-visible, input:focus-visible, select:focus-visible { outline:2px solid #7be5c5; outline-offset:3px; }
  @media(max-width:760px) { .worlds { grid-template-columns:1fr; } }
</style>
