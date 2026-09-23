<script lang="ts">
  import type { ExperimentTelemetry } from '../experiment/runner';

  /** Read-only telemetry region (WP6 item 5): elapsed simulated time, per-agent
      food/hazard/distance, active-rate summary, neural step latency, graph
      node/edge counts, seed, and current topology. Purely presentational —
      never writes back into the experiment. */
  interface Props {
    telemetry: ExperimentTelemetry;
  }

  let { telemetry }: Props = $props();

  const formatSeconds = (value: number): string => `${value.toFixed(1)}s`;
  const formatPercent = (value: number): string => `${(value * 100).toFixed(1)}%`;
  const formatMs = (value: number): string => `${value.toFixed(2)} ms`;
  const formatDistance = (value: number): string => value.toFixed(1);

  const ARM_LABEL = { left: 'Left (BIO shape)', right: 'Right (REWIRED shape)' } as const;
  const ARM_IDS = ['left', 'right'] as const;
</script>

<section class="panel" aria-labelledby="telemetry-heading">
  <div class="section-heading">
    <h2 id="telemetry-heading">Telemetry</h2>
    <span>{telemetry.behindRealtime ? 'Running slower than 30 Hz' : `Tick ${telemetry.tick} / ${telemetry.totalTicks}`}</span>
  </div>

  <dl>
    <div><dt>Elapsed simulated time</dt><dd>{formatSeconds(telemetry.elapsedSimulatedSeconds)}</dd></div>
    <div><dt>Seed</dt><dd>{telemetry.seed}</dd></div>
  </dl>

  {#each ARM_IDS as agentId (agentId)}
    {@const agent = telemetry.agents[agentId]}
    <div class="arm">
      <h3>{ARM_LABEL[agentId]}</h3>
      <dl>
        <div><dt>Topology</dt><dd>{agent.topology}</dd></div>
        <div><dt>Graph nodes / edges</dt><dd>{agent.neuronCount} / {agent.edgeCount}</dd></div>
        <div><dt>Food collected</dt><dd>{agent.foodPickups}</dd></div>
        <div><dt>Hazard contacts</dt><dd>{agent.hazardContacts}</dd></div>
        <div><dt>Distance travelled</dt><dd>{formatDistance(agent.distanceTravelled)}</dd></div>
        <div><dt>Active-rate fraction</dt><dd>{formatPercent(agent.activeFraction)}</dd></div>
        <div><dt>Neural step latency (median)</dt><dd>{formatMs(agent.medianStepLatencyMs)}</dd></div>
        <div><dt>Neural step latency (last)</dt><dd>{formatMs(agent.lastStepLatencyMs)}</dd></div>
      </dl>
    </div>
  {/each}
</section>

<style>
  .arm {
    margin-top: 0.9rem;
    padding-top: 0.6rem;
    border-top: 1px solid #1d2b3a;
  }

  .arm h3 {
    margin: 0 0 0.3rem;
    color: #79d8d0;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
</style>
