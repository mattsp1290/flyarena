import { createWorld } from '../../src/lib/arena/world';
import { ARENA_CONFIG } from '../../src/lib/arena/config';
import type { ReadonlyWorldState } from '../../src/lib/arena/types';
import { captureFrame } from '../../src/lib/counterfactual/engine';
import type { Frame } from '../../src/lib/counterfactual/types';
import { runEpisode, type AgentEpisodeConfig } from '../training/episode';
import type { Metrics } from '../../src/lib/atlas/types';

export function evaluateBehavior(
  left: AgentEpisodeConfig,
  seed: number,
  ticks: number,
  capture = false
): { metrics: Metrics; frames: Frame[] } {
  const visited = new Set<number>();
  let yaw = 0;
  const visit = (world: ReadonlyWorldState) => {
    const p = world.agents.find((a) => a.id === 'left')!.position;
    const x = Math.max(
      0,
      Math.min(9, Math.floor(((p.x + ARENA_CONFIG.halfWidth) / (2 * ARENA_CONFIG.halfWidth)) * 10))
    );
    const z = Math.max(
      0,
      Math.min(9, Math.floor(((p.z + ARENA_CONFIG.halfDepth) / (2 * ARENA_CONFIG.halfDepth)) * 10))
    );
    visited.add(z * 10 + x);
  };
  const initial = createWorld(seed);
  visit(initial);
  const frames: Frame[] = capture ? [captureFrame(initial)] : [];
  const samples = new Set(
    Array.from({ length: Math.min(60, ticks) + 1 }, (_, i) =>
      Math.round((i * ticks) / Math.min(60, ticks))
    )
  );
  const result = runEpisode({
    seed,
    ticks,
    left,
    right: { decoder: 'parked' },
    onTick: (_tick, actions, world) => {
      visit(world);
      yaw += actions.left.yaw;
      if (capture && samples.has(world.tick)) frames.push(captureFrame(world));
    }
  });
  return {
    metrics: { ...result.left, coverage: visited.size / 100, turning: yaw / ticks },
    frames
  };
}
