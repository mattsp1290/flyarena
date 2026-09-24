"""Atlas descriptors accumulated alongside the canonical batched simulation."""
import torch
from .atlas import HIDDEN_SIZE, SUBSTEPS
from .model import create_model_state
from .rollout import RolloutEnv, theta_batch_to_readout_weights, expand_readout_weights, step_readout_world
from .world import create_world_batch


def evaluate_behaviors(env: RolloutEnv, candidates: torch.Tensor, seeds: list[int], ticks: int) -> torch.Tensor:
    if not seeds or not 1 <= len(seeds) <= 16 or not 1 <= ticks <= 1800:
        raise ValueError("Invalid bounded rollout")
    population, episodes = candidates.shape[0], len(seeds)
    batch = population * episodes
    weights = expand_readout_weights(theta_batch_to_readout_weights(
        candidates.to(env.device), env.input_size, HIDDEN_SIZE), episodes)
    world = create_world_batch(seeds * population, device=env.device, config=env.config)
    state = create_model_state(env.graph, batch_size=batch, device=env.device)
    visited = torch.zeros(batch, 100, dtype=torch.bool, device=env.device)
    yaw_sum = torch.zeros(batch, dtype=torch.float64, device=env.device)
    rows = torch.arange(batch, device=env.device)

    def visit():
        position = world.agent_position[:, 0]
        x = ((position[:, 0] + env.config.half_width) / (2 * env.config.half_width) * 10).long().clamp(0, 9)
        z = ((position[:, 1] + env.config.half_depth) / (2 * env.config.half_depth) * 10).long().clamp(0, 9)
        visited[rows, z * 10 + x] = True

    visit()
    for _ in range(ticks):
        world, action = step_readout_world(env, state, weights, world, SUBSTEPS)
        yaw_sum += action[:, 1].double()
        visit()
    quality = world.agent_movement_score[:, 0].reshape(population, episodes).mean(1)
    coverage = (visited.double().mean(1)).reshape(population, episodes).mean(1)
    turning = (yaw_sum / ticks).reshape(population, episodes).mean(1)
    return torch.stack([quality, coverage, turning], dim=1)
