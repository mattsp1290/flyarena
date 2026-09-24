"""Batched episode rollout for WP3's CEM trainer: world + observe + model +
readout + decode, run for a batch of `P * E` parallel episodes (`P`
candidate readouts, each replayed against its own `E` training seeds), per
the fitness definition in `.agents/plans/trained-readout/03-cem-training.md`
("Fitness and episode"): the trained agent is `left`, `right` receives the
zero action every tick, neural state resets to zero at episode start, and
candidate fitness is the mean over `E` seeds of `left`'s final
`movementScore`.

This module owns exactly one thing: given a batch of flat `theta` vectors and
a list of seeds, run the episodes and return fitness. It does not decide
which seeds to draw (that is `seeds.py`'s job, consumed by `cem.py`'s "Seed
policy" loop) or how CEM updates its search distribution (`cem.py`). It
does, however, refuse to score a held-out seed on its own
(`assert_no_held_out_seeds`, imported from `seeds.py`), so a caller cannot
accidentally leak one into a batch even by bypassing `cem.py`'s own seed
sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .config import ARENA_CONFIG, ArenaConfig
from .graph import ConnectomeGraph, output_neuron_indices
from .model import PreparedGraph, ModelState, create_model_state, run_substeps
from .readout import ReadoutWeights, gather_output_rates, readout_forward, readout_parameter_count
from .seeds import HELD_OUT_SEED_COUNT, HELD_OUT_SEED_START, assert_no_held_out_seeds
from .sensors import observe_batch
from .world import WorldBatch, create_world_batch, step_world_batched


def theta_batch_to_readout_weights(theta: torch.Tensor, input_size: int, hidden_size: int) -> ReadoutWeights:
    """`theta`: `[P, parameterCount]`, each row the flat concatenation
    `[w1 (H*D, row-major), b1 (H), w2 (3*H, row-major), b2 (3)]` —
    the exact layout `scripts/training/run-dir.ts`'s `RunConfig` doc comment
    specifies for `theta_final.npy`, matching `readoutParameterCount`'s own
    summed order. Returns a `P`-batched `ReadoutWeights` view (no copy for
    the slices; `reshape` may copy if the slice isn't already contiguous)."""
    if theta.ndim != 2:
        raise ValueError(f"theta must be 2-D [P, parameterCount], got shape {tuple(theta.shape)}")
    p = theta.shape[0]
    expected = readout_parameter_count(input_size, hidden_size)
    if theta.shape[1] != expected:
        raise ValueError(
            f"theta has parameterCount {theta.shape[1]}, expected {expected} for "
            f"input_size={input_size}, hidden_size={hidden_size}"
        )

    cursor = 0

    def take(count: int) -> torch.Tensor:
        nonlocal cursor
        chunk = theta[:, cursor : cursor + count]
        cursor += count
        return chunk

    w1 = take(hidden_size * input_size).reshape(p, hidden_size, input_size)
    b1 = take(hidden_size)
    w2 = take(3 * hidden_size).reshape(p, 3, hidden_size)
    b2 = take(3)
    return ReadoutWeights(input_size=input_size, hidden_size=hidden_size, w1=w1, b1=b1, w2=w2, b2=b2)


def expand_readout_weights(weights: ReadoutWeights, e: int) -> ReadoutWeights:
    """Repeats each of `P` candidates' weights `e` times consecutively (row
    `i` of the result is candidate `i // e`), matching the row order
    `evaluate_fitness` tiles seeds against (see its doc comment)."""
    return ReadoutWeights(
        input_size=weights.input_size,
        hidden_size=weights.hidden_size,
        w1=torch.repeat_interleave(weights.w1, e, dim=0),
        b1=torch.repeat_interleave(weights.b1, e, dim=0),
        w2=torch.repeat_interleave(weights.w2, e, dim=0),
        b2=torch.repeat_interleave(weights.b2, e, dim=0),
    )


@dataclass
class RolloutEnv:
    """Precomputed, reusable per-`(graph, device)` state for
    `evaluate_fitness`: the sparse operator (`PreparedGraph`) and
    output-neuron indices. Built once per training run (`build_rollout_env`)
    and passed to every generation's `evaluate_fitness` call, matching
    `model.py`'s allocation-free-per-step convention."""

    graph: ConnectomeGraph
    prepared: PreparedGraph
    output_indices: torch.Tensor
    input_size: int
    device: torch.device
    config: ArenaConfig = ARENA_CONFIG


def build_rollout_env(graph: ConnectomeGraph, device: str | torch.device, config: ArenaConfig = ARENA_CONFIG) -> RolloutEnv:
    device = torch.device(device)
    prepared = PreparedGraph(graph, device)
    indices = output_neuron_indices(graph).to(device)
    return RolloutEnv(
        graph=graph, prepared=prepared, output_indices=indices, input_size=int(indices.numel()), device=device, config=config
    )


def step_readout_world(env: RolloutEnv, model_state: ModelState, weights: ReadoutWeights,
                       world: WorldBatch, substeps: int) -> tuple[WorldBatch, torch.Tensor]:
    """Canonical trained closed-loop tick, shared by CEM and atlas discovery."""
    observation = observe_batch(world, "left", env.config).to(torch.float32)
    run_substeps(env.prepared, model_state, observation, substeps)
    gathered = gather_output_rates(env.graph, model_state.rate, env.output_indices)
    action = readout_forward(weights, gathered)
    # Omitted right action means a parked opponent in both experiments.
    return step_world_batched(world, {"left": action}, env.config, validate=False), action


def evaluate_fitness(
    env: RolloutEnv,
    theta_batch: torch.Tensor,
    seeds: Sequence[int],
    hidden_size: int,
    ticks: int,
    substeps: int,
) -> torch.Tensor:
    """Runs one `ticks`-tick episode for every `(candidate, seed)` pair:
    `left` driven by that candidate's readout on `env.graph`, `right` parked
    (the zero action every tick), matching the plan's "Fitness and episode"
    section. Returns fitness `[P]`: the mean over `E = len(seeds)` seeds of
    `left`'s final `movementScore`.

    Row order: candidate `p`'s `E` seeds occupy consecutive rows
    `[p*E, (p+1)*E)` of the `B = P*E` batch (`expand_readout_weights`'s
    `repeat_interleave` order), so `seeds` is tiled `P` times (`list(seeds) *
    p`) to line up with it — reshaping the final `[B]` movement-score vector
    to `[P, E]` and averaging over dim 1 then recovers each candidate's own
    mean fitness.

    Refuses (`AssertionError`) to run if any `seeds` entry is held-out
    (`assert_no_held_out_seeds`), and refuses (`ValueError`) a malformed
    `theta_batch` shape or an empty `seeds` list.
    """
    assert_no_held_out_seeds(seeds)
    if theta_batch.ndim != 2:
        raise ValueError(f"theta_batch must be 2-D [P, parameterCount], got shape {tuple(theta_batch.shape)}")
    p = theta_batch.shape[0]
    e = len(seeds)
    if e == 0:
        raise ValueError("evaluate_fitness requires at least one seed")

    weights = theta_batch_to_readout_weights(
        theta_batch.to(device=env.device, dtype=torch.float32), env.input_size, hidden_size
    )
    expanded_weights = expand_readout_weights(weights, e)

    tiled_seeds = list(seeds) * p
    state = create_world_batch(tiled_seeds, device=env.device, config=env.config)
    model_state = create_model_state(env.graph, batch_size=p * e, device=env.device)

    for _ in range(ticks):
        state, _ = step_readout_world(env, model_state, expanded_weights, state, substeps)

    movement_score = state.agent_movement_score[:, 0]  # [B], left agent, AGENT_IDS index 0
    return movement_score.reshape(p, e).mean(dim=1)
