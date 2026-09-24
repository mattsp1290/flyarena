"""Batched authored foraging task and recurrent controller (version 1)."""
from dataclasses import dataclass
import math
import torch

VERSION = "synthetic-foraging-v1"
UNITS = 64
GROUPS = [f"G{i + 1}" for i in range(8)]
ARMS = ["baseline", "sham", "disconnected", *GROUPS]
DT = 1 / 15
EXTENT = 8.0


def circuit(seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    recurrent = torch.randn(UNITS, UNITS, generator=generator) * (0.8 / math.sqrt(UNITS))
    sensory = torch.randn(8, UNITS, generator=generator) * 0.6
    readout = torch.randn(UNITS + 1, 2, generator=generator) * 0.12
    return tuple(t.to(device) for t in (recurrent, sensory, readout))


def neural_step(state, observation, recurrent, sensory, mask, recurrence):
    updated = 0.75 * state + 0.25 * torch.tanh(
        (state @ recurrent) * recurrence + observation @ sensory
    )
    return updated * mask


def move(position, heading, speed, action):
    """Authored kinematics; tanh motor outputs must be in [-1, 1]."""
    heading = torch.remainder(heading + action[..., 1] * 3 * DT + math.pi, math.tau) - math.pi
    speed = (speed * (1 - 0.7 * DT) + (action[..., 0] + 1) * 0.5 * 5 * DT).clamp(0, 4)
    direction = torch.stack((torch.sin(heading), torch.cos(heading)), -1)
    position = (position + direction * speed[..., None] * DT).clamp(-EXTENT, EXTENT)
    return position, heading, speed


def reward(food_distance, next_distance, hazard_distance, previous_contact):
    hit = next_distance < 0.65
    contact = hazard_distance < 0.9
    entered = contact & ~previous_contact
    return food_distance - next_distance + hit * 10 - entered * 2 - 0.005, hit, contact, entered


@dataclass
class Rollout:
    scores: torch.Tensor
    pickups: torch.Tensor
    contacts: torch.Tensor
    paths: list


def simulate(readouts, recurrent, sensory, seeds, ticks, *, arms=None, checkpoint=lambda: None,
             capture=False):
    """Independent candidate/arm × seed worlds, identical external schedules per seed."""
    device = readouts.device
    candidates, count = readouts.shape[0], len(seeds)
    # CPU generation makes the exogenous schedule independent of device and batch order.
    schedules, headings = [], []
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed)
        schedules.append(torch.rand(ticks + 1, 2, generator=rng) * 12 - 6)
        headings.append(torch.rand((), generator=rng).item() * math.tau - math.pi)
    targets = torch.stack(schedules).to(device)
    position = torch.zeros(candidates, count, 2, device=device)
    heading = torch.tensor(headings, device=device).expand(candidates, -1).clone()
    speed = torch.zeros(candidates, count, device=device)
    state = torch.zeros(candidates, count, UNITS, device=device)
    pickups = torch.zeros(candidates, count, dtype=torch.long, device=device)
    contacts = torch.zeros_like(pickups)
    previous_contact = torch.zeros_like(pickups, dtype=torch.bool)
    scores = torch.zeros_like(speed)
    mask = torch.ones(candidates, 1, UNITS, device=device)
    recurrence = torch.ones(candidates, 1, 1, device=device)
    if arms:
        for index, arm in enumerate(arms):
            if arm in GROUPS:
                start = GROUPS.index(arm) * 8
                mask[index, :, start:start + 8] = 0
            elif arm == "disconnected":
                recurrence[index] = 0
    seed_indices = torch.arange(count, device=device).expand(candidates, -1)
    paths = []
    sample_ticks = set(round(i * ticks / min(ticks, 99)) for i in range(min(ticks, 99) + 1))

    def frame(tick, food, hazard):
        # Retain all held-out seed trajectories, bounded to 100 frames.
        if capture and tick in sample_ticks:
            paths.append({"tick": tick, "agents": position.cpu().tolist(),
                          "food": food.cpu().tolist(), "hazard": hazard.cpu().tolist()})

    def hazard_at(tick):
        # Fixed world-time schedule, never dependent on actions or intervention.
        phase = tick * DT * 0.6
        return torch.tensor([3 * math.sin(phase), 3 * math.cos(phase)], device=device)

    for tick in range(ticks):
        if tick % 15 == 0:
            checkpoint()
        food = targets[seed_indices, pickups]
        hazard = hazard_at(tick)
        frame(tick, food, hazard)
        food_delta, hazard_delta = food - position, hazard - position
        food_distance = torch.linalg.vector_norm(food_delta, dim=-1)
        hazard_distance = torch.linalg.vector_norm(hazard_delta, dim=-1)

        def bearing(delta):
            angle = torch.atan2(delta[..., 0], delta[..., 1]) - heading
            return (torch.remainder(angle + math.pi, math.tau) - math.pi) / math.pi

        clearances = []
        for offset in (0, -math.pi / 2, math.pi / 2):
            direction = torch.stack((torch.sin(heading + offset), torch.cos(heading + offset)), -1)
            boundary = torch.where(direction >= 0, EXTENT, -EXTENT)
            distances = (boundary - position) / torch.where(direction.abs() < 1e-7, 1e-7, direction)
            clearances.append((distances.amin(-1) / 16).clamp(0, 1))
        observation = torch.stack((bearing(food_delta), (food_distance / 16).clamp(0, 1),
                                   bearing(hazard_delta), (hazard_distance / 16).clamp(0, 1),
                                   *clearances, speed / 4), -1)
        state = neural_step(state, observation, recurrent, sensory, mask, recurrence)
        action = torch.tanh(torch.bmm(torch.cat((state, torch.ones_like(state[..., :1])), -1), readouts))
        position, heading, speed = move(position, heading, speed, action)
        next_distance = torch.linalg.vector_norm(food - position, dim=-1)
        delta_score, hit, contact, entered = reward(
            food_distance, next_distance,
            torch.linalg.vector_norm(hazard_at(tick + 1) - position, dim=-1), previous_contact)
        scores += delta_score
        pickups += hit.long()
        contacts += entered.long()
        previous_contact = contact
    frame(ticks, targets[seed_indices, pickups], hazard_at(ticks))
    checkpoint()
    return Rollout(scores, pickups, contacts, paths)
