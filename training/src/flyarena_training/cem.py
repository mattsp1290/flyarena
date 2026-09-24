"""CEM (cross-entropy method) trainer: the seeded, GPU-batched optimization
loop over a pluggable vectorized `evaluate(theta_batch, seeds) -> fitness`
(WP3, `.agents/plans/trained-readout/03-cem-training.md`). This module owns
the CEM defaults, the elite-selection/smoothing update, and the seed-sampling
policy (training seeds per generation, validation seeds, and the held-out
assertion) laid out in the plan's "CEM (defaults, calibrate in WP3)" and
"Seed policy" sections. `rollout.py`'s `evaluate_fitness` is the one
pluggable `evaluate` function this module calls; this module never itself
decides how an episode is simulated.

Reproducibility (03-cem-training.md's "Reproducibility" section): on CPU, the
same `trainer_seed`, `evaluate`, and `CemConfig` reproduce `theta_final`
bit-identically, because every source of randomness here (candidate sampling
via a `torch.Generator` seeded once from `trainer_seed`, and seed sampling
via `numpy.random.default_rng(trainer_seed + generation)`) is deterministic
given the same seed and device. On CUDA, elite selection is a hard rank cut,
so GPU-kernel nondeterminism can compound across generations; that is
measured and recorded as informational (not gated) by `test_cem.py`'s GPU
rerun test, which prints `theta_final`'s max-abs diff across two CUDA runs —
`env.json` itself does not carry that field (it is a training-config
manifest, not a comparison across two runs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .rollout import assert_no_held_out_seeds

EvaluateFn = Callable[[torch.Tensor, Sequence[int]], torch.Tensor]

# Seed policy (03-cem-training.md's "Seed policy" table).
TRAINING_SEED_LOW = 1
TRAINING_SEED_HIGH = 10000
VALIDATION_SEED_START = 20001
VALIDATION_SEED_COUNT = 64


@dataclass(frozen=True)
class CemConfig:
    """CEM hyperparameters. Defaults are 03-cem-training.md's "CEM (defaults,
    calibrate in WP3)" table: population 256, elites 32, generations 150,
    smoothing alpha 0.7 on mean and std, std floor 0.02, init std 0.5, E = 16
    training seeds per generation (so one generation evaluates a `B = P * E
    = 4096` batch at the defaults).

    `alpha` follows the standard smoothed cross-entropy method convention
    (De Boer, Kroese, Mannor & Rubinstein 2005, "A Tutorial on the
    Cross-Entropy Method": `v_t = alpha * v_hat_t + (1 - alpha) * v_{t-1}`):
    it is the weight on the *new* elite estimate, so `alpha=0.7` moves the
    search distribution 70% of the way toward this generation's elites. The
    plan (03-cem-training.md) states only "smoothing alpha = 0.7 on mean and
    std" with no formula; this is the reading the CEM literature's own
    "alpha ~ 0.7" convention implies, and is what `run_cem` implements (see
    its own doc comment)."""

    population: int = 256
    elites: int = 32
    generations: int = 150
    alpha: float = 0.7
    std_floor: float = 0.02
    init_std: float = 0.5
    init_mean: float = 0.0
    training_seeds_per_generation: int = 16
    validation_seed_start: int = VALIDATION_SEED_START
    validation_seed_count: int = VALIDATION_SEED_COUNT
    training_seed_low: int = TRAINING_SEED_LOW
    training_seed_high: int = TRAINING_SEED_HIGH

    def __post_init__(self) -> None:
        if self.population <= 0:
            raise ValueError(f"population must be positive, got {self.population}")
        if self.elites <= 0 or self.elites > self.population:
            raise ValueError(f"elites ({self.elites}) must be in (0, population={self.population}]")
        if self.generations <= 0:
            raise ValueError(f"generations must be positive, got {self.generations}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        for field_name in ("alpha", "std_floor", "init_std", "init_mean"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
        if self.std_floor < 0:
            raise ValueError(f"std_floor must be non-negative, got {self.std_floor}")
        if self.init_std < 0:
            raise ValueError(f"init_std must be non-negative, got {self.init_std}")
        if self.training_seeds_per_generation <= 0:
            raise ValueError(
                f"training_seeds_per_generation must be positive, got {self.training_seeds_per_generation}"
            )
        if self.training_seed_high - self.training_seed_low + 1 < self.training_seeds_per_generation:
            raise ValueError(
                "training seed range "
                f"[{self.training_seed_low}, {self.training_seed_high}] is too small to sample "
                f"{self.training_seeds_per_generation} distinct seeds without replacement"
            )
        if self.validation_seed_count <= 0:
            raise ValueError(f"validation_seed_count must be positive, got {self.validation_seed_count}")
        validation_low = self.validation_seed_start
        validation_high = self.validation_seed_start + self.validation_seed_count - 1
        if self.training_seed_low <= validation_high and validation_low <= self.training_seed_high:
            raise ValueError(
                f"training seed range [{self.training_seed_low}, {self.training_seed_high}] overlaps "
                f"validation seed range [{validation_low}, {validation_high}]; 03-cem-training.md's seed "
                "policy requires them disjoint"
            )


@dataclass
class GenerationRecord:
    generation: int
    mean_fitness: float
    max_fitness: float
    validation_fitness: float


@dataclass
class CemResult:
    """`theta_final`: the smoothed CEM mean after the last generation — "the
    published candidate" (03-cem-training.md: "the published candidate is
    the final CEM mean, which is less noise-selected"). `theta_best`: the
    best-ever candidate by validation mean, tracked alongside but not
    published. Both are `[parameterCount]` float32 tensors on the device
    `run_cem` was called with."""

    theta_final: torch.Tensor
    theta_best: torch.Tensor
    best_validation_fitness: float
    history: list[GenerationRecord]


def sample_training_seeds(
    trainer_seed: int,
    generation: int,
    count: int,
    low: int = TRAINING_SEED_LOW,
    high: int = TRAINING_SEED_HIGH,
) -> list[int]:
    """`count` training seeds for one generation, sampled without
    replacement from `numpy.random.default_rng(trainer_seed + generation)`
    over `[low, high]` inclusive (03-cem-training.md's "Seed policy" table,
    training row: "seeds 1...10000, sampled E = 16 per generation from
    numpy.random.default_rng(trainer_seed + generation)")."""
    rng = np.random.default_rng(trainer_seed + generation)
    seeds = rng.choice(np.arange(low, high + 1), size=count, replace=False)
    return [int(seed) for seed in seeds]


def validation_seeds(config: CemConfig) -> list[int]:
    """The fixed validation seed set (03-cem-training.md: seeds 20001..20064
    by default), used for elite tracking / best-ever selection only — never
    for the CEM fitness that drives elite selection itself."""
    return list(range(config.validation_seed_start, config.validation_seed_start + config.validation_seed_count))


def run_cem(
    theta_dim: int,
    evaluate: EvaluateFn,
    config: CemConfig,
    trainer_seed: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CemResult:
    """Run CEM for `config.generations` generations, seeded entirely by
    `trainer_seed`.

    Each generation: sample `config.training_seeds_per_generation` training
    seeds (`sample_training_seeds`, asserted non-held-out), sample `P =
    config.population` candidates from `N(mean, std)` using a
    `torch.Generator` seeded once from `trainer_seed` (so successive
    generations draw successive, not repeated, noise), score them with
    `evaluate(candidates, training_seeds) -> fitness[P]`, take the top
    `config.elites` by fitness, and smooth `mean`/`std` toward the elite
    mean/std with `config.alpha` weighting the *new* elite estimate (standard
    smoothed-CEM convention — see `CemConfig.alpha`'s doc comment; `std`
    floored at `config.std_floor`). Then evaluate the *new* mean (a single
    candidate) on the fixed validation seed set and update the best-ever
    candidate if it improves.

    `evaluate(theta_batch[P, theta_dim], seeds) -> fitness[P]` never decides
    which seeds to use; this function samples them and asserts none is
    held-out (`assert_no_held_out_seeds`) before every call, including the
    validation call — "the trainer asserts that no held-out seed enters any
    batch" (03-cem-training.md). `evaluate`'s returned fitness is required to
    be finite (`ValueError` otherwise): `torch.topk` ranks `NaN` as the
    largest value, so a single non-finite fitness would otherwise silently
    become an elite and poison `mean`/`std` for every subsequent generation.
    """
    if theta_dim <= 0:
        raise ValueError(f"theta_dim must be positive, got {theta_dim}")
    if trainer_seed < 0:
        raise ValueError(f"trainer_seed must be non-negative (numpy.random.default_rng requires it), got {trainer_seed}")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(trainer_seed)

    mean = torch.full((theta_dim,), config.init_mean, dtype=dtype, device=device)
    std = torch.full((theta_dim,), config.init_std, dtype=dtype, device=device)

    theta_best = mean.clone()
    best_validation_fitness = float("-inf")
    history: list[GenerationRecord] = []

    val_seeds = validation_seeds(config)
    assert_no_held_out_seeds(val_seeds)

    for generation in range(config.generations):
        training_seeds = sample_training_seeds(
            trainer_seed,
            generation,
            config.training_seeds_per_generation,
            config.training_seed_low,
            config.training_seed_high,
        )
        assert_no_held_out_seeds(training_seeds)

        noise = torch.randn(config.population, theta_dim, generator=generator, device=device, dtype=dtype)
        candidates = mean.unsqueeze(0) + noise * std.unsqueeze(0)

        fitness = evaluate(candidates, training_seeds)
        if tuple(fitness.shape) != (config.population,):
            raise ValueError(f"evaluate() must return shape ({config.population},), got {tuple(fitness.shape)}")
        fitness = fitness.to(device=candidates.device)
        if not torch.isfinite(fitness).all():
            bad = torch.nonzero(~torch.isfinite(fitness), as_tuple=False).flatten().tolist()
            raise ValueError(f"evaluate() returned non-finite fitness for candidates {bad} in generation {generation}")

        # Ranked in evaluate()'s own dtype (e.g. float64 from a real rollout's
        # movement score), not truncated to `dtype` first — keeps topk's
        # ranking at full precision; `candidates` (used for the elite
        # mean/std below) stays in `dtype` regardless.
        elite_indices = torch.topk(fitness, config.elites).indices
        elite_candidates = candidates.index_select(0, elite_indices)
        new_mean = elite_candidates.mean(dim=0)
        new_std = elite_candidates.std(dim=0, unbiased=False)

        # Standard smoothed-CEM convention: alpha weights the NEW elite
        # estimate (see CemConfig.alpha's doc comment).
        mean = config.alpha * new_mean + (1 - config.alpha) * mean
        std = (config.alpha * new_std + (1 - config.alpha) * std).clamp(min=config.std_floor)

        assert_no_held_out_seeds(val_seeds)
        validation_fitness = float(evaluate(mean.unsqueeze(0), val_seeds)[0].item())
        if not math.isfinite(validation_fitness):
            raise ValueError(f"evaluate() returned non-finite validation fitness in generation {generation}")
        if validation_fitness > best_validation_fitness:
            best_validation_fitness = validation_fitness
            theta_best = mean.clone()

        history.append(
            GenerationRecord(
                generation=generation,
                mean_fitness=float(fitness.mean().item()),
                max_fitness=float(fitness.max().item()),
                validation_fitness=validation_fitness,
            )
        )

    return CemResult(
        theta_final=mean, theta_best=theta_best, best_validation_fitness=best_validation_fitness, history=history
    )
