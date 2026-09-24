"""Seed policy (`.agents/plans/trained-readout/03-cem-training.md`'s "Seed
policy" table): the training/validation/held-out seed ranges, deterministic
per-generation training-seed sampling, and the held-out guard.

This module is the single owner of "which seeds may be used for what" so
that `cem.py` (the CEM optimizer, generic over any `evaluate(theta_batch,
seeds) -> fitness` callable) and `rollout.py` (the batched-episode
simulation module) both depend on it instead of one depending on the other.
Before this module existed, `cem.py` imported `assert_no_held_out_seeds`
from `rollout.py` to reach one seed-range assertion, which transitively
pulled the entire world/sensors/model/readout/config simulation stack into
what is otherwise a pure, simulation-agnostic optimizer module — a
module-boundary leak `cem.py`'s own docstring disclaimed. `cem.py` no
longer imports `rollout.py` at all; both `cem.py` and `rollout.py` import
this module instead.

**Training-seed RNG keying (deviates from the plan's literal formula — see
`training/README.md`'s "Replica training-seed independence" section for the
full rationale):** `sample_training_seeds` keys `numpy.random.default_rng`
on the two-element entropy tuple `[trainer_seed, generation]`
(`numpy.random.SeedSequence` semantics), not the plan's literal
`trainer_seed + generation` scalar sum. The literal formula aliases whenever
two replicas' `trainer_seed`s differ by less than `generations` (e.g. at the
plan's own default replica seeds 101/202/303 and `G=150`, replica 101's
generation `g >= 101` draws exactly the same seed set as replica 202's
generation `g - 101`) — which directly contradicts the plan's own "R = 3
independent trainer_seed values per arm" framing. The entropy-tuple keying
has no such aliasing (confirmed by
`test_sample_training_seeds_is_replica_independent_across_defaults`) and
changes no other behavior: same signature, same per-generation determinism,
same held-out-disjoint range.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Training seed range (03-cem-training.md's "Seed policy" table).
TRAINING_SEED_LOW = 1
TRAINING_SEED_HIGH = 10000

# Validation seed range (fixed; used only for best-ever tracking, never for
# elite selection).
VALIDATION_SEED_START = 20001
VALIDATION_SEED_COUNT = 64

# Held-out seed range (reserved for WP4's authoritative evaluation; never
# sampled by the training or validation policy).
HELD_OUT_SEED_START = 30001
HELD_OUT_SEED_COUNT = 100
HELD_OUT_SEED_END = HELD_OUT_SEED_START + HELD_OUT_SEED_COUNT - 1


def assert_no_held_out_seeds(seeds: Sequence[int]) -> None:
    """Raise `AssertionError` naming the offending seed if any element of
    `seeds` falls in `[HELD_OUT_SEED_START, HELD_OUT_SEED_END]`."""
    for seed in seeds:
        if HELD_OUT_SEED_START <= seed <= HELD_OUT_SEED_END:
            raise AssertionError(
                f"seed {seed} is in the held-out range [{HELD_OUT_SEED_START}, {HELD_OUT_SEED_END}]; "
                "held-out seeds are reserved for WP4's authoritative evaluation and must never enter "
                "a CEM training or validation batch"
            )


def sample_training_seeds(
    trainer_seed: int,
    generation: int,
    count: int,
    low: int = TRAINING_SEED_LOW,
    high: int = TRAINING_SEED_HIGH,
) -> list[int]:
    """`count` training seeds for one generation, sampled without
    replacement from `numpy.random.default_rng([trainer_seed, generation])`
    over `[low, high]` inclusive.

    03-cem-training.md's "Seed policy" table literally specifies
    `numpy.random.default_rng(trainer_seed + generation)` (a scalar sum).
    This module deliberately keys on the two-element entropy tuple
    `[trainer_seed, generation]` instead — see this module's doc comment for
    why (the scalar-sum formula aliases between replicas whose trainer_seeds
    differ by less than the generation count, which contradicts the plan's
    own stated intent of R independent replicas)."""
    rng = np.random.default_rng([trainer_seed, generation])
    seeds = rng.choice(np.arange(low, high + 1), size=count, replace=False)
    return [int(seed) for seed in seeds]
