# Authored counterfactual circuit lab

Everything in this experiment is synthetic: topology, weights, dynamics, grouping, sensors, rewards and geometry. Groups G1–G8 are contiguous blocks of eight units, not anatomical regions. This lab does not use a measured connectome. It is independent of the browser arena's two-agent physics and replay schema.

## Numerical model: synthetic-foraging-v1

A batch contains independent candidate/arm × seed worlds. Each world has one point agent, a food target and one moving hazard in the square [-8, 8]². Time advances by 1/15 second per tick. The agent starts at the origin with seed-generated heading and zero speed. CPU-seeded target coordinates are uniform in [-6, 6]². A pickup selects the next coordinate from that seed's precomputed schedule. The hazard follows `(3 sin(0.6t), 3 cos(0.6t))`. Interventions share starting state and exogenous schedules, but pickup times can diverge.

Eight observations enter the controller, in order: food bearing / π, food distance / 16, hazard bearing / π, hazard distance / 16, forward/left/right wall-ray clearance / 16, and speed / 4. Bearings wrap to [-1, 1]; distances and clearances clamp to [0, 1]. There is no absolute-position input. World coordinates are used by simulation and rendering only.

The 64-unit circuit uses fixed seeded Gaussian recurrent weights with scale 0.8/√64 and sensory weights with scale 0.6. One neural update occurs per world tick:

`h_next = mask × (0.75 h + 0.25 tanh(recurrence × (h W) + observation E))`

Actions are `tanh([h_next, 1] R)` with a 65×2 trained readout, including bias. All arms reset neural state to zero. Silenced groups remain zero after every update and before readout; disconnected recurrence zeros only the recurrent contribution, retaining leak and sensory input.

Heading adds `yaw × 3 × dt` and wraps to [-π, π). Thrust maps the first output from [-1, 1] to [0, 1]. Speed updates to `clamp(speed × (1 − 0.7dt) + thrust × 5dt, 0, 4)`. Position adds `speed × (sin heading, cos heading) × dt`, then clamps to the square. This point-agent task slides at walls; it has no collision impulses or biomechanics.

Food is collected within distance 0.65. Hazard contact occurs within distance 0.9 and is penalized only upon entry. Score per tick is old food distance minus new food distance, plus 10 for pickup, minus 2 for hazard entry, minus 0.005. Distance reward uses the current target before respawn. The reward is an authored objective, not evidence of animal-like behavior.

## Training and controls

Readout evolution begins with seeded Gaussian weights (scale 0.12). Each generation evaluates the retained readout plus Gaussian mutations, selects the highest mean training score, and retains that candidate. Mutation scale is `0.15 × (1 − generation / generations) + 0.02`. The unchanged incumbent is always candidate zero. Training fitness is not held-out performance and is not guaranteed to improve.

Training seeds start at `(experiment_seed × 97) mod 2³¹`; held-out seeds start 10,000 later. Evaluation uses the frozen readout and 11 arms: baseline, empty-mask sham, disconnected recurrence, and silencing each of G1–G8. The sham must match baseline exactly on a given runtime. No arm gets different sensory or motor weights, rewards, or retraining.

For each arm, pair its score with baseline by held-out seed. Report the mean difference and `mean ± 1.96 × sample_standard_deviation / sqrt(n)`. These are **descriptive normal intervals**, not corrected hypothesis tests, biological confidence claims, or guarantees about unsampled seeds. With only eight seeds, normal intervals are particularly rough. Zero effect is a valid result.

## Limits and reproducibility

Requests accept population 4–64, generations 1–40, training seeds 4–32, held-out seeds 8–64, and ticks 30–600. The service runs one active job, rejects simultaneous submissions with 409, retains at most four jobs, and loses them on restart. Cancellation is checked every 15 ticks and between phases. A cancellation that arrives before completion publication wins. Shutdown signals cancellation and waits up to 30 seconds; a driver stall can exceed that bound and is an operational failure.

CUDA uses float32 with TF32 disabled and deterministic algorithms enabled. PyTorch's allocator is limited to 2 GiB and the container has a 6 GiB RAM limit. These do not reserve or fully isolate unified GPU memory from other applications. The displayed peak is PyTorch tensor allocation, not total device/process memory. Results depend on runtime versions; CPU and GPU rounding can change contact events and evolution rankings. Do not expect bit-identical training across devices. The small frozen-weight parity test uses 2e-4 absolute and relative tolerances and avoids observed contact divergences.

Exports contain task/model identifiers, constants, intervention definitions, all weights, options and seeds, per-seed scores/contact counts, training history, runtime versions, timings and at most 100 replay frames for every arm and seed. Replay is sampled presentation, not a full state checkpoint. Exact reevaluation runs the versioned engine from weights and seeds.

```bash
# Mount an exported result read-only; write reevaluation inside the ephemeral container.
docker run --rm --network none \
  -v "$PWD/my-export.json:/input.json:ro" --entrypoint python \
  flyarena-lab:local -m flyarena_lab.cli \
  --reevaluate /input.json --device cpu --output /tmp/reevaluated.json
```

To retain the reevaluation, mount a writable output directory and point `--output` into it (the image runs as UID 65532), or use `docker cp` from a named container before removing it. For GPU reevaluation add `--gpus all` and `--device cuda`. The CLI rejects incompatible versions, seed-manifest changes, malformed weights and exports larger than 16 MiB.
