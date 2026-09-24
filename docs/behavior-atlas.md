# Behavior atlas and circuit probes

Open **Behavior atlas** to select a GPU-discovered controller, replay its behavior,
inspect held-out controls, and run a fresh circuit-silencing experiment in the browser.
The measured MaleCNS-derived graph is fixed. Sensory encoding, rate dynamics, arena,
readout architecture, score and behavior descriptors are authored. Decoder weights
are optimized. These are model behaviors, not natural fly behavior or biological
causal findings.

## Discovery contract

The offline MAP-Elites search keeps the best mean movement score in each of 36
descriptor cells. The left agent uses the existing 48→8→3 readout; the right agent
receives zero action. Each episode starts with zero neural state and lasts 900
world ticks (30 seconds), with four neural substeps per tick.

- Discovery seeds: 61001–61008. Held-out seeds: 62001–62012. Held-out evaluation
  begins only after canonical selection is frozen and the diversity gate passes.
- Coverage is the fraction of a 10×10 arena grid visited, including the initial
  position and every post-step position, averaged across discovery seeds. Edges
  are `[0, .05, .1, .2, .35, .6, 1]`.
- Turning is the mean signed decoded yaw command, averaged over ticks and seeds.
  Edges are `[-1, -2/3, -1/3, 0, 1/3, 2/3, 1]`. This is a command descriptor,
  not an anatomical claim. Bins are half-open except the final upper endpoint.
- The first population is CPU-RNG-seeded Gaussian weights (standard deviation
  0.5). Subsequent populations keep one quarter fresh candidates and mutate
  uniformly selected occupied elites at standard deviations 0.05, 0.15 or 0.4.
  All weights are clipped to ±8. Equal quality retains the earlier candidate.
- The shipped search uses seed 1729, population 64 and 24 generations. Search is
  bounded to population 128, 96 generations, 1800 ticks and a 1 GiB CUDA tensor
  allocator cap. Two CPU threads are used. This is not process memory isolation.

GPU search supplies candidate controllers. TypeScript independently evaluates every
final candidate on discovery seeds, recomputes descriptors, rebins and resolves
collisions before choosing published cells. A minimum of six occupied cells across
two coverage bins and three turning bins is required. If discovery fails this
gate, the declared extensions are 48 then 96 generations, before looking at any
held-out result. The initial 24-generation run passed, so no extension was used.

The published artifact retains all final candidates, their original GPU metrics,
canonical discovery evaluations, selected cells, weights, source graph bundle,
runtime/configuration identities, and per-seed held-out results. It can be checked
without access to an untracked training directory. It does not claim that final
elites alone can reconstruct every discarded intermediate search candidate.

## Controls and interactive experiments

Every selected readout is evaluated on the same held-out seeds in three conditions:
intact graph, disconnected graph, and zeroed inputs to the readout. Weights are not
retrained for these controls. An authored-decoder reference uses identical seeds.
Biases remain active under zero input. Behavior that survives this control must
not be attributed entirely to neural propagation.

The initial replay always uses seed 62001, selected before evaluation; it is not
the best-performing seed. Food/hazard positions belong to that actual trajectory.
Playback samples at most 61 frames over the full episode.

The initial probe seed matches the displayed replay seed (62001); it can be changed.
After selection, the existing counterfactual engine runs a new experiment with
the selected readout during warmup and in every future. Full world state, neural
rates and RNG state are cloned at the fork. Baseline and sham are identical;
the selected group is clamped to zero before the first neural substep and after
every substep in the lesion branch. Different contacts may subsequently produce
different food placements. Group labels are authored channel/bridge/output
mappings, not anatomical regions. The disconnected topology is also available;
rewired transfer is not part of this atlas's interactive contract.

The UI exposes 4–16 matched seeds, fork ticks 0–300 and horizons 30–300. Results
show paired effects and descriptive intervals. Repeated exploratory probes are
not corrected for multiple comparisons. Workers are terminated on cancellation,
selection changes or navigation; a late response cannot overwrite a new choice.

Every request pins the atlas SHA-256 displayed when the controller was selected.
If a deployment changes the atlas before the worker loads it, the probe refuses
to run and asks for a reload/reselection. Exports include controller ID, exact
weights, weights hash, atlas hash, graph identity, target neuron IDs, settings,
seed manifest, outcomes and replay frames.

## Reproduction

Use Node 22 and the locked `training/` environment. This host requires disabling
Datadog injection; `training/scripts/run.sh` does this. No beans command or backend
service is involved.

```bash
npm ci
npm run training:export-arms -- --graph public/data/malecns-arena-v1.bin.gz \
  --out training/runs/atlas/arms
# Use the biological.json inside the hash-named directory printed by the exporter.
cd training
./scripts/run.sh flyarena-atlas --graph runs/atlas/arms/GRAPH_SHA/biological.json \
  --output runs/atlas/discovery-24.json
cd ..
npm run atlas:publish -- --input training/runs/atlas/discovery-24.json
npm run atlas:verify
```

Search refuses an existing output path; use a new run filename. CUDA is explicit
and never falls back silently. `--device cpu` is available for small diagnostics.
`--population 4 --generations 2 --ticks 30` creates a smoke run; publication's
`--diagnostic` option reports GPU/TypeScript differences without publishing it.

To export or verify a selected-controller probe:

```bash
npm run experiment:counterfactual -- --controller 1316 --seed 62001 --seeds 4 \
  --warmup 30 --horizon 60 --target output --output /tmp/atlas-probe.json
npm run experiment:counterfactual -- --verify /tmp/atlas-probe.json
```

Use an ID in the current atlas; controller 1316 is the shipped default. Export
verification resolves against the locally pinned atlas and rejects changed
controller/atlas/graph metadata before simulation. Same-host exact reproduction
is the primary gate. The existing `--compare-numerical` option remains an explicitly
labeled diagnostic with narrow continuous-field tolerances, not exact reproduction.

`atlas:verify` reruns canonical selection, all public held-out results, controls and
replays. It can take several minutes. GPU repeatability is measured separately;
CUDA/version changes may alter archive trajectories even with an identical RNG
recipe. Long coupled trajectories can diverge between GPU and TypeScript, which
is why GPU scores never become the public canonical scores without recomputation.

## Deployment and provenance

The atlas and browser probes are static assets. Existing `scripts/deploy.sh`
already packages them, so no server deployment changes are needed. Use a clean
checkout of the tested integrated commit, deploy after review and merge, and verify
the public atlas/probe journey. The existing release-symlink rollback also rolls
back atlas data and UI together. Local publication writes artifact then manifest;
a concurrent development fetch can fail integrity during that short transition
and should retry. Production deployment activates the complete built release.

The graph retains its existing dataset attribution and license, available through
the model ledger and graph manifest. Method source: [Mouret and Clune,
MAP-Elites](https://arxiv.org/abs/1504.04909). Hardware source:
[NVIDIA DGX Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
Measured validation and review evidence is in [the validation record](behavior-atlas-validation.md).
