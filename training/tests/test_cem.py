"""WP3 acceptance tests (`.agents/plans/trained-readout/03-cem-training.md`'s
"Tests and acceptance" section):

- CEM converges on a synthetic quadratic objective in <= 50 generations.
- On the trace graph with G=3, P=16, E=2, the CLI writes all run files.
- The held-out assertion fires if a held-out seed is injected.
- `readoutParameterCount(D, H)` matches a value exported by
  `export-traces.ts` (the golden readout case).
- CPU reproducibility: the same trainer_seed/bundle/config reproduces
  `theta_final` bit-identically (blocking gate).
- GPU rerun `theta_final` max-abs-diff is measured and recorded
  (informational, no gate).
- End-to-end contract: a run directory the Python CLI writes loads and
  scores under the real TypeScript evaluator (`scripts/training/evaluate.ts`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from flyarena_training import cli
from flyarena_training.cem import CemConfig, run_cem
from flyarena_training.graph import load_graph_json, output_neuron_indices
from flyarena_training.readout import readout_parameter_count
from flyarena_training.rollout import assert_no_held_out_seeds, build_rollout_env, evaluate_fitness
from flyarena_training.seeds import sample_training_seeds

# training/tests/test_cem.py -> training/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_GRAPH_PATH = REPO_ROOT / "tests" / "fixtures" / "golden" / "trace-graph.json"
TRACE_GRAPH_READOUT_PATH = REPO_ROOT / "tests" / "fixtures" / "golden" / "trace-graph-readout.json"

# Same pinned-nvm convention as conftest.py's `_find_node_bin_dir` (kept
# local here rather than imported from conftest.py, which is WP2's parity
# suite infra and not in this WP's change surface).
PINNED_NODE_BIN_DIR = Path.home() / ".nvm" / "versions" / "node" / "v22.22.3" / "bin"


def _find_node_bin_dir() -> str | None:
    found = shutil.which("node")
    if found:
        return str(Path(found).parent)
    if (PINNED_NODE_BIN_DIR / "node").exists():
        return str(PINNED_NODE_BIN_DIR)
    return None


def _build_synthetic_trace_bundle(arm: str = "biological") -> dict:
    """Wraps the committed `trace-graph.json` fixture (a raw graph, no
    `export-arms.ts` bundle envelope) with the extra bundle fields
    `cli.py` reads (`arm`, `graphSource`, etc.), so this test does not need
    Node/`npm run training:export-arms` to exercise the CLI's own file
    writing. `graphArtifactSha256`/`sha256` are placeholders: `cli.py` never
    verifies bundle integrity (only the real `evaluate.ts` does — see
    `test_end_to_end_contract_with_ts_evaluator` below, which uses a real
    export-arms.ts-produced bundle instead)."""
    graph = json.loads(TRACE_GRAPH_PATH.read_text())
    indices = [i for i, v in enumerate(graph["outputPopulationIndex"]) if v >= 0]
    bundle = dict(graph)
    bundle.update(
        {
            "formatVersion": 1,
            "arm": arm,
            "graphId": "trace-graph",
            "graphSource": "trace-graph-fixture",
            "graphArtifactSha256": "a" * 64,
            "provenance": {"kind": "biological-trace-graph-fixture"},
            "D": len(indices),
            "outputNeuronIndices": indices,
            "sha256": "b" * 64,
        }
    )
    return bundle


def _write_bundle(tmp_path: Path, bundle: dict, name: str = "biological.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(bundle))
    return path


# ---------------------------------------------------------------------------
# Sanity: CEM converges on a synthetic objective.
# ---------------------------------------------------------------------------


def test_cem_converges_on_synthetic_quadratic():
    """`evaluate` is a pure function of theta (fitness = -||theta -
    target||^2), ignoring the sampled seeds entirely — this isolates the CEM
    optimizer itself from the rollout. Convergence must happen in
    <= 50 generations (03-cem-training.md's acceptance bound). `run_cem`'s
    own reproducibility is governed entirely by `trainer_seed` (a
    `torch.Generator` it owns), not any global RNG state, so no global seed
    call is needed here."""
    target = torch.tensor([0.3, -0.2, 0.1, 0.5, -0.4, 0.2, 0.0, -0.1], dtype=torch.float32)
    dim = target.numel()

    def evaluate(theta_batch: torch.Tensor, seeds) -> torch.Tensor:
        assert_no_held_out_seeds(seeds)  # exercised on every call, even for a toy objective
        return -((theta_batch - target) ** 2).sum(dim=1)

    config = CemConfig(population=64, elites=16, generations=50, training_seeds_per_generation=2, init_std=0.5)
    result = run_cem(dim, evaluate, config, trainer_seed=42, device="cpu")

    assert len(result.history) == 50
    final_distance = (result.theta_final - target).norm().item()
    assert final_distance < 0.1, f"CEM did not converge within 50 generations: final distance {final_distance}"
    # Fitness should trend upward (mean_fitness's sign is negative distance,
    # so it increases toward 0 as theta approaches target).
    assert result.history[-1].mean_fitness > result.history[0].mean_fitness


# ---------------------------------------------------------------------------
# Held-out seed assertion.
# ---------------------------------------------------------------------------


def test_assert_no_held_out_seeds_raises_for_held_out_seed():
    with pytest.raises(AssertionError, match="held-out range"):
        assert_no_held_out_seeds([1, 2, 30050])


def test_assert_no_held_out_seeds_allows_training_and_validation_ranges():
    assert_no_held_out_seeds([1, 5000, 10000])  # training range
    assert_no_held_out_seeds([20001, 20064])  # validation range
    assert_no_held_out_seeds([30000, 30101])  # just outside the held-out band on both sides


def test_cem_held_out_injection_fires_the_assertion():
    """A `CemConfig` whose training seed range is set (by the caller) to lie
    entirely inside the held-out band must make `run_cem` raise on its first
    generation — "the trainer asserts that no held-out seed enters any
    batch" (03-cem-training.md)."""
    config = CemConfig(
        population=4, elites=1, generations=1, training_seeds_per_generation=5, training_seed_low=30001, training_seed_high=30100
    )

    def evaluate(theta_batch: torch.Tensor, seeds) -> torch.Tensor:
        return -((theta_batch) ** 2).sum(dim=1)

    with pytest.raises(AssertionError, match="held-out range"):
        run_cem(3, evaluate, config, trainer_seed=1, device="cpu")


def test_evaluate_fitness_held_out_seed_raises():
    """The rollout's own held-out guard (`rollout.py`'s "defense in depth" —
    it refuses a held-out seed even independent of `cem.py`'s own seed
    sampling), exercised directly rather than only through `run_cem`."""
    graph = load_graph_json(TRACE_GRAPH_PATH, device="cpu")
    env = build_rollout_env(graph, device="cpu")
    theta = torch.zeros(1, readout_parameter_count(env.input_size, 4))
    with pytest.raises(AssertionError, match="held-out range"):
        evaluate_fitness(env, theta, [30050], hidden_size=4, ticks=5, substeps=4)


def test_cem_config_rejects_validation_range_overlapping_training_range():
    with pytest.raises(ValueError, match="overlaps"):
        CemConfig(training_seed_low=1, training_seed_high=25000)  # overlaps the default validation range


def test_cem_config_rejects_non_finite_and_negative_hyperparameters():
    for kwargs in (
        {"init_std": float("nan")},
        {"init_std": float("inf")},
        {"init_std": -0.1},
        {"std_floor": float("nan")},
        {"alpha": float("nan")},
        {"validation_seed_count": 0},
        {"validation_seed_count": -1},
    ):
        with pytest.raises(ValueError):
            CemConfig(**kwargs)


def test_run_cem_rejects_negative_trainer_seed():
    def evaluate(theta_batch: torch.Tensor, seeds) -> torch.Tensor:
        return -((theta_batch) ** 2).sum(dim=1)

    with pytest.raises(ValueError, match="non-negative"):
        run_cem(3, evaluate, CemConfig(population=4, elites=1, generations=1), trainer_seed=-1, device="cpu")


def test_run_cem_rejects_non_finite_fitness_from_evaluate():
    """`evaluate()` returning `NaN` must raise, not silently become an elite
    (`torch.topk` ranks `NaN` as the largest value)."""

    def evaluate(theta_batch: torch.Tensor, seeds) -> torch.Tensor:
        fitness = -((theta_batch) ** 2).sum(dim=1)
        fitness[0] = float("nan")
        return fitness

    with pytest.raises(ValueError, match="non-finite"):
        run_cem(3, evaluate, CemConfig(population=4, elites=1, generations=1), trainer_seed=1, device="cpu")


def test_cem_alpha_weights_the_new_elite_estimate():
    """Pins the smoothing direction explicitly (Important finding from
    review: the plan states only "smoothing alpha = 0.7" with no formula).
    `P=2`, `elites=1`, one generation, a deterministic `evaluate` that always
    prefers candidate 0: after smoothing, `mean` must equal `alpha *
    candidate_0 + (1 - alpha) * init_mean` — i.e. `alpha` weights the *new*
    elite estimate (the standard smoothed-CEM convention), not the old mean."""
    alpha = 0.7
    config = CemConfig(population=2, elites=1, generations=1, alpha=alpha, init_std=0.5, init_mean=0.0)

    def evaluate(theta_batch: torch.Tensor, seeds) -> torch.Tensor:
        # Candidate 0 always wins; candidate 1 always loses.
        return torch.tensor([1.0, 0.0])

    result = run_cem(3, evaluate, config, trainer_seed=7, device="cpu")

    generator = torch.Generator(device="cpu").manual_seed(7)
    noise = torch.randn(2, 3, generator=generator, device="cpu", dtype=torch.float32)
    candidate_0 = noise[0] * config.init_std  # init_mean is 0
    expected_mean = alpha * candidate_0 + (1 - alpha) * torch.zeros(3)

    assert torch.allclose(result.theta_final, expected_mean, atol=1e-6), (
        f"expected alpha to weight the NEW elite estimate: {result.theta_final} != {expected_mean}"
    )


def test_cem_update_with_multiple_elites_matches_hand_computed_mean_and_std():
    """Regression test for `run_cem`'s elite-selection/smoothing update with
    `elites > 1` (review suggestion: every existing update-formula test used
    `elites=1`, so an off-by-one in the `topk` count, or an accidental
    `population`-sized elite set, would not be caught). `P=4`, `elites=2`,
    2 generations, a deterministic `evaluate` that always ranks candidates
    1 and 3 as the winners (independent of candidate value, isolating the
    CEM update from the rollout, and deliberately not candidates 0/1 so an
    accidental "first N candidates" bug would also be caught).

    Hand-computes the expected `mean`/`std` trajectory by replaying the
    exact same `torch.Generator` draw sequence and update formula `run_cem`
    uses (population std for the elite std, `alpha` weighting the *new*
    elite estimate for both `mean` and `std`, `std` floored at
    `config.std_floor`). `std_floor=5.0` is set far above the naturally
    smoothed std at `init_std=0.5` (verified: the unclamped smoothed std is
    ~0.2-0.5 in generation 0), so the floor clamp is guaranteed to fire —
    exercising that path, not just the smoothing arithmetic — and generation
    1's candidates are drawn using the *clamped* std, so a wrong or missing
    clamp would change generation 1's candidates and therefore
    `theta_final`, not just an intermediate value this test can't observe
    directly (`CemResult` does not expose `std`)."""
    theta_dim = 3
    trainer_seed = 11
    alpha = 0.7
    init_mean = 0.0
    init_std = 0.5
    std_floor = 5.0
    population = 4
    elites = 2
    config = CemConfig(
        population=population,
        elites=elites,
        generations=2,
        alpha=alpha,
        init_std=init_std,
        init_mean=init_mean,
        std_floor=std_floor,
    )

    fitness_by_index = torch.tensor([1.0, 4.0, 2.0, 3.0])  # candidates 1, 3 win every generation

    def evaluate(theta_batch: torch.Tensor, seeds) -> torch.Tensor:
        if theta_batch.shape[0] == 1:
            return torch.zeros(1)  # the validation call (mean only); value unused by this test
        return fitness_by_index

    result = run_cem(theta_dim, evaluate, config, trainer_seed=trainer_seed, device="cpu")

    # Reference implementation: same generator seed, same per-generation
    # sequential torch.randn(population, theta_dim) draw run_cem makes, same
    # elite indices ([1, 3], matching fitness_by_index's top-2), same
    # smoothing/floor formula.
    generator = torch.Generator(device="cpu").manual_seed(trainer_seed)
    mean = torch.full((theta_dim,), init_mean)
    std = torch.full((theta_dim,), init_std)
    for _ in range(config.generations):
        noise = torch.randn(population, theta_dim, generator=generator, dtype=torch.float32)
        candidates = mean.unsqueeze(0) + noise * std.unsqueeze(0)
        elite_candidates = candidates[[1, 3]]
        new_mean = elite_candidates.mean(dim=0)
        new_std = elite_candidates.std(dim=0, unbiased=False)
        mean = alpha * new_mean + (1 - alpha) * mean
        std = (alpha * new_std + (1 - alpha) * std).clamp(min=std_floor)

    assert std.eq(std_floor).all(), "test setup must force the std_floor clamp to fire (see docstring)"
    assert torch.allclose(result.theta_final, mean, atol=1e-6), (
        f"multi-elite CEM update mismatch: {result.theta_final} != hand-computed {mean}"
    )


# ---------------------------------------------------------------------------
# Seed policy: replica independence (review Important finding).
# ---------------------------------------------------------------------------


def test_sample_training_seeds_is_replica_independent_across_defaults():
    """Regression test for the review's Important replica-independence
    finding: the plan's literal `numpy.random.default_rng(trainer_seed +
    generation)` formula aliases whenever two replicas' `trainer_seed`s
    differ by less than `G` -- at the plan's own default replica seeds
    (101, 202, 303) and `G=150`, replica 101's generation `g >= 101` drew
    EXACTLY the same 16-seed training set as replica 202's generation
    `g - 101` (49/150 = 32.7% of generations for each adjacent pair, per the
    review's empirical table), undercutting the plan's "R = 3 independent
    trainer_seed values per arm" framing.

    `seeds.sample_training_seeds` keys `numpy.random.default_rng` on the
    entropy tuple `[trainer_seed, generation]` instead (see `seeds.py`'s doc
    comment) -- this test asserts that fix directly: for every pair of the
    plan's default replica seeds, no generation's training-seed set
    (`g in [0, G)`) for one replica ever equals ANY generation's
    training-seed set for another replica (compared as an unordered set,
    since the old formula's collision was drawing the identical 16-seed
    set, not merely the identical list order). It also re-confirms held-out
    isolation still holds under the new formula (`assert_no_held_out_seeds`
    does not raise for any sampled seed), and sanity-checks that the OLD
    formula would still exhibit the exact 49/150 collision count the review
    measured -- so this test would actually fail if the fix were reverted or
    never applied, not just vacuously pass."""
    trainer_seeds = (101, 202, 303)
    generations = 150
    e = 16

    seed_sets_by_trainer_seed: dict[int, list[frozenset[int]]] = {}
    for trainer_seed in trainer_seeds:
        per_generation: list[frozenset[int]] = []
        for generation in range(generations):
            sampled = sample_training_seeds(trainer_seed, generation, e)
            assert len(sampled) == e
            assert len(set(sampled)) == e, "sampled without replacement"
            assert_no_held_out_seeds(sampled)  # held-out isolation still holds
            per_generation.append(frozenset(sampled))
        seed_sets_by_trainer_seed[trainer_seed] = per_generation

    for i, seed_a in enumerate(trainer_seeds):
        for seed_b in trainer_seeds[i + 1 :]:
            sets_a = seed_sets_by_trainer_seed[seed_a]
            sets_b = seed_sets_by_trainer_seed[seed_b]
            for generation_a, set_a in enumerate(sets_a):
                for generation_b, set_b in enumerate(sets_b):
                    assert set_a != set_b, (
                        f"trainer_seed={seed_a} generation={generation_a} drew the same training-seed "
                        f"set as trainer_seed={seed_b} generation={generation_b}: {sorted(set_a)}"
                    )

    # Sanity check: the OLD (reverted) formula should still show the review's
    # measured 49/150 collision count between adjacent replica pairs -- this
    # confirms the assertions above are actually exercising the fix, not
    # passing vacuously for an unrelated reason.
    def _old_formula_seed_set(trainer_seed: int, generation: int) -> frozenset[int]:
        rng = np.random.default_rng(trainer_seed + generation)
        return frozenset(int(s) for s in rng.choice(np.arange(1, 10001), size=e, replace=False))

    old_collisions_101_202 = sum(
        1 for g in range(101, generations) if _old_formula_seed_set(101, g) == _old_formula_seed_set(202, g - 101)
    )
    old_collisions_202_303 = sum(
        1 for g in range(101, generations) if _old_formula_seed_set(202, g) == _old_formula_seed_set(303, g - 101)
    )
    assert old_collisions_101_202 == 49
    assert old_collisions_202_303 == 49


# ---------------------------------------------------------------------------
# Batched rollout: row alignment and held-out guard.
# ---------------------------------------------------------------------------


def test_evaluate_fitness_batch_rows_match_individual_candidates():
    """Regression test for `evaluate_fitness`'s row-tiling contract
    (`_expand_weights`'s `repeat_interleave` against `list(seeds) * p`):
    batching P candidates x E seeds into one call must give bit-identical
    per-candidate fitness to calling each candidate alone."""
    graph = load_graph_json(TRACE_GRAPH_PATH, device="cpu")
    env = build_rollout_env(graph, device="cpu")
    hidden_size = 4
    parameter_count = readout_parameter_count(env.input_size, hidden_size)
    seeds = [1, 2, 3]

    torch.manual_seed(0)
    theta = torch.randn(3, parameter_count) * 0.3

    batched = evaluate_fitness(env, theta, seeds, hidden_size, ticks=15, substeps=4)
    individual = torch.stack(
        [evaluate_fitness(env, theta[i : i + 1], seeds, hidden_size, ticks=15, substeps=4)[0] for i in range(3)]
    )
    assert torch.equal(batched, individual)

    # Reversing candidate order must permute the result identically (catches
    # an accidental seed/candidate index swap that a same-value coincidence
    # in the forward-order check above could mask).
    reversed_batched = evaluate_fitness(env, theta.flip(0), seeds, hidden_size, ticks=15, substeps=4)
    assert torch.equal(reversed_batched, batched.flip(0))


# ---------------------------------------------------------------------------
# Parameter-count cross-check against a value exported by export-traces.ts.
# ---------------------------------------------------------------------------


def test_readout_parameter_count_matches_exported_golden_readout_case():
    """`trace-graph-readout.json` (`scripts/training/export-traces.ts`'s
    committed golden readout case) declares `inputSize`/`hiddenSize` and
    concrete `w1`/`b1`/`w2`/`b2` arrays; their lengths sum to the real,
    TS-produced parameter count for that (D, H) pair. This cross-checks
    `readout_parameter_count` (this port) against that externally-produced
    value, independent of this port's own formula."""
    case = json.loads(TRACE_GRAPH_READOUT_PATH.read_text())
    weights = case["weights"] if "weights" in case else case
    input_size = weights["inputSize"]
    hidden_size = weights["hiddenSize"]
    exported_count = len(weights["w1"]) + len(weights["b1"]) + len(weights["w2"]) + len(weights["b2"])

    assert readout_parameter_count(input_size, hidden_size) == exported_count

    graph = load_graph_json(TRACE_GRAPH_PATH, device="cpu")
    d = int(output_neuron_indices(graph).numel())
    assert d == input_size, "trace-graph-readout.json's inputSize must match trace-graph.json's D"


# ---------------------------------------------------------------------------
# CLI: trace-graph run writes all files (G=3, P=16, E=2).
# ---------------------------------------------------------------------------


# Episode length is shortened from the plan's T=1800 for this fast
# CLI-file-writing check only: the plan's acceptance bullet pins G/P/E, not
# T, and a full-length episode at these tiny P/E would add wall time with no
# additional coverage of "does the CLI write every file". T=1800 is
# exercised by the calibration run (README) and is this CLI's own default.
_FAST_TEST_TICKS = 20


def test_cli_trace_graph_run_writes_all_files(tmp_path):
    bundle_path = _write_bundle(tmp_path, _build_synthetic_trace_bundle())
    out_dir = tmp_path / "run"

    args = cli.parse_args(
        [
            "--arm", "biological",
            "--graph", str(bundle_path),
            "--replica-seed", "101",
            "--out", str(out_dir),
            "--population", "16",
            "--elites", "4",
            "--generations", "3",
            "--train-seeds-per-generation", "2",
            "--ticks", str(_FAST_TEST_TICKS),
            "--hidden-size", "4",
            "--device", "cpu",
        ]
    )
    result = cli.run_training(args)

    assert result.theta_final_path.exists()
    assert result.theta_best_path.exists()
    assert result.config_path.exists()
    assert result.generations_csv_path.exists()
    assert result.env_path.exists()

    config = json.loads(result.config_path.read_text())
    assert config["arm"] == "biological"
    assert config["trainerSeed"] == 101
    assert config["D"] == 6
    assert config["H"] == 4
    assert config["parameterCount"] == readout_parameter_count(6, 4)
    assert config["substeps"] == 4  # TRACE_SUBSTEPS default (trace-graph-fixture bundle)
    # Regression: `armBundleSha256` must always be the bundle's own sha256,
    # never silently dropped — `evaluate.ts`'s `loadArmGraphs` cross-checks
    # this field against the loaded bundle as its "trained against the right
    # bundle" integrity check (review finding: a `None`-value filter used to
    # drop this field whenever it happened to be missing).
    bundle_sha256 = json.loads(bundle_path.read_text())["sha256"]
    assert config["armBundleSha256"] == bundle_sha256 == "b" * 64

    theta_final = np.load(result.theta_final_path)
    assert theta_final.shape == (config["parameterCount"],)
    assert theta_final.dtype == np.float32

    generations_csv = result.generations_csv_path.read_text().strip().splitlines()
    assert generations_csv[0] == "generation,meanFitness,maxFitness,validationFitness"
    assert len(generations_csv) == 1 + 3  # header + 3 generations

    env = json.loads(result.env_path.read_text())
    assert env["torchVersion"] == torch.__version__
    assert "precision" in env


def test_cli_production_graph_requires_explicit_substeps(tmp_path):
    """A bundle with `graphSource: "artifact"` (a production graph, not
    trace-graph development) must refuse to run without `--substeps`."""
    bundle = _build_synthetic_trace_bundle()
    bundle["graphSource"] = "artifact"
    bundle["provenance"] = {"kind": "biological-artifact", "artifactPath": "n/a", "artifactSha256": "c" * 64}
    bundle_path = _write_bundle(tmp_path, bundle)

    args = cli.parse_args(
        ["--arm", "biological", "--graph", str(bundle_path), "--replica-seed", "101", "--out", str(tmp_path / "out"), "--device", "cpu"]
    )
    with pytest.raises(SystemExit, match="substeps is required"):
        cli.run_training(args)


def _base_cli_args(bundle_path: Path, out_dir: Path, **overrides: str) -> list[str]:
    values = {
        "--arm": "biological",
        "--graph": str(bundle_path),
        "--replica-seed": "101",
        "--out": str(out_dir),
        "--population": "4",
        "--elites": "1",
        "--generations": "1",
        "--train-seeds-per-generation": "2",
        "--ticks": str(_FAST_TEST_TICKS),
        "--hidden-size": "4",
        "--device": "cpu",
    }
    values.update(overrides)
    argv: list[str] = []
    for flag, value in values.items():
        argv.extend([flag, value])
    return argv


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--replica-seed", "0"),
        ("--replica-seed", "-1"),
        ("--hidden-size", "0"),
        ("--hidden-size", "-1"),
        ("--ticks", "0"),
        ("--ticks", "-1"),
    ],
)
def test_cli_rejects_non_positive_values_before_writing_any_file(tmp_path, flag, bad_value):
    """`run-dir.ts`'s `POSITIVE_INT_RUN_CONFIG_FIELDS` rejects a non-positive
    trainerSeed/H/substeps; this CLI must refuse the same inputs itself,
    before training (not merely leave the evaluator to reject the output
    afterward)."""
    bundle_path = _write_bundle(tmp_path, _build_synthetic_trace_bundle())
    out_dir = tmp_path / "out"
    args = cli.parse_args(_base_cli_args(bundle_path, out_dir, **{flag: bad_value}))
    with pytest.raises(ValueError, match="positive integer"):
        cli.run_training(args)
    assert not out_dir.exists() or not any(out_dir.iterdir()), "no output file should be written on a rejected run"


def test_cli_rejects_non_positive_substeps_even_for_production_graph(tmp_path):
    bundle = _build_synthetic_trace_bundle()
    bundle["graphSource"] = "artifact"
    bundle["provenance"] = {"kind": "biological-artifact", "artifactPath": "n/a", "artifactSha256": "c" * 64}
    bundle_path = _write_bundle(tmp_path, bundle)
    out_dir = tmp_path / "out"
    args = cli.parse_args(_base_cli_args(bundle_path, out_dir, **{"--substeps": "0"}))
    with pytest.raises(ValueError, match="positive integer"):
        cli.run_training(args)


def test_cli_rejects_bundle_missing_required_fields(tmp_path):
    bundle = _build_synthetic_trace_bundle()
    del bundle["sha256"]
    bundle_path = _write_bundle(tmp_path, bundle)
    args = cli.parse_args(_base_cli_args(bundle_path, tmp_path / "out"))
    with pytest.raises(ValueError, match="missing bundle field"):
        cli.run_training(args)


def test_cli_rejects_arm_mismatch(tmp_path):
    bundle_path = _write_bundle(tmp_path, _build_synthetic_trace_bundle(arm="biological"))
    args = cli.parse_args(_base_cli_args(bundle_path, tmp_path / "out", **{"--arm": "rewired"}))
    with pytest.raises(ValueError, match="does not match the bundle"):
        cli.run_training(args)


def test_cli_rejects_bundle_d_mismatch(tmp_path):
    """A bundle whose self-declared `D`/`outputNeuronIndices` disagree with
    what this loader computes from the graph arrays (corruption, or a stale
    bundle) must be refused rather than silently trained against the
    recomputed value."""
    bundle = _build_synthetic_trace_bundle()
    bundle["D"] = 999
    bundle_path = _write_bundle(tmp_path, bundle)
    args = cli.parse_args(_base_cli_args(bundle_path, tmp_path / "out"))
    with pytest.raises(ValueError, match="disagree"):
        cli.run_training(args)


def test_cli_reused_out_dir_leaves_no_stale_config_on_crash(tmp_path, monkeypatch):
    """Regression for the review finding that writing `config.json` last
    only protects a *fresh* `--out`: reusing `--out` for a second run whose
    write fails partway through — after `theta_final.npy`/`theta_best.npy`
    are overwritten with the new run's weights, but before this run's own
    `config.json` is written — must leave `--out` with NO `config.json` at
    all, never the *first* run's stale one now paired with the second run's
    weights."""
    bundle_path = _write_bundle(tmp_path, _build_synthetic_trace_bundle())
    out_dir = tmp_path / "run"

    args1 = cli.parse_args(_base_cli_args(bundle_path, out_dir, **{"--replica-seed": "101"}))
    cli.run_training(args1)
    first_config = json.loads((out_dir / "config.json").read_text())
    assert first_config["trainerSeed"] == 101
    first_theta_bytes = (out_dir / "theta_final.npy").read_bytes()

    # Fail json.dumps on its first call within the second run (env.json,
    # written before config.json) — simulates a crash after theta files are
    # already overwritten but before this run's own config.json exists.
    original_dumps = cli.json.dumps
    call_count = {"n": 0}

    def failing_dumps(*dumps_args, **dumps_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated write failure")
        return original_dumps(*dumps_args, **dumps_kwargs)

    monkeypatch.setattr(cli.json, "dumps", failing_dumps)
    args2 = cli.parse_args(_base_cli_args(bundle_path, out_dir, **{"--replica-seed": "202"}))
    with pytest.raises(RuntimeError, match="simulated write failure"):
        cli.run_training(args2)

    # The second run got far enough to overwrite theta_final.npy...
    assert (out_dir / "theta_final.npy").read_bytes() != first_theta_bytes
    # ...but config.json must be entirely absent, never the stale
    # trainerSeed=101 config now paired with trainerSeed=202's weights.
    assert not (out_dir / "config.json").exists()


# ---------------------------------------------------------------------------
# Reproducibility: CPU bit-identity (blocking gate) and GPU informational.
# ---------------------------------------------------------------------------

_REPRO_ARGS = [
    "--arm", "biological",
    "--replica-seed", "101",
    "--population", "16",
    "--elites", "4",
    "--generations", "3",
    "--train-seeds-per-generation", "2",
    "--ticks", str(_FAST_TEST_TICKS),
    "--hidden-size", "4",
]


def test_cpu_theta_final_bit_identical_across_two_runs(tmp_path):
    """Blocking gate (03-cem-training.md's "Reproducibility" section): on
    CPU, the same trainer_seed, graph bundle, and config reproduce
    theta_final bit-identically."""
    bundle_path = _write_bundle(tmp_path, _build_synthetic_trace_bundle())

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    args1 = cli.parse_args(["--graph", str(bundle_path), "--out", str(out1), "--device", "cpu", *_REPRO_ARGS])
    args2 = cli.parse_args(["--graph", str(bundle_path), "--out", str(out2), "--device", "cpu", *_REPRO_ARGS])

    cli.run_training(args1)
    cli.run_training(args2)

    theta1 = np.load(out1 / "theta_final.npy")
    theta2 = np.load(out2 / "theta_final.npy")
    assert np.array_equal(theta1, theta2), "CPU reruns of the same trainer_seed/bundle/config must be bit-identical"

    # Byte-identical, not just numerically equal (stronger check on the
    # actual file this evaluator reads).
    assert (out1 / "theta_final.npy").read_bytes() == (out2 / "theta_final.npy").read_bytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_gpu_theta_final_rerun_diff_is_informational(tmp_path):
    """GPU rerun agreement is informational (measured and recorded, not
    gated — 03-cem-training.md: "GPU rerun agreement is informational... not
    gated")."""
    bundle_path = _write_bundle(tmp_path, _build_synthetic_trace_bundle())

    out1 = tmp_path / "gpu-run1"
    out2 = tmp_path / "gpu-run2"
    args1 = cli.parse_args(["--graph", str(bundle_path), "--out", str(out1), "--device", "cuda", *_REPRO_ARGS])
    args2 = cli.parse_args(["--graph", str(bundle_path), "--out", str(out2), "--device", "cuda", *_REPRO_ARGS])

    cli.run_training(args1)
    cli.run_training(args2)

    theta1 = np.load(out1 / "theta_final.npy")
    theta2 = np.load(out2 / "theta_final.npy")
    max_abs_diff = float(np.abs(theta1 - theta2).max())
    print(f"[gpuRerunMaxAbsDiff] {max_abs_diff:.6e} (informational, no gate)")
    assert np.isfinite(max_abs_diff)


# ---------------------------------------------------------------------------
# End-to-end contract: Python CLI run dir -> real TypeScript evaluator.
# ---------------------------------------------------------------------------


def test_end_to_end_contract_with_ts_evaluator(tmp_path):
    """Produces a tiny trace-graph run directory with the Python CLI, then
    runs the real `scripts/training/evaluate.ts` (via `npm run
    training:evaluate`) on it and confirms it loads, validates, and scores
    it — and that the score it reports for `theta_final` numerically matches
    what `rollout.evaluate_fitness` (this trainer's own optimized objective)
    computes for the same theta and seeds, not merely that some finite
    number came back. Requires Node (skips with the manual command
    otherwise, matching `conftest.py`'s convention) and a real
    `export-arms.ts` bundle (unlike the CLI-only tests above, `evaluate.ts`
    verifies the bundle's own self-certifying sha256, so a hand-rolled
    placeholder bundle would fail its integrity check).

    Both the exporter and the evaluator are pointed at `tmp_path` (`--out`
    for `export-arms.ts`, `--arms-dir` for `evaluate.ts`) rather than the
    repo's shared `training/runs/arms/`: `export-arms.ts` deletes any arm
    bundle it did not just write ("stale-bundle cleanup"), so running it
    with no `--out` against the shared directory would delete a developer's
    own `--fixture-rewire` bundle there, and could race a concurrent test
    run touching the same files.
    """
    node_bin_dir = _find_node_bin_dir()
    if node_bin_dir is None:
        pytest.skip(
            "Node not found on PATH or at the pinned nvm location; cannot run this contract test. "
            "Run manually from the repo root: npm run training:export-arms -- --out <tmp-arms> && "
            "uv run --project training python -m flyarena_training.cli --arm biological "
            "--graph <tmp-arms>/<sha>/biological.json --replica-seed 101 --out <tmp> "
            "--population 8 --elites 2 --generations 2 --train-seeds-per-generation 2 --ticks 20 "
            "--hidden-size 4 --device cpu && "
            "npm run training:evaluate -- --runs <tmp> --arms-dir <tmp-arms>/<sha> --out <tmp-out> "
            "--ticks 20 --substeps 4 --held-out-start 5 --held-out-count 2 --bootstrap-resamples 10"
        )
    if not (REPO_ROOT / "node_modules").exists():
        pytest.skip("node_modules not installed (run `npm ci` from the repo root); cannot run this contract test.")

    env = dict(os.environ)
    env["PATH"] = node_bin_dir + os.pathsep + env.get("PATH", "")

    arms_root = tmp_path / "arms"
    export_result = subprocess.run(
        ["npm", "run", "training:export-arms", "--", "--out", str(arms_root)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    if export_result.returncode != 0:
        pytest.fail(f"export-arms failed (exit {export_result.returncode}):\n{export_result.stdout}\n{export_result.stderr}")

    sha_dirs = [p for p in arms_root.iterdir() if p.is_dir()]
    assert len(sha_dirs) == 1, f"expected exactly one graph-sha directory under {arms_root}, got {sha_dirs}"
    arms_dir = sha_dirs[0]
    bundle_path = arms_dir / "biological.json"
    assert bundle_path.exists(), f"expected bundle at {bundle_path}"

    run_dir = tmp_path / "run"
    hidden_size = 4
    ticks = 20
    substeps = 4
    args = cli.parse_args(
        [
            "--arm", "biological",
            "--graph", str(bundle_path),
            "--replica-seed", "101",
            "--out", str(run_dir),
            "--population", "8",
            "--elites", "2",
            "--generations", "2",
            "--train-seeds-per-generation", "2",
            "--ticks", str(ticks),
            "--hidden-size", str(hidden_size),
            "--device", "cpu",
        ]
    )
    cli.run_training(args)
    assert (run_dir / "theta_final.npy").exists()
    assert (run_dir / "config.json").exists()

    # A non-held-out seed band (evaluate.ts's --held-out-start/--held-out-count
    # is just a seed range to score, not itself an enforced-disjoint-from-
    # training set): using seeds outside [30001, 30100] lets this test also
    # compute the same fitness independently in Python below, which
    # `rollout.assert_no_held_out_seeds` would otherwise refuse for the real
    # held-out band.
    cross_check_start = 5
    cross_check_count = 2
    eval_out_dir = tmp_path / "eval-out"
    evaluate_result = subprocess.run(
        [
            "npm", "run", "training:evaluate", "--",
            "--runs", str(run_dir),
            "--arms-dir", str(arms_dir),
            "--out", str(eval_out_dir),
            "--ticks", str(ticks),
            "--substeps", str(substeps),
            "--held-out-start", str(cross_check_start),
            "--held-out-count", str(cross_check_count),
            "--bootstrap-resamples", "10",
        ],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=180,
    )
    assert evaluate_result.returncode == 0, (
        f"evaluate.ts failed (exit {evaluate_result.returncode}):\n{evaluate_result.stdout}\n{evaluate_result.stderr}"
    )

    report_path = eval_out_dir / "trained-readout-v1.report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "biological" in report["arms"]
    assert "101" in report["arms"]["biological"]["replicas"]
    trained_stats = report["arms"]["biological"]["replicas"]["101"]["trained"]
    assert trained_stats["n"] == cross_check_count
    assert np.isfinite(trained_stats["mean"])

    # Fitness-parity cross-check: independently recompute the same
    # (theta_final, seeds) fitness with this trainer's own rollout and
    # confirm it agrees with what evaluate.ts (the TypeScript-authoritative
    # scorer) reported for the identical condition. This is the test that
    # would actually catch a transposed weight, a substep dropped, an
    # observe-before/after-step ordering bug, or the wrong agent being
    # scored — none of which would make evaluate.ts's report merely absent
    # or non-finite.
    graph = load_graph_json(bundle_path, device="cpu")
    rollout_env = build_rollout_env(graph, device="cpu")
    theta = torch.from_numpy(np.load(run_dir / "theta_final.npy")).unsqueeze(0)
    cross_check_seeds = list(range(cross_check_start, cross_check_start + cross_check_count))
    python_fitness = evaluate_fitness(rollout_env, theta, cross_check_seeds, hidden_size, ticks, substeps)[0].item()
    assert python_fitness == pytest.approx(trained_stats["mean"], rel=1e-4, abs=1e-4), (
        f"Python rollout fitness {python_fitness} disagrees with evaluate.ts's reported mean "
        f"{trained_stats['mean']} for the same theta_final and seeds {cross_check_seeds}"
    )
