"""`flyarena-train` CLI: WP3's seeded CEM readout trainer entry point.

    flyarena-train --arm {biological,rewired,disconnected} \\
        --graph <bundle.json> --replica-seed N --out training/runs/<run-id>/

`--graph` is an `export-arms.ts` bundle JSON (`training/runs/arms/<graph
sha>/<arm>.json`, produced by `npm run training:export-arms`), never a
re-derived graph — "Arm CSR arrays exported from TypeScript/compiled
artifacts, not re-derived in Python"
(`.agents/plans/trained-readout/00-overview.md`'s key decisions). Because
`load_graph_json` (`graph.py`) validates only the fields a `ConnectomeGraph`
needs and a bundle is a strict superset of that schema (it adds `arm`,
`graphId`, `graphSource`, `graphArtifactSha256`, `provenance`, `D`,
`outputNeuronIndices`, and its own self-certifying `sha256`, on top of every
field `load_graph_json` reads), the same loader already reads a bundle
correctly with no format-specific branch.

`--substeps` (K) defaults to `TRACE_SUBSTEPS` only for a trace-graph bundle
(`graphSource == "trace-graph-fixture"`); a production bundle
(`graphSource == "artifact"`) must pass `--substeps` explicitly, or this CLI
refuses to run (03-cem-training.md: "Production runs must pass the
closed-loop value fixed by flyarena-bb45, and the CLI refuses a production
graph without an explicit --substeps").
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import PRECISION_APPLIED  # noqa: F401  (import applies TF32/determinism settings)
from .cem import CemConfig, CemResult, run_cem
from .graph import ConnectomeGraph, load_graph_json, output_neuron_indices
from .readout import readout_parameter_count
from .rollout import build_rollout_env, evaluate_fitness
from .seeds import HELD_OUT_SEED_COUNT, HELD_OUT_SEED_START

ARM_NAMES: tuple[str, ...] = ("biological", "rewired", "disconnected")

# Matches `export const TRACE_SUBSTEPS = NEURAL_SUBSTEPS_PER_TICK;`
# (`scripts/training/export-traces.ts`, itself `NEURAL_SUBSTEPS_PER_TICK`
# from `src/lib/connectome/constants.ts`, currently 4). Mirrored by hand,
# same convention `training/tests/test_parity.py` already uses for its own
# `TRACE_SUBSTEPS` constant. Used only as the trace-graph-development
# default; a production bundle must always pass --substeps explicitly (see
# this module's doc comment) regardless of whether this constant's numeric
# value happens to still match the real closed-loop K.
TRACE_SUBSTEPS = 4

DEFAULT_HIDDEN_SIZE = 16  # H (00-overview.md's single source of truth).
DEFAULT_TICKS = 1800  # T (03-cem-training.md: 60 simulated seconds at 1/30).

# Every field this CLI reads off a raw `export-arms.ts` `SerializedArmBundle`
# JSON (`cli.py`'s doc comment) — required present, or the input does not
# look like a real bundle (see run_training's "missing bundle field(s)"
# check). `D`/`outputNeuronIndices` are also cross-checked against what
# `output_neuron_indices` recomputes from the loaded graph arrays.
REQUIRED_BUNDLE_FIELDS: tuple[str, ...] = (
    "formatVersion",
    "arm",
    "graphId",
    "graphSource",
    "graphArtifactSha256",
    "sha256",
    "D",
    "outputNeuronIndices",
)


# `scripts/training/run-dir.ts`'s `POSITIVE_INT_RUN_CONFIG_FIELDS`
# (trainerSeed, D, H, parameterCount, substeps) rejects a non-positive value
# in config.json. Every one of these is set by a CLI flag here except D
# (computed from the graph), so this CLI validates each flag up front —
# before any training runs — rather than let a bad value train to
# completion and only fail once `readRunDir` reads the output.
def _require_positive_int(flag: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{flag} must be a positive integer, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flyarena-train", description="Seeded, GPU-batched CEM readout trainer (WP3)."
    )
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument(
        "--graph", required=True, help="Path to an export-arms.ts bundle JSON (training/runs/arms/<sha>/<arm>.json)."
    )
    parser.add_argument(
        "--replica-seed",
        type=int,
        required=True,
        dest="trainer_seed",
        help="trainer_seed identifying this replica (e.g. 101, 202, 303).",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output run directory (created if missing).")
    parser.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="K, neural substeps per tick. Required for a production (non-trace) graph bundle.",
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, dest="hidden_size")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="T, episode length in ticks.")
    parser.add_argument("--population", type=int, default=256, help="P.")
    parser.add_argument("--elites", type=int, default=32, help="N_e.")
    parser.add_argument("--generations", type=int, default=150, help="G.")
    parser.add_argument(
        "--train-seeds-per-generation", type=int, default=16, dest="e_train", help="E, training seeds per generation."
    )
    parser.add_argument("--alpha", type=float, default=0.7, help="CEM mean/std smoothing factor.")
    parser.add_argument("--std-floor", type=float, default=0.02, dest="std_floor")
    parser.add_argument("--init-std", type=float, default=0.5, dest="init_std")
    parser.add_argument(
        "--device", default=None, help="torch device string (e.g. cpu, cuda). Defaults to cuda if available, else cpu."
    )
    return parser.parse_args(argv)


@dataclass
class TrainingResult:
    out_dir: Path
    theta_final_path: Path
    theta_best_path: Path
    config_path: Path
    generations_csv_path: Path
    env_path: Path
    cem_result: CemResult
    wall_seconds: float


def _git_rev() -> str | None:
    """`git rev-parse HEAD` for `env.json`'s provenance. `None` on any
    failure (not a git checkout, `git` missing, etc.) — informational, never
    a gate."""
    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, timeout=10, check=True
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _resolve_substeps(graph_source: str, explicit_substeps: int | None) -> int:
    if explicit_substeps is not None:
        return explicit_substeps
    if graph_source == "trace-graph-fixture":
        return TRACE_SUBSTEPS
    raise SystemExit(
        "flyarena-train: --substeps is required for a production graph bundle "
        f"(graphSource={graph_source!r}); trace-graph development is the only case with a default "
        f"(TRACE_SUBSTEPS={TRACE_SUBSTEPS}). Pass the closed-loop K fixed by flyarena-bb45."
    )


def _load_and_validate_bundle(graph_path: Path, arm: str) -> dict:
    """Loads `--graph`'s bundle JSON and validates it looks like a real
    `export-arms.ts` bundle: file exists, is a JSON object, has every
    `REQUIRED_BUNDLE_FIELDS` entry (non-empty for the hash fields), a
    supported `formatVersion`, and an `arm` matching `--arm`. Pure function
    of a path and a string — independently testable without touching
    `torch`/`graph.py`."""
    if not graph_path.exists():
        raise FileNotFoundError(f"--graph bundle not found: {graph_path}")
    raw_bundle = json.loads(graph_path.read_text())
    if not isinstance(raw_bundle, dict):
        raise ValueError(f"--graph {graph_path} is not a JSON object")

    missing_bundle_fields = [field for field in REQUIRED_BUNDLE_FIELDS if field not in raw_bundle]
    if missing_bundle_fields:
        raise ValueError(
            f"--graph {graph_path} is missing bundle field(s) {missing_bundle_fields}; pass a bundle "
            "written by `npm run training:export-arms`, not a raw graph JSON."
        )
    # `field in raw_bundle` above accepts a JSON `null`/empty value for a
    # required field (the key exists, its value doesn't). `sha256`/
    # `graphArtifactSha256` in particular flow straight into `config.json`'s
    # `armBundleSha256` (evaluate.ts's bundle-integrity check) and `env.json`,
    # so an empty one there would silently disable that check rather than
    # fail loudly here.
    for hash_field in ("sha256", "graphArtifactSha256"):
        if not isinstance(raw_bundle[hash_field], str) or not raw_bundle[hash_field]:
            raise ValueError(f"--graph {graph_path}: bundle field {hash_field!r} must be a non-empty string")
    if raw_bundle["formatVersion"] != 1:
        raise ValueError(f"--graph {graph_path} has unsupported bundle formatVersion {raw_bundle['formatVersion']!r}")
    if raw_bundle["arm"] != arm:
        raise ValueError(f"--arm {arm!r} does not match the bundle's own arm {raw_bundle['arm']!r} ({graph_path})")

    graph_source = raw_bundle["graphSource"]
    if graph_source not in ("artifact", "trace-graph-fixture"):
        raise ValueError(f"--graph {graph_path} has unsupported graphSource {graph_source!r}")

    return raw_bundle


def _resolve_graph(
    raw_bundle: dict, graph_path: Path, device: torch.device
) -> tuple[ConnectomeGraph, int]:
    """Loads the graph a validated bundle points to and cross-checks the
    bundle's self-declared `D`/`outputNeuronIndices` against what
    `output_neuron_indices` actually computes from the loaded graph arrays —
    a free guard against a corrupted/mismatched bundle or a Python/TS
    divergence, since the two are otherwise never compared. Returns the
    loaded graph and `D`."""
    graph = load_graph_json(graph_path, device=device)
    d = int(output_neuron_indices(graph).numel())
    if d <= 0:
        raise ValueError(f"--graph {graph_path} has no output-assigned neurons (D=0); cannot train a readout")
    bundle_indices = list(raw_bundle["outputNeuronIndices"])
    computed_indices = output_neuron_indices(graph).tolist()
    if raw_bundle["D"] != d or bundle_indices != computed_indices:
        raise ValueError(
            f"--graph {graph_path}: bundle D={raw_bundle['D']!r}/outputNeuronIndices disagree with the "
            f"graph's own outputPopulationIndex (computed D={d}); bundle may be stale or corrupted"
        )
    return graph, d


def _build_run_config(
    args: argparse.Namespace,
    raw_bundle: dict,
    d: int,
    substeps: int,
    parameter_count: int,
    cem_config: CemConfig,
    result: CemResult,
) -> dict:
    """The `config.json` payload: `run-dir.ts`'s required `RunConfig` fields
    plus informational hyperparameters/seed-set/provenance fields (the
    plan's "config.json records every hyperparameter, the seed sets, D, H,
    the parameter count, and the arm name"). Pure function of already-
    computed values — trivially unit-testable without running CEM."""
    return {
        # RunConfig fields (scripts/training/run-dir.ts) — required.
        "arm": args.arm,
        "trainerSeed": args.trainer_seed,
        "D": d,
        "H": args.hidden_size,
        "parameterCount": parameter_count,
        "substeps": substeps,
        # `armBundleSha256` is required present now that REQUIRED_BUNDLE_FIELDS
        # gates every bundle load: evaluate.ts's loadArmGraphs cross-checks it
        # against the loaded bundle's own sha256 (its "trained against the
        # right bundle" integrity check), so this field must never be
        # silently dropped for being `None`.
        "armBundleSha256": raw_bundle["sha256"],
        # Every hyperparameter, the seed sets, and provenance (informational;
        # not read by run-dir.ts/evaluate.ts, but recorded per the plan:
        # "config.json records every hyperparameter, the seed sets, D, H,
        # the parameter count, and the arm name").
        "graphId": raw_bundle["graphId"],
        "graphSource": raw_bundle["graphSource"],
        "graphArtifactSha256": raw_bundle["graphArtifactSha256"],
        "ticks": args.ticks,
        "population": cem_config.population,
        "elites": cem_config.elites,
        "generations": cem_config.generations,
        "alpha": cem_config.alpha,
        "stdFloor": cem_config.std_floor,
        "initStd": cem_config.init_std,
        "trainingSeedsPerGeneration": cem_config.training_seeds_per_generation,
        "trainingSeedRange": [cem_config.training_seed_low, cem_config.training_seed_high],
        # Recorded explicitly (not just implied by the CLI's source code)
        # because this deliberately deviates from the plan's literal
        # `trainer_seed + generation` formula — see `seeds.py`'s and
        # `README.md`'s "Replica training-seed independence" sections.
        "trainingSeedRng": "default_rng([trainerSeed, generation])",
        "validationSeedRange": [
            cem_config.validation_seed_start,
            cem_config.validation_seed_start + cem_config.validation_seed_count - 1,
        ],
        "heldOutSeedRange": [HELD_OUT_SEED_START, HELD_OUT_SEED_START + HELD_OUT_SEED_COUNT - 1],
        "bestValidationFitness": result.best_validation_fitness,
    }


def _capture_env_info(device: torch.device, raw_bundle: dict, wall_seconds: float, history: list) -> dict:
    """The `env.json` payload: torch/CUDA versions, device name, git rev,
    graph bundle provenance, precision flags, and wall time. Pure function
    of already-computed values — independently testable without running a
    full (if tiny) training loop first."""
    return {
        "torchVersion": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaVersion": torch.version.cuda,
        "deviceName": torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or platform.machine()),
        "device": str(device),
        "gitRev": _git_rev(),
        "graphBundleSha256": raw_bundle["sha256"],
        "graphArtifactSha256": raw_bundle["graphArtifactSha256"],
        "precision": {
            "cudaMatmulAllowTf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnnAllowTf32": torch.backends.cudnn.allow_tf32,
            "float32MatmulPrecision": torch.get_float32_matmul_precision(),
            "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "wallSeconds": wall_seconds,
        "wallSecondsPerGeneration": wall_seconds / len(history),
    }


def run_training(args: argparse.Namespace) -> TrainingResult:
    """Core training logic, separated from CLI parsing/`main` (same
    `runExportArms`/`runEvaluate` convention `scripts/training/export-arms.ts`
    and `evaluate.ts` use), so tests can call it in-process. A thin
    orchestrator: validate flags -> load bundle -> resolve graph -> build
    env/CEM config -> run CEM -> write outputs (via the helpers above)."""
    # Validate every flag that becomes a `run-dir.ts` RunConfig field before
    # any I/O or training starts (see REQUIRED_BUNDLE_FIELDS's doc comment).
    _require_positive_int("--replica-seed", args.trainer_seed)
    _require_positive_int("--hidden-size", args.hidden_size)
    _require_positive_int("--ticks", args.ticks)
    if args.substeps is not None:
        _require_positive_int("--substeps", args.substeps)

    graph_path = Path(args.graph)
    raw_bundle = _load_and_validate_bundle(graph_path, args.arm)
    substeps = _resolve_substeps(raw_bundle["graphSource"], args.substeps)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph, d = _resolve_graph(raw_bundle, graph_path, device)
    parameter_count = readout_parameter_count(d, args.hidden_size)

    env = build_rollout_env(graph, device=device)

    def evaluate(theta_batch: torch.Tensor, seeds) -> torch.Tensor:
        return evaluate_fitness(env, theta_batch, seeds, args.hidden_size, args.ticks, substeps)

    cem_config = CemConfig(
        population=args.population,
        elites=args.elites,
        generations=args.generations,
        alpha=args.alpha,
        std_floor=args.std_floor,
        init_std=args.init_std,
        training_seeds_per_generation=args.e_train,
    )

    # Created before training starts: fails fast on a bad --out (permissions,
    # a read-only mount, --out pointing at an existing file) rather than
    # discovering it only after a run that can take ~20+ minutes at
    # production scale.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove any config.json already in --out *now*, before training starts,
    # not merely write a new one last. Writing config.json last (below) only
    # protects a *fresh* --out: it stops an interrupted write from producing
    # a directory that newly looks complete. It does nothing for a *reused*
    # --out that already has a complete previous run's config.json in it —
    # without this unlink, a crash after theta_final.npy/theta_best.npy are
    # overwritten but before this run's own config.json is written would
    # leave the new weights paired with the OLD run's config.json (a
    # different trainerSeed/substeps/parameterCount), and `readRunDir` would
    # silently accept that mismatched pair whenever D/H happen to still
    # match. Deleting it up front means every code path below either writes
    # this run's own complete config.json or leaves none at all.
    (out_dir / "config.json").unlink(missing_ok=True)

    start = time.perf_counter()
    result = run_cem(parameter_count, evaluate, cem_config, args.trainer_seed, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start

    if not torch.isfinite(result.theta_final).all() or not torch.isfinite(result.theta_best).all():
        raise RuntimeError(
            "flyarena-train: CEM produced a non-finite theta_final/theta_best; refusing to write a run "
            "directory (this should be unreachable — run_cem rejects non-finite fitness every generation)"
        )

    theta_final = result.theta_final.detach().to("cpu", dtype=torch.float32).numpy()
    theta_best = result.theta_best.detach().to("cpu", dtype=torch.float32).numpy()

    config = _build_run_config(args, raw_bundle, d, substeps, parameter_count, cem_config, result)

    # Written in this order — theta/csv/env, then config.json LAST — so that
    # a failure partway through (disk full, a write error) can never leave a
    # *complete-looking* run directory behind: `readRunDir` reads
    # config.json first, and this run's own config.json (any prior one was
    # already removed above) doesn't exist until every other file has been
    # written successfully, so an interrupted write is a clean "file not
    # found" for any downstream reader instead of a directory that silently
    # pairs this run's new theta_final.npy with a stale config.json.
    theta_final_path = out_dir / "theta_final.npy"
    theta_best_path = out_dir / "theta_best.npy"
    np.save(theta_final_path, theta_final)
    np.save(theta_best_path, theta_best)

    generations_csv_path = out_dir / "generations.csv"
    with generations_csv_path.open("w") as handle:
        handle.write("generation,meanFitness,maxFitness,validationFitness\n")
        for record in result.history:
            handle.write(f"{record.generation},{record.mean_fitness},{record.max_fitness},{record.validation_fitness}\n")

    env_info = _capture_env_info(device, raw_bundle, wall_seconds, result.history)
    env_path = out_dir / "env.json"
    # allow_nan=False: fail here, in Python, rather than writing a token
    # (`NaN`/`Infinity`/`-Infinity`) that `JSON.parse` in `run-dir.ts` rejects
    # much later, in a different language, with a much less specific error.
    env_path.write_text(json.dumps(env_info, indent=2, sort_keys=True, allow_nan=False))

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False))

    return TrainingResult(
        out_dir=out_dir,
        theta_final_path=theta_final_path,
        theta_best_path=theta_best_path,
        config_path=config_path,
        generations_csv_path=generations_csv_path,
        env_path=env_path,
        cem_result=result,
        wall_seconds=wall_seconds,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = run_training(args)
    # `SystemExit` (raised by `_resolve_substeps`) is a `BaseException`, not
    # an `Exception`, so it already propagates through the `except Exception`
    # below untouched — no explicit re-raise needed.
    except Exception as error:
        print(f"flyarena-train failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    per_generation = result.wall_seconds / len(result.cem_result.history)
    print(f"flyarena-train: wrote {result.out_dir} ({result.wall_seconds:.1f}s total, {per_generation:.2f}s/gen)")


if __name__ == "__main__":
    main()
