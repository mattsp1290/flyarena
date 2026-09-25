#!/usr/bin/env python3
"""Steady-state linear input->output transfer analysis
(`.agents/plans/null-explanation/02-transfer-and-features.md`, WP2).

For each graph, compute the 3x8 steady-state linear transfer matrix `T = O
(lambda I - g A)^-1 B`, exact within the model's rate/input clamps (see
`docs/graph-format.md`'s "Dynamics" section and `src/lib/connectome/model.ts`'s
`stepModel`, the ground truth this module's algebra reproduces):

- `A[post, pre] = presynapticSigns[pre] * contactMagnitudes(pre -> post)` for
  every CSR edge (`scripts/analysis/graph_io.py`'s `build_dense_matrices`);
- `B[i, c] = inputWeight[i]` where `inputChannelIndex[i] == c`;
- `O[p, i] = outputWeight[i]` where `outputPopulationIndex[i] == p`.

At a fixed point with constant input `u` and no active clamp, `0 = -lambda r
+ g A r + B u`, so `r* = (lambda I - g A)^-1 B u` and `T = O (lambda I - g
A)^-1 B`. Also reports (per the plan's "Linear transfer" section):

- the spectral abscissa `max Re(eig(g A))`, compared with `lambda` (the
  continuous-time system `dr/dt = -lambda r + g A r` is stable iff the
  abscissa is `< lambda`; for non-normal `A` the spectral abscissa, not the
  spectral radius, is the correct stability criterion);
- `numpy.linalg.cond(lambda I - g A)`; graphs above `1e8` are flagged
  `illConditioned`;
- the steady-state time constant `1 / (lambda - spectralAbscissa)` from the
  dominant eigenvalue of `-lambda I + g A` (`None` when unstable);
- the per-substep discretization's spectral radius,
  `max |eig(I + dt(-lambda I + g A))| = max |1 - dt*lambda + dt*g*eig(A)|`
  (an affine function of `eig(A)`, since `-lambda I` is a scalar multiple of
  the identity and therefore shares every eigenvector of `A`), which must be
  `< 1` for the per-substep Euler integration itself to be stable, reported
  as `discretizedStable` alongside (never instead of) `stable`. The two are
  conceptually different claims -- `stable` is the continuous-time fixed-
  point criterion `T` itself represents; `discretizedStable` is whether the
  real, simulated (discrete Euler) trajectory actually converges toward it
  -- and are correctly kept distinct here rather than conflated (a
  thermo-architecture review note, hand-off for WP3's
  `03-explanation-report.md`, not yet implemented): when WP3 writes the
  report's stability prose, it must present both numbers side by side (or
  fold both into one sentence, e.g. "stable in continuous time (abscissa
  0.1449 < lambda=0.35) and in the discretized update (spectral radius
  0.99xx < 1)"), not report `stable` alone under a bare "stable" label that
  a reader could reasonably (but wrongly) take to already certify the
  discrete trajectory;
- the derived predictors `turnGain = T[yaw, foodBearing] - T[yaw,
  hazardBearing]` and `approachGain = T[thrust, foodDistance]`
  (`src/lib/arena/actions.ts`'s `OUTPUT_POPULATION` and
  `src/lib/arena/sensors.ts`'s `OBSERVATION_CHANNELS` give the exact
  row/column indices below).

Also writes, per graph, the steady-state input map `M = (lambda I - g A)^-1
B` (shape `neuronCount x inputChannelCount`, float64, row-major) as a raw
binary sidecar under `--steady-state-dir`: `scripts/null/regime-task.ts`
reuses it (`r*(u_t) = M @ clamp(u_t)`) to compute the linear-regime distance
metric per tick without repeating this module's `O(n^3)` dense solve in
TypeScript.

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DD_IAST_ENABLED=false` and `PYTHONPATH` unset (`.agents/plans/
null-explanation/05-execution-handoff.md`'s required run environment). This
module refuses to run unless the three thread-count variables above are all
`"1"` (`env_guard.assert_single_threaded_blas`, checked before `numpy` does
any work) -- `DD_IAST_ENABLED`/`PYTHONPATH` are not independently enforced
here (a dual-review finding: an earlier version of this docstring implied
they were).
"""

from __future__ import annotations

import argparse
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402

import graph_io  # noqa: E402
from graph_io import (  # noqa: E402
    DenseGraphMatrices,
    build_dense_matrices,
    load_verified_graph,
    sha256_hex,
    write_canonical_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import binfmt  # noqa: E402

#: `src/lib/arena/sensors.ts`'s `OBSERVATION_CHANNELS`, index-for-index.
OBSERVATION_CHANNEL_INDEX: Mapping[str, int] = {
    "foodBearing": 0,
    "foodDistance": 1,
    "hazardBearing": 2,
    "hazardDistance": 3,
    "forwardClearance": 4,
    "leftClearance": 5,
    "rightClearance": 6,
    "speed": 7,
}

#: `src/lib/arena/actions.ts`'s `OUTPUT_POPULATION`.
OUTPUT_POPULATION_INDEX: Mapping[str, int] = {"thrust": 0, "yaw": 1, "brake": 2}

#: Graphs whose `(lambda I - g A)` condition number exceeds this are flagged
#: `illConditioned` (the plan's predeclared threshold).
ILL_CONDITIONED_THRESHOLD = 1e8

#: Files that determine `transfer.json`'s own output bytes: this module
#: itself, plus the two shared helpers it depends on for graph loading/
#: dense-matrix construction (`graph_io.py`) and the single-threaded-BLAS
#: guard (`env_guard.py`). Hashed into `producer.sourceSha256` below (via
#: `graph_io.source_identity_sha256`, the same filename+NUL+bytes-per-file
#: scheme `scripts/data/compile.py`'s `compiler_source_sha256()` already
#: uses) so `explain.py`'s `verify_provenance` can mechanically refuse a
#: `transfer.json` regenerated from a different version of this code -- a
#: thermo-methodology review finding: this pipeline previously pinned only
#: *graph* identity (`sourceGraphSha256`/`rewireSourceSha256`), never *code*
#: identity, so a `transfer.json` regenerated from stale code against the
#: same 502 graphs would have passed every existing check silently.
TRANSFER_SOURCE_FILENAMES: tuple[str, ...] = ("transfer.py", "graph_io.py", "env_guard.py")
TRANSFER_SOURCE_DIR = Path(__file__).resolve().parent


def transfer_producer() -> dict:
    """This run's code-identity block: which script produced this output,
    the sha256 of its own source (`TRANSFER_SOURCE_FILENAMES`), and the host
    it ran on. `host` uses `platform.machine()`/`platform.python_version()`
    (not `scripts/null/regime-check.ts`'s `process.arch`/`process.version`
    convention) because this is a Python producer -- `platform.machine()`
    and Node's `process.arch` report the *same* physical ARM64 host
    differently (`"aarch64"` vs `"arm64"`), so `explain.py` cross-checks this
    block's `arch` only against `features.py`'s own (another Python
    producer), never against a TypeScript producer's `host.arch` string."""
    return {
        "script": "scripts/analysis/transfer.py",
        "sourceSha256": graph_io.source_identity_sha256(TRANSFER_SOURCE_DIR, TRANSFER_SOURCE_FILENAMES),
        "host": {"arch": platform.machine(), "python": platform.python_version()},
    }


@dataclass(frozen=True)
class TransferComputation:
    result: dict
    steady_state_map: np.ndarray  # (neuronCount, inputChannelCount), float64


#: `T`'s shape every real graph (biological/disconnected/rewired) has --
#: `_compute_transfer`'s `strict_shape=True` callers (the CLI) raise rather
#: than silently reporting `None` derived predictors when a graph doesn't
#: match it (a dual-review finding: a silent `None` on a future metadata
#: change, e.g. an added observation channel, would make `explain.py` test a
#: predictor that is null for every graph with no error anywhere).
PRODUCTION_T_SHAPE = (len(OUTPUT_POPULATION_INDEX), len(OBSERVATION_CHANNEL_INDEX))


def _finite_or_none(value: float) -> float | None:
    """`np.linalg.cond` returns `inf` (never `nan`, and never raises) for a
    singular or numerically singular matrix -- independent of whether
    `np.linalg.solve` on the *same* matrix succeeds or raises `LinAlgError`
    (they use different LAPACK routines: `cond`'s SVD-based ratio can hit a
    floating-point `inf` for a matrix `solve`'s LU decomposition still finds
    *a* numerical solution for). `graph_io.canonical_json_text` writes with
    `allow_nan=False`, so a non-finite `conditionNumber` reaching the result
    dict crashes the whole batch at JSON-write time, well after every graph
    has already been computed -- a round-2 dual-review finding, whose fix
    only normalized the `LinAlgError` branch; a round-3 finding caught that
    `cond` can return `inf` even when `solve` *succeeds*, which the round-2
    fix missed. Extracted as its own function (rather than inlined at the
    one call site) specifically so it has a direct unit test independent of
    constructing a real matrix with this exact, LAPACK-implementation-
    dependent "solve succeeds but cond is inf" property."""
    return value if np.isfinite(value) else None


def _compute_transfer(
    matrices: DenseGraphMatrices,
    leak_rate: float,
    global_gain: float,
    timestep_seconds: float,
    strict_shape: bool = False,
) -> TransferComputation:
    A = matrices.adjacency
    B = matrices.input_matrix
    O = matrices.output_matrix
    neuron_count = A.shape[0]

    system_matrix = leak_rate * np.eye(neuron_count, dtype=np.float64) - global_gain * A

    # `condition_number is None` is ambiguous on its own -- it means "not
    # applicable" for an empty graph (`neuron_count == 0`) but "non-finite,
    # i.e. as ill-conditioned as it gets" for a non-empty one (see
    # `_finite_or_none`'s doc comment). `condition_number_non_finite` keeps
    # those two cases distinct for the `ill_conditioned` computation below.
    condition_number_non_finite = False
    if neuron_count == 0:
        condition_number = None
        singular = False
        steady_state_map = np.zeros((0, B.shape[1]), dtype=np.float64)
        eig_A = np.zeros((0,), dtype=np.complex128)
    else:
        # See `_finite_or_none`'s doc comment for why this must be
        # normalized here, unconditionally, before either branch below runs
        # -- not only in the `LinAlgError` branch (that was round-2's fix;
        # round-3 found it was incomplete).
        raw_condition_number = float(np.linalg.cond(system_matrix))
        condition_number = _finite_or_none(raw_condition_number)
        condition_number_non_finite = condition_number is None
        try:
            steady_state_map = np.linalg.solve(system_matrix, B)
            singular = False
        except np.linalg.LinAlgError:
            # The plan's policy for a bad graph is to flag and exclude it,
            # not to abort the whole batch (`illConditioned` already does
            # this for a merely ill-conditioned matrix); an *exactly*
            # singular one is the limiting case of the same policy, and
            # extremely unlikely at this study's `leakRate` (the real
            # biological graph's condition number is ~4.8) -- but a `main()`
            # that crashes on it would otherwise lose every other graph's
            # multi-hour work in the same batch (a dual-review finding).
            steady_state_map = np.full((neuron_count, B.shape[1]), np.nan, dtype=np.float64)
            singular = True
        eig_A = np.linalg.eigvals(A)

    T = O @ steady_state_map  # (outputPopulationCount, inputChannelCount)

    ill_conditioned = (
        singular
        or condition_number_non_finite
        or (condition_number is not None and condition_number > ILL_CONDITIONED_THRESHOLD)
    )

    spectral_abscissa = float(global_gain * np.max(eig_A.real)) if eig_A.size > 0 else 0.0
    stable = (not singular) and spectral_abscissa < leak_rate
    time_constant_seconds = (1.0 / (leak_rate - spectral_abscissa)) if stable else None

    discretized_eig = 1.0 - timestep_seconds * leak_rate + timestep_seconds * global_gain * eig_A
    discretized_spectral_radius = (
        float(np.max(np.abs(discretized_eig)))
        if discretized_eig.size > 0
        else abs(1.0 - timestep_seconds * leak_rate)
    )

    # The derived predictors index specific rows/columns of `T` by the
    # production `OUTPUT_POPULATION`/`OBSERVATION_CHANNELS` convention
    # (`PRODUCTION_T_SHAPE`) -- every real graph has exactly that shape, but
    # a hand-built test fixture (`tests_python/test_transfer.py`'s 3-neuron
    # graph) may not. `strict_shape=True` (the CLI path) raises on a
    # mismatch instead of silently reporting `None`; `strict_shape=False`
    # (the public `transfer_matrix()` used by tests/tooling on
    # non-production-shaped graphs) reports `None`. `singular` is checked
    # first and unconditionally reports `None` regardless of `strict_shape`
    # or `T`'s shape: `T` itself is already `None` for a singular graph (see
    # `result["T"]` below), so there is nothing to index into -- and it must
    # never raise here, since `_verify_jobs`'s pre-flight check cannot catch
    # an exactly singular graph (a round-2 dual-review finding: a flattened
    # `if singular / elif shape-matches / elif strict_shape / else` reads
    # this precedence directly, rather than requiring `and not singular` on
    # two of three branches to work it out).
    if singular:
        turn_gain = None
        approach_gain = None
        output_population_order: list[str] | None = None
        input_channel_order: list[str] | None = None
    elif T.shape == PRODUCTION_T_SHAPE:
        turn_gain = float(
            T[OUTPUT_POPULATION_INDEX["yaw"], OBSERVATION_CHANNEL_INDEX["foodBearing"]]
            - T[OUTPUT_POPULATION_INDEX["yaw"], OBSERVATION_CHANNEL_INDEX["hazardBearing"]]
        )
        approach_gain = float(T[OUTPUT_POPULATION_INDEX["thrust"], OBSERVATION_CHANNEL_INDEX["foodDistance"]])
        output_population_order = list(OUTPUT_POPULATION_INDEX.keys())
        input_channel_order = list(OBSERVATION_CHANNEL_INDEX.keys())
    elif strict_shape:
        raise ValueError(
            f"transfer: T has shape {T.shape}, expected {PRODUCTION_T_SHAPE} "
            "(OUTPUT_POPULATION/OBSERVATION_CHANNELS changed? update transfer.py's constants)"
        )
    else:
        turn_gain = None
        approach_gain = None
        output_population_order = None
        input_channel_order = None

    result = {
        "T": None if singular else T.tolist(),
        "outputPopulationOrder": output_population_order,
        "inputChannelOrder": input_channel_order,
        "leakRate": float(leak_rate),
        "globalGain": float(global_gain),
        "spectralAbscissa": spectral_abscissa,
        "stable": bool(stable),
        "timeConstantSeconds": time_constant_seconds,
        "conditionNumber": condition_number,
        "singular": bool(singular),
        "illConditioned": bool(ill_conditioned),
        "discretizedSpectralRadius": discretized_spectral_radius,
        "discretizedStable": bool(discretized_spectral_radius < 1.0),
        "turnGain": turn_gain,
        "approachGain": approach_gain,
    }
    return TransferComputation(result=result, steady_state_map=steady_state_map)


def transfer_matrix(graph: "binfmt.GraphArrays") -> dict:
    """`transfer_matrix(graph) -> dict` per the plan's change-surface table.
    Recomputes the dense solve; CLI callers that also need the steady-state
    map use `_compute_transfer`/`build_dense_matrices` directly instead, to
    avoid a second `O(n^3)` solve for the same graph. `strict_shape=False`:
    a library/test entry point, usable on any well-formed graph, not only
    production-shaped ones (see `_compute_transfer`'s doc comment)."""
    meta = graph.metadata
    matrices = build_dense_matrices(graph)
    computation = _compute_transfer(
        matrices,
        leak_rate=float(meta["leakRate"]),
        global_gain=float(meta["globalGain"]),
        timestep_seconds=float(meta["timestepSeconds"]),
        strict_shape=False,
    )
    return computation.result


def disconnected_matrices(matrices: DenseGraphMatrices) -> DenseGraphMatrices:
    """The disconnected negative control's dense matrices: same `B`/`O`, `A`
    zeroed out entirely -- matches `format.ts`'s `createDisconnectedGraph`
    ("every recurrent edge removed"; `B`/`O`'s per-neuron input/output
    mapping is unchanged)."""
    return DenseGraphMatrices(
        adjacency=np.zeros_like(matrices.adjacency),
        input_matrix=matrices.input_matrix,
        output_matrix=matrices.output_matrix,
    )


def write_steady_state_sidecar(path: Path, steady_state_map: np.ndarray) -> None:
    """Raw float64, row-major (`C` order): `regime-task.ts` reads this as a
    flat `Float64Array` of length `neuronCount * inputChannelCount` and
    indexes `M[i * inputChannelCount + c]`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.ascontiguousarray(steady_state_map, dtype=np.float64).tofile(tmp_path)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _one_graph(graph_id: str, path: Path, expected_sha256: str) -> tuple[str, dict, np.ndarray]:
    graph = load_verified_graph(path, expected_sha256)
    meta = graph.metadata
    matrices = build_dense_matrices(graph)
    computation = _compute_transfer(
        matrices,
        leak_rate=float(meta["leakRate"]),
        global_gain=float(meta["globalGain"]),
        timestep_seconds=float(meta["timestepSeconds"]),
        strict_shape=True,
    )
    return graph_id, computation.result, computation.steady_state_map


def _one_disconnected(graph_id: str, path: Path, expected_sha256: str) -> tuple[str, dict, np.ndarray]:
    graph = load_verified_graph(path, expected_sha256)
    meta = graph.metadata
    matrices = disconnected_matrices(build_dense_matrices(graph))
    computation = _compute_transfer(
        matrices,
        leak_rate=float(meta["leakRate"]),
        global_gain=float(meta["globalGain"]),
        timestep_seconds=float(meta["timestepSeconds"]),
        strict_shape=True,
    )
    return graph_id, computation.result, computation.steady_state_map


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = graph_io.base_arg_parser(__doc__, out_help="combined transfer.json output path")
    parser.add_argument(
        "--steady-state-dir",
        type=Path,
        default=None,
        help="directory for per-graph steady-state binary sidecars (default: <out's parent>/steady-state)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.biological is not None and args.skip_biological:
        raise SystemExit("transfer: --biological and --skip-biological are mutually exclusive")
    index = graph_io.read_rewire_index(args.index, "transfer")
    steady_state_dir = args.steady_state_dir or (args.out.parent / "steady-state")

    jobs = graph_io.build_jobs(args, index, "transfer")
    graph_io.verify_jobs(jobs, "transfer")

    # Delete any manifest left over from a previous run into this same
    # `--steady-state-dir` *before* writing anything new: without this, a
    # run that crashes partway (after overwriting some sidecars but before
    # reaching the manifest write below) leaves the *previous* run's
    # manifest.json sitting next to a directory it no longer accurately
    # describes -- `regime-check.ts`'s `readSteadyStateManifest` doc comment
    # says "its mere presence already rules out a run that crashed partway
    # through", which is only true if this happens (a round-2 dual-review
    # finding).
    manifest_path = steady_state_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)

    results: dict[str, dict] = {}
    manifest_graphs: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_one_disconnected if is_disconnected else _one_graph, graph_id, path, sha): graph_id
            for graph_id, path, sha, is_disconnected in jobs
        }
        try:
            for future in futures:
                graph_id, result, steady_state_map = future.result()
                results[graph_id] = result
                if result.get("singular"):
                    # No sidecar, no manifest entry: `regime-check.ts`'s
                    # manifest-based verification then refuses to run this
                    # graph (fails loud) instead of consuming a NaN-filled
                    # steady-state map silently.
                    continue
                sidecar_path = steady_state_dir / f"{graph_id}.steadystate.f64"
                write_steady_state_sidecar(sidecar_path, steady_state_map)
                job_sha = next(sha for gid, _p, sha, _d in jobs if gid == graph_id)
                manifest_graphs[graph_id] = {
                    "graphBinarySha256": job_sha,
                    "sidecarSha256": sha256_hex(sidecar_path.read_bytes()),
                    "neuronCount": int(steady_state_map.shape[0]),
                    "inputChannelCount": int(steady_state_map.shape[1]),
                }
        except BaseException:
            # A failed job otherwise leaves every *queued* job (everything
            # not yet dispatched to a worker) running to completion before
            # the error is even raised, wasting the rest of a multi-hour
            # batch (a dual-review finding: `ProcessPoolExecutor.__exit__`
            # calls `shutdown(wait=True)` by default). `graph_io.verify_jobs`
            # above already rules out the common cause (a bad graph file); this
            # guards against everything else (a `LinAlgError` `_compute_transfer`
            # doesn't itself catch, an `OSError` writing a sidecar, `Ctrl-C`).
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    out_payload = {
        "version": 1,
        "sourceGraphSha256": index["sourceSha256"],
        "rewireSourceSha256": index["rewireSourceSha256"],
        "graphs": results,
        "producer": transfer_producer(),
    }
    write_canonical_json(args.out, out_payload)

    # Written only after every job has succeeded (see the `try`/`except`
    # above): ties each steady-state sidecar to the exact graph bytes it was
    # solved from, so `regime-check.ts` can detect a stale or partial
    # sidecar directory left over from an earlier run instead of silently
    # computing `steadyStateDistance` against the wrong graph (a dual-review
    # finding -- the one input in this pipeline that previously had no
    # sha256 verification tying it to its source).
    manifest_payload = {
        "version": 1,
        "sourceGraphSha256": index["sourceSha256"],
        "rewireSourceSha256": index["rewireSourceSha256"],
        "graphs": manifest_graphs,
    }
    write_canonical_json(manifest_path, manifest_payload)

    # eslint-equivalent user-facing summary line for a CLI tool.
    print(f"transfer: wrote {args.out} ({len(results)} graphs) and steady-state sidecars under {steady_state_dir}")


if __name__ == "__main__":
    main()
