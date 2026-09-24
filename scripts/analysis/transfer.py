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
  `< 1` for the per-substep Euler integration itself to be stable;
- the derived predictors `turnGain = T[yaw, foodBearing] - T[yaw,
  hazardBearing]` and `approachGain = T[thrust, foodDistance]`
  (`src/lib/arena/actions.ts`'s `OUTPUT_POPULATION` and
  `src/lib/arena/sensors.ts`'s `OBSERVATION_CHANNELS` give the exact
  row/column indices below).

Also writes, per graph, the steady-state input map `M = (lambda I - g A)^-1
B` (shape `neuronCount x inputChannelCount`, float64, row-major) as a raw
binary sidecar under `--steady-state-dir`: `scripts/null/regime-worker.ts`
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
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402

from graph_io import (  # noqa: E402
    DenseGraphMatrices,
    GraphVerificationError,
    PUBLIC_DATA_DIR,
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

    if neuron_count == 0:
        condition_number = None
        singular = False
        steady_state_map = np.zeros((0, B.shape[1]), dtype=np.float64)
        eig_A = np.zeros((0,), dtype=np.complex128)
    else:
        condition_number = float(np.linalg.cond(system_matrix))
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

    ill_conditioned = singular or (condition_number is not None and condition_number > ILL_CONDITIONED_THRESHOLD)

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
    # non-production-shaped graphs) reports `None`.
    if T.shape == PRODUCTION_T_SHAPE and not singular:
        turn_gain = float(
            T[OUTPUT_POPULATION_INDEX["yaw"], OBSERVATION_CHANNEL_INDEX["foodBearing"]]
            - T[OUTPUT_POPULATION_INDEX["yaw"], OBSERVATION_CHANNEL_INDEX["hazardBearing"]]
        )
        approach_gain = float(T[OUTPUT_POPULATION_INDEX["thrust"], OBSERVATION_CHANNEL_INDEX["foodDistance"]])
        output_population_order: list[str] | None = list(OUTPUT_POPULATION_INDEX.keys())
        input_channel_order: list[str] | None = list(OBSERVATION_CHANNEL_INDEX.keys())
    elif strict_shape and not singular:
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
    """Raw float64, row-major (`C` order): `regime-worker.ts` reads this as a
    flat `Float64Array` of length `neuronCount * inputChannelCount` and
    indexes `M[i * inputChannelCount + c]`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.ascontiguousarray(steady_state_map, dtype=np.float64).tofile(tmp_path)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_rewire_index(path: Path) -> dict:
    with path.open("r") as fh:
        index = json.load(fh)
    for required in ("sourceArtifact", "sourceSha256", "rewireSourceSha256", "seeds"):
        if required not in index:
            raise ValueError(f"transfer: {path} is missing '{required}'")
    return index


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


def _verify_jobs(jobs: list[tuple[str, Path, str, bool]]) -> None:
    """Pre-flight decompressed-sha256 verification for every job, before any
    pool worker is submitted -- mirrors `null-evaluate.ts`'s
    `verifyRewiredFiles`'s "fails in seconds rather than after however much
    of a multi-hour run has already completed" up-front check (a dual-review
    finding: neither CLI previously verified anything before starting work,
    unlike the TS side). Aggregates every mismatch into one error, the same
    "report everything wrong at once" convention `verifyRewiredFiles` uses.
    """
    mismatches: list[str] = []
    checked: set[Path] = set()
    for graph_id, path, expected_sha256, _is_disconnected in jobs:
        if path in checked:
            continue  # biological and disconnected share the same source file
        checked.add(path)
        try:
            load_verified_graph(path, expected_sha256)
        except (OSError, GraphVerificationError, binfmt.InvalidGraphError) as error:
            mismatches.append(f"{graph_id}: {error}")
    if mismatches:
        raise GraphVerificationError(
            f"transfer: {len(mismatches)} graph file(s) failed pre-flight verification:\n" + "\n".join(mismatches)
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="rewire_batch.py index.json")
    parser.add_argument("--graphs-dir", type=Path, required=True, help="directory holding rewired .bin.gz files")
    parser.add_argument(
        "--biological",
        type=Path,
        default=None,
        help="biological source .bin.gz (default: public/data/<index.sourceArtifact>); "
        "also computes the disconnected control",
    )
    parser.add_argument(
        "--skip-biological",
        action="store_true",
        help="omit biological/disconnected entirely (rewired graphs only)",
    )
    parser.add_argument("--out", type=Path, required=True, help="combined transfer.json output path")
    parser.add_argument(
        "--steady-state-dir",
        type=Path,
        default=None,
        help="directory for per-graph steady-state binary sidecars (default: <out's parent>/steady-state)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="process-pool workers (each pinned to single-threaded BLAS); default min(8, cpu_count)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.biological is not None and args.skip_biological:
        raise SystemExit("transfer: --biological and --skip-biological are mutually exclusive")
    index = _read_rewire_index(args.index)
    steady_state_dir = args.steady_state_dir or (args.out.parent / "steady-state")

    jobs: list[tuple[str, Path, str, bool]] = []  # (graphId, path, expectedSha256, isDisconnected)
    if not args.skip_biological:
        # Default matches `regime-check.ts --biological`'s own resolution
        # (`resolve(PUBLIC_DATA_DIR, index.sourceArtifact)`) -- previously
        # this help text claimed the same default but the code silently
        # skipped biological/disconnected instead (a dual-review finding).
        biological_path = args.biological if args.biological is not None else PUBLIC_DATA_DIR / index["sourceArtifact"]
        if not biological_path.exists():
            raise SystemExit(f"transfer: biological source {biological_path} does not exist")
        jobs.append(("biological", biological_path, index["sourceSha256"], False))
        jobs.append(("disconnected", biological_path, index["sourceSha256"], True))

    seeds = sorted(index["seeds"], key=lambda entry: entry["seed"])
    for entry in seeds:
        graph_path = args.graphs_dir / entry["artifact"]
        jobs.append((f"rewired-{entry['seed']}", graph_path, entry["binarySha256"], False))

    _verify_jobs(jobs)

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
            # calls `shutdown(wait=True)` by default). `_verify_jobs` above
            # already rules out the common cause (a bad graph file); this
            # guards against everything else (a `LinAlgError` `_compute_transfer`
            # doesn't itself catch, an `OSError` writing a sidecar, `Ctrl-C`).
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    out_payload = {
        "version": 1,
        "sourceGraphSha256": index["sourceSha256"],
        "rewireSourceSha256": index["rewireSourceSha256"],
        "graphs": results,
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
    write_canonical_json(steady_state_dir / "manifest.json", manifest_payload)

    # eslint-equivalent user-facing summary line for a CLI tool.
    print(f"transfer: wrote {args.out} ({len(results)} graphs) and steady-state sidecars under {steady_state_dir}")


if __name__ == "__main__":
    main()
