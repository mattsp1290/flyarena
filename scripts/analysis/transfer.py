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
DD_IAST_ENABLED=false` and `PYTHONPATH` unset; refuses to run otherwise
(`env_guard.assert_single_threaded_blas`, checked before `numpy` does any
work).
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
    build_dense_matrices,
    load_verified_graph,
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


def _compute_transfer(matrices: DenseGraphMatrices, leak_rate: float, global_gain: float, timestep_seconds: float) -> TransferComputation:
    A = matrices.adjacency
    B = matrices.input_matrix
    O = matrices.output_matrix
    neuron_count = A.shape[0]

    system_matrix = leak_rate * np.eye(neuron_count, dtype=np.float64) - global_gain * A

    if neuron_count == 0:
        condition_number = 0.0
        steady_state_map = np.zeros((0, B.shape[1]), dtype=np.float64)
        eig_A = np.zeros((0,), dtype=np.complex128)
    else:
        condition_number = float(np.linalg.cond(system_matrix))
        steady_state_map = np.linalg.solve(system_matrix, B)
        eig_A = np.linalg.eigvals(A)

    T = O @ steady_state_map  # (outputPopulationCount, inputChannelCount)

    ill_conditioned = condition_number > ILL_CONDITIONED_THRESHOLD

    spectral_abscissa = float(global_gain * np.max(eig_A.real)) if eig_A.size > 0 else 0.0
    stable = spectral_abscissa < leak_rate
    time_constant_seconds = (1.0 / (leak_rate - spectral_abscissa)) if stable else None

    discretized_eig = 1.0 - timestep_seconds * leak_rate + timestep_seconds * global_gain * eig_A
    discretized_spectral_radius = (
        float(np.max(np.abs(discretized_eig)))
        if discretized_eig.size > 0
        else abs(1.0 - timestep_seconds * leak_rate)
    )

    # The derived predictors index specific rows/columns of `T` by the
    # production `OUTPUT_POPULATION`/`OBSERVATION_CHANNELS` convention (3
    # populations, 8 channels) -- every real graph (biological/disconnected/
    # rewired) has exactly that shape, but a hand-built test fixture
    # (`tests_python/test_transfer.py`'s 3-neuron graph) may not. Report
    # `None` rather than raising `IndexError` for a smaller `T`, so this
    # function stays usable on any well-formed graph, not only
    # production-shaped ones.
    fits_production_shape = T.shape[0] > max(OUTPUT_POPULATION_INDEX.values()) and T.shape[1] > max(
        OBSERVATION_CHANNEL_INDEX.values()
    )
    if fits_production_shape:
        turn_gain = float(
            T[OUTPUT_POPULATION_INDEX["yaw"], OBSERVATION_CHANNEL_INDEX["foodBearing"]]
            - T[OUTPUT_POPULATION_INDEX["yaw"], OBSERVATION_CHANNEL_INDEX["hazardBearing"]]
        )
        approach_gain = float(T[OUTPUT_POPULATION_INDEX["thrust"], OBSERVATION_CHANNEL_INDEX["foodDistance"]])
    else:
        turn_gain = None
        approach_gain = None

    result = {
        "T": T.tolist(),
        "outputPopulationOrder": ["thrust", "yaw", "brake"],
        "inputChannelOrder": list(OBSERVATION_CHANNEL_INDEX.keys()),
        "leakRate": float(leak_rate),
        "globalGain": float(global_gain),
        "spectralAbscissa": spectral_abscissa,
        "stable": bool(stable),
        "timeConstantSeconds": time_constant_seconds,
        "conditionNumber": condition_number,
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
    avoid a second `O(n^3)` solve for the same graph."""
    meta = graph.metadata
    matrices = build_dense_matrices(graph)
    computation = _compute_transfer(
        matrices,
        leak_rate=float(meta["leakRate"]),
        global_gain=float(meta["globalGain"]),
        timestep_seconds=float(meta["timestepSeconds"]),
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
    )
    return graph_id, computation.result, computation.steady_state_map


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
    index = _read_rewire_index(args.index)
    steady_state_dir = args.steady_state_dir or (args.out.parent / "steady-state")

    jobs: list[tuple[str, Path, str, bool]] = []  # (graphId, path, expectedSha256, isDisconnected)
    if args.biological is not None:
        biological_path = args.biological
        if not biological_path.exists():
            raise SystemExit(f"transfer: --biological path {biological_path} does not exist")
        jobs.append(("biological", biological_path, index["sourceSha256"], False))
        jobs.append(("disconnected", biological_path, index["sourceSha256"], True))

    seeds = sorted(index["seeds"], key=lambda entry: entry["seed"])
    for entry in seeds:
        graph_path = args.graphs_dir / entry["artifact"]
        jobs.append((f"rewired-{entry['seed']}", graph_path, entry["binarySha256"], False))

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_one_disconnected if is_disconnected else _one_graph, graph_id, path, sha): graph_id
            for graph_id, path, sha, is_disconnected in jobs
        }
        for future in futures:
            graph_id, result, steady_state_map = future.result()
            results[graph_id] = result
            write_steady_state_sidecar(steady_state_dir / f"{graph_id}.steadystate.f64", steady_state_map)

    out_payload = {
        "version": 1,
        "sourceGraphSha256": index["sourceSha256"],
        "rewireSourceSha256": index["rewireSourceSha256"],
        "graphs": results,
    }
    write_canonical_json(args.out, out_payload)
    # eslint-equivalent user-facing summary line for a CLI tool.
    print(f"transfer: wrote {args.out} ({len(results)} graphs) and steady-state sidecars under {steady_state_dir}")


if __name__ == "__main__":
    main()
