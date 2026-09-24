"""`.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2 stop/go
gate 2: "on the trace graph fixture, `T` matches a float64 Python
fixed-point iteration within 1e-9 ... [and] a TS finite-difference estimate
... within 1e-3 relative", plus a 3-neuron hand graph whose `T` has a known
closed form.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# tests_python/conftest.py puts scripts/analysis on sys.path (and sets the
# required single-threaded BLAS env vars before transfer.py/graph_io.py ever
# import numpy); scripts/data is not on it by default, since most of
# tests_python/ never needs binfmt directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import binfmt  # noqa: E402

from env_guard import REQUIRED_SINGLE_THREADED_ENV_VARS, assert_single_threaded_blas  # noqa: E402
from graph_io import build_dense_matrices  # noqa: E402
from transfer import ILL_CONDITIONED_THRESHOLD, transfer_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests_python" / "fixtures" / "trace-graph-transfer.json"


# ---------------------------------------------------------------------------
# env_guard
# ---------------------------------------------------------------------------


def test_assert_single_threaded_blas_passes_when_all_set_to_one():
    assert_single_threaded_blas({name: "1" for name in REQUIRED_SINGLE_THREADED_ENV_VARS})


def test_assert_single_threaded_blas_rejects_missing_or_wrong_value():
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS"):
        assert_single_threaded_blas({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with pytest.raises(RuntimeError):
        assert_single_threaded_blas({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})


# ---------------------------------------------------------------------------
# 3-neuron hand graph: closed-form T
# ---------------------------------------------------------------------------


def _make_three_neuron_graph(leak_rate: float, global_gain: float, a: float, b: float, c: float, w_in: float, w_out: float) -> "binfmt.GraphArrays":
    """Neuron 0: input (channel 0), one outgoing edge 0->1 (sign +1, magnitude
    `a`). Neuron 1: two outgoing edges, 1->0 (magnitude `b`) and 1->2
    (magnitude `c`), both sign -1 (Dale's law: a neuron's sign applies to
    every outgoing edge). Neuron 2: output (population 0), no outgoing
    edges. CSR rows: row 0 = [1] (edge to neuron 1); row 1 = [0, 2] (edges to
    neurons 0 and 2, ascending); row 2 = [] (empty)."""
    metadata = {
        "formatVersion": 1,
        "neuronCount": 3,
        "edgeCount": 3,
        "inputChannelCount": 1,
        "outputPopulationCount": 1,
        "timestepSeconds": 1.0 / 30.0,
        "leakRate": leak_rate,
        "rateMin": -2.0,
        "rateMax": 2.0,
        "inputClampMin": -1.0,
        "inputClampMax": 1.0,
        "globalGain": global_gain,
    }
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.array([1, 2, 3], dtype=np.uint64),
        presynaptic_offsets=np.array([0, 1, 3, 3], dtype=np.uint32),
        postsynaptic_indices=np.array([1, 0, 2], dtype=np.uint32),
        contact_magnitudes=np.array([a, b, c], dtype=np.float32),
        presynaptic_signs=np.array([1, -1, 1], dtype=np.int8),
        input_channel_index=np.array([0, -1, -1], dtype=np.int32),
        input_weight=np.array([w_in, 0.0, 0.0], dtype=np.float32),
        output_population_index=np.array([-1, -1, 0], dtype=np.int32),
        output_weight=np.array([0.0, 0.0, w_out], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


def test_three_neuron_hand_graph_matches_closed_form():
    leak_rate, global_gain, a, b, c, w_in, w_out = 0.35, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
    graph = _make_three_neuron_graph(leak_rate, global_gain, a, b, c, w_in, w_out)
    result = transfer_matrix(graph)
    T = np.array(result["T"])
    assert T.shape == (1, 1)

    # Closed form (derived by hand, see tests_python/test_transfer.py's own
    # module doc comment for the derivation): with S = lambda*I - g*A and
    # A[1,0]=a, A[0,1]=-b, A[2,1]=-c (Dale's law: neuron 1's single sign
    # applies to both its outgoing edges),
    #   T = -(w_in * w_out * g^2 * a * c) / (lambda * (lambda^2 + g^2 * a * b))
    expected = -(w_in * w_out * global_gain**2 * a * c) / (leak_rate * (leak_rate**2 + global_gain**2 * a * b))
    assert T[0, 0] == pytest.approx(expected, rel=1e-9, abs=1e-12)

    assert result["conditionNumber"] > 0
    assert isinstance(result["illConditioned"], bool)
    assert result["illConditioned"] is False
    assert result["stable"] is True
    assert result["timeConstantSeconds"] is not None and result["timeConstantSeconds"] > 0


def _make_two_neuron_mutual_excitation_graph(leak_rate: float, global_gain: float, gain: float) -> "binfmt.GraphArrays":
    """Two neurons, both sign +1, each excites the other (0->1 and 1->0,
    both magnitude `gain`): `A = [[0, gain], [gain, 0]]`, real eigenvalues
    `+-gain` (a mutual-excitation loop, not an oscillator -- contrast
    `_make_three_neuron_graph`'s 0<->1 loop, whose opposite-sign neuron 1
    makes its `A` submatrix's eigenvalues purely imaginary regardless of
    magnitude, so it can never be unstable by this test's definition). No
    input/output neurons: only `spectralAbscissa`/`stable`/
    `timeConstantSeconds` are exercised here, not `T` itself."""
    metadata = {
        "formatVersion": 1,
        "neuronCount": 2,
        "edgeCount": 2,
        "inputChannelCount": 1,
        "outputPopulationCount": 1,
        "timestepSeconds": 1.0 / 30.0,
        "leakRate": leak_rate,
        "rateMin": -2.0,
        "rateMax": 2.0,
        "inputClampMin": -1.0,
        "inputClampMax": 1.0,
        "globalGain": global_gain,
    }
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.array([1, 2], dtype=np.uint64),
        presynaptic_offsets=np.array([0, 1, 2], dtype=np.uint32),
        postsynaptic_indices=np.array([1, 0], dtype=np.uint32),
        contact_magnitudes=np.array([gain, gain], dtype=np.float32),
        presynaptic_signs=np.array([1, 1], dtype=np.int8),
        input_channel_index=np.array([-1, -1], dtype=np.int32),
        input_weight=np.array([0.0, 0.0], dtype=np.float32),
        output_population_index=np.array([-1, -1], dtype=np.int32),
        output_weight=np.array([0.0, 0.0], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


def test_two_neuron_mutual_excitation_unstable_reports_no_time_constant():
    # A = [[0, 4], [4, 0]] has real eigenvalues +-4; spectral abscissa of
    # g*A = 1.0 * 4 = 4, far exceeding leakRate (0.35).
    graph = _make_two_neuron_mutual_excitation_graph(leak_rate=0.35, global_gain=1.0, gain=4.0)
    result = transfer_matrix(graph)
    assert result["spectralAbscissa"] == pytest.approx(4.0, rel=1e-9)
    assert result["stable"] is False
    assert result["timeConstantSeconds"] is None


def test_two_neuron_mutual_excitation_stable_reports_time_constant():
    # Same topology, small enough gain that g*A's spectral abscissa (0.1)
    # stays below leakRate (0.35): time constant is 1 / (0.35 - 0.1) = 4.
    graph = _make_two_neuron_mutual_excitation_graph(leak_rate=0.35, global_gain=1.0, gain=0.1)
    result = transfer_matrix(graph)
    assert result["stable"] is True
    # rel=1e-6, not 1e-9: `gain` (0.1) is stored as float32 in `contactMagnitudes`
    # (the wire format's own dtype), so `0.1`'s ~1e-7 relative rounding into
    # float32 propagates here -- this asserts against the graph's *actual*
    # stored magnitude, not idealized decimal 0.1.
    assert result["timeConstantSeconds"] == pytest.approx(1.0 / (0.35 - 0.1), rel=1e-6)


# ---------------------------------------------------------------------------
# Trace-graph fixture: exactness (float64 fixed-point) and TS finite-difference
# ---------------------------------------------------------------------------


def _graph_arrays_from_fixture(fixture: dict) -> "binfmt.GraphArrays":
    g = fixture["graph"]
    metadata = dict(g["metadata"])
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.array([int(x) for x in g["biologicalIds"]], dtype=np.uint64),
        presynaptic_offsets=np.array(g["presynapticOffsets"], dtype=np.uint32),
        postsynaptic_indices=np.array(g["postsynapticIndices"], dtype=np.uint32),
        contact_magnitudes=np.array(g["contactMagnitudes"], dtype=np.float32),
        presynaptic_signs=np.array(g["presynapticSigns"], dtype=np.int8),
        input_channel_index=np.array(g["inputChannelIndex"], dtype=np.int32),
        input_weight=np.array(g["inputWeight"], dtype=np.float32),
        output_population_index=np.array(g["outputPopulationIndex"], dtype=np.int32),
        output_weight=np.array(g["outputWeight"], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


def _float64_fixed_point_steady_state_map(graph: "binfmt.GraphArrays", iterations: int = 2_000_000) -> np.ndarray:
    """An independent float64 computation of `M = (lambda I - g A)^-1 B`:
    plain Euler relaxation (`r_{k+1} = r_k + dt * (-lambda*r_k + g*A@r_k +
    B)`, run from `r_0 = 0` for `B`'s columns simultaneously) rather than
    `numpy.linalg.solve` -- deliberately a *different* algorithm from
    `transfer.py`'s own dense solve, so this test cannot pass merely because
    both sides call the same numpy routine on the same inputs. Converges
    because the fixture graph is stable at its authored `globalGain` (its
    `stepModel`-side counterpart, `export-trace-graph-fixture.ts`, already
    verifies convergence and non-clamping when generating the committed
    fixture)."""
    matrices = build_dense_matrices(graph)
    A = matrices.adjacency
    B = matrices.input_matrix
    leak_rate = float(graph.metadata["leakRate"])
    global_gain = float(graph.metadata["globalGain"])
    dt = float(graph.metadata["timestepSeconds"])
    n = A.shape[0]
    R = np.zeros((n, B.shape[1]), dtype=np.float64)
    for _ in range(iterations):
        R = R + dt * (-leak_rate * R + global_gain * (A @ R) + B)
    return R


@pytest.fixture(scope="module")
def trace_fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"{FIXTURE_PATH} not generated (run scripts/analysis/export-trace-graph-fixture.ts)")
    with FIXTURE_PATH.open("r") as fh:
        return json.load(fh)


def test_trace_graph_matches_float64_fixed_point_iteration(trace_fixture):
    graph = _graph_arrays_from_fixture(trace_fixture)
    result = transfer_matrix(graph)
    T = np.array(result["T"])

    O = build_dense_matrices(graph).output_matrix
    steady_state_map = _float64_fixed_point_steady_state_map(graph)
    T_reference = O @ steady_state_map

    np.testing.assert_allclose(T, T_reference, rtol=1e-9, atol=1e-9)


def test_trace_graph_matches_ts_finite_difference(trace_fixture):
    graph = _graph_arrays_from_fixture(trace_fixture)
    result = transfer_matrix(graph)
    T = np.array(result["T"])

    reference = np.array(trace_fixture["steadyStateByChannel"]).T  # (outputPopulationCount, inputChannelCount)
    assert reference.shape == T.shape

    relative_error = np.abs(T - reference) / np.maximum(np.abs(reference), 1e-9)
    assert float(relative_error.max()) < 1e-3


def test_trace_graph_reports_spectral_abscissa_and_condition_number(trace_fixture):
    graph = _graph_arrays_from_fixture(trace_fixture)
    result = transfer_matrix(graph)

    leak_rate = float(graph.metadata["leakRate"])
    global_gain = float(graph.metadata["globalGain"])
    A = build_dense_matrices(graph).adjacency
    expected_abscissa = float(global_gain * np.max(np.linalg.eigvals(A).real))
    expected_condition_number = float(np.linalg.cond(leak_rate * np.eye(A.shape[0]) - global_gain * A))

    assert result["spectralAbscissa"] == pytest.approx(expected_abscissa, rel=1e-9)
    assert result["conditionNumber"] == pytest.approx(expected_condition_number, rel=1e-6)
    assert result["stable"] is (expected_abscissa < leak_rate)
    assert result["illConditioned"] is (expected_condition_number > ILL_CONDITIONED_THRESHOLD)
    assert 0.0 < result["discretizedSpectralRadius"] < 1.0
    assert result["discretizedStable"] is True


# ---------------------------------------------------------------------------
# CLI: determinism across --workers, and sha256 verification
# ---------------------------------------------------------------------------


def _write_cli_fixture(root: Path) -> tuple[Path, Path, Path]:
    """A tiny 2-graph (`biological` + one `rewired` seed) fixture on disk for
    `transfer.py`'s CLI, independent of the trace-graph fixture above."""
    graphs_dir = root / "graphs"
    graphs_dir.mkdir()

    bio_graph = _make_three_neuron_graph(leak_rate=0.35, global_gain=0.5, a=1.0, b=1.0, c=1.0, w_in=1.0, w_out=1.0)
    bio_binary = binfmt.encode_graph_binary(bio_graph)
    bio_sha256 = binfmt.sha256_hex(bio_binary)
    bio_path = root / "biological.bin.gz"
    bio_path.write_bytes(gzip.compress(bio_binary))

    rewired_graph = _make_three_neuron_graph(leak_rate=0.35, global_gain=0.5, a=2.0, b=1.0, c=1.0, w_in=1.0, w_out=1.0)
    rewired_binary = binfmt.encode_graph_binary(rewired_graph)
    rewired_sha256 = binfmt.sha256_hex(rewired_binary)
    rewired_gzip = gzip.compress(rewired_binary)
    rewired_path = graphs_dir / "rewired-seed0.bin.gz"
    rewired_path.write_bytes(rewired_gzip)

    index = {
        "sourceArtifact": "biological.bin.gz",
        "sourceSha256": bio_sha256,
        "rewireSourceSha256": "0" * 64,
        "seeds": [
            {
                "seed": 0,
                "artifact": "rewired-seed0.bin.gz",
                "binarySha256": rewired_sha256,
                "binaryBytes": len(rewired_binary),
                "gzipSha256": binfmt.sha256_hex(rewired_gzip),
                "gzipBytes": len(rewired_gzip),
                "stats": {"acceptedSwaps": 1, "attempts": 1},
            }
        ],
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index))
    return bio_path, index_path, graphs_dir


def _run_transfer_cli(bio_path: Path, index_path: Path, graphs_dir: Path, out_path: Path, workers: int) -> subprocess.CompletedProcess:
    script = REPO_ROOT / "scripts" / "analysis" / "transfer.py"
    # sys.executable is this test's own interpreter (the uv-managed venv's
    # python), so the subprocess sees the same installed numpy -- no need to
    # route back through `uv run` a second time. Inherits the real
    # environment (already carrying the required single-threaded BLAS vars
    # from tests_python/conftest.py's `os.environ.setdefault` calls).
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--biological",
            str(bio_path),
            "--index",
            str(index_path),
            "--graphs-dir",
            str(graphs_dir),
            "--out",
            str(out_path),
            "--steady-state-dir",
            str(out_path.parent / f"steady-state-{workers}"),
            "--workers",
            str(workers),
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_transfer_cli_is_byte_identical_across_worker_counts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bio_path, index_path, graphs_dir = _write_cli_fixture(root)

        out1 = root / "transfer-w1.json"
        result1 = _run_transfer_cli(bio_path, index_path, graphs_dir, out1, workers=1)
        assert result1.returncode == 0, result1.stderr

        out3 = root / "transfer-w3.json"
        result3 = _run_transfer_cli(bio_path, index_path, graphs_dir, out3, workers=3)
        assert result3.returncode == 0, result3.stderr

        assert out1.read_bytes() == out3.read_bytes()

        payload = json.loads(out1.read_text())
        assert set(payload["graphs"].keys()) == {"biological", "disconnected", "rewired-0"}


def test_transfer_cli_rejects_a_tampered_rewired_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bio_path, index_path, graphs_dir = _write_cli_fixture(root)
        victim = graphs_dir / "rewired-seed0.bin.gz"
        # Flip a byte *inside* the compressed stream, not append one: gzip's
        # own decompressor tolerates (and ignores) trailing garbage bytes
        # after a complete, valid DEFLATE stream, so appending would decompress
        # to byte-identical content and this test would prove nothing (the
        # same distinction `tests/unit/null-evaluate.test.ts`'s own
        # "flipped byte" regression test makes for the TypeScript path).
        # Flipping a byte partway through the stream changes the decompressed
        # payload's sha256 (or breaks decompression outright), which is what
        # `load_verified_graph`'s post-decompression check actually guards.
        bytes_ = bytearray(victim.read_bytes())
        bytes_[len(bytes_) // 2] ^= 0xFF
        victim.write_bytes(bytes_)

        out = root / "transfer-tampered.json"
        result = _run_transfer_cli(bio_path, index_path, graphs_dir, out, workers=1)
        assert result.returncode != 0
        assert "does not match expected" in result.stderr or "Error" in result.stderr
