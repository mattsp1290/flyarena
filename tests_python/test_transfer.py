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
from graph_io import DenseGraphMatrices, build_dense_matrices, canonical_json_text  # noqa: E402
from transfer import (  # noqa: E402
    ILL_CONDITIONED_THRESHOLD,
    OBSERVATION_CHANNEL_INDEX,
    OUTPUT_POPULATION_INDEX,
    _compute_transfer,
    transfer_matrix,
)

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
    every outgoing edge). Neuron 2: output (population 0), sign -1, no
    outgoing edges. CSR rows: row 0 = [1] (edge to neuron 1); row 1 = [0, 2]
    (edges to neurons 0 and 2, ascending); row 2 = [] (empty).

    Neuron 2's own sign is deliberately *not* +1 (unlike neurons 0 and 1's
    signs, which are meaningfully constrained by the topology): neuron 2 has
    no outgoing edges, so its sign can never affect the correct
    (`sign[pre]`) computation of `T` -- but it *would* affect a buggy
    `sign[post]` implementation's result for the edge `1 -> 2`, since that
    edge's "post" is neuron 2. Setting it to -1 (matching neuron 1's sign,
    rather than +1) is what makes the closed-form assertion below able to
    tell the two conventions apart -- see
    `test_three_neuron_hand_graph_matches_closed_form`'s own comment (a
    dual-review finding: the earlier `+1` choice happened to make `sign[pre]`
    and `sign[post]` produce the same `T` for this exact topology, so this
    test previously could not have caught that class of bug)."""
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
        presynaptic_signs=np.array([1, -1, -1], dtype=np.int8),
        input_channel_index=np.array([0, -1, -1], dtype=np.int32),
        input_weight=np.array([w_in, 0.0, 0.0], dtype=np.float32),
        output_population_index=np.array([-1, -1, 0], dtype=np.int32),
        output_weight=np.array([0.0, 0.0, w_out], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


def test_three_neuron_hand_graph_matches_closed_form():
    # a, b, c deliberately distinct (not all 1): with a == b == c the closed
    # form below can't tell `a*c` apart from `b*c` or `a*b`, so a transposed-
    # index or swapped-parameter bug could still produce the right number by
    # coincidence (a dual-review finding).
    leak_rate, global_gain, a, b, c, w_in, w_out = 0.35, 0.5, 1.5, 0.75, 2.0, 1.0, 1.0
    graph = _make_three_neuron_graph(leak_rate, global_gain, a, b, c, w_in, w_out)
    result = transfer_matrix(graph)
    T = np.array(result["T"])
    assert T.shape == (1, 1)

    # Closed form, derived by hand from S @ r = e0 * w_in * u0 where
    # S = lambda*I - g*A and A[1,0]=a, A[0,1]=-b, A[2,1]=-c (Dale's law:
    # neuron 1's single sign applies to both its outgoing edges; see
    # `_make_three_neuron_graph`'s doc comment for why neuron 2's own sign
    # is also -1, not +1):
    #   row 1: -g*a*r0 + lambda*r1 = 0            => r1 = (g*a/lambda) * r0
    #   row 2:  g*c*r1 + lambda*r2 = 0             => r2 = -(g*c/lambda) * r1
    #   row 0: lambda*r0 + g*b*r1 = w_in*u0        => r0 = w_in*u0*lambda / (lambda^2 + g^2*a*b)
    # Substituting r1 into r2, then r0:
    #   T = w_out * r2/u0 = -(w_in * w_out * g^2 * a * c) / (lambda * (lambda^2 + g^2 * a * b))
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
    `transfer.py`'s CLI. Uses the *trace graph* fixture (24 neurons, the
    production `outputPopulationCount=3`/`inputChannelCount=8` shape), not
    `_make_three_neuron_graph`: the CLI's per-graph dispatch now runs with
    `strict_shape=True` (a dual-review finding -- `turnGain`/`approachGain`
    must raise, not silently go `None`, for a production-shaped graph), and
    the 3-neuron hand graph's 1x1 `T` would trip that guard.

    The "rewired" graph is the trace graph with its first edge's magnitude
    scaled by 1.5 -- structurally identical (same shape, same CSR topology)
    but numerically distinct, so its `T` differs from `biological`'s. This
    matters for what these tests can actually detect (a round-2 dual-review
    finding): an earlier version of this fixture reused the *exact same*
    bytes for both graphs, which meant a worker mixing up which result or
    steady-state sidecar belongs to which graph ID would have gone
    completely undetected by the determinism test below, and a
    `graphBinarySha256`/`sidecarSha256` swap in the manifest would have been
    invisible too -- both graphs looked identical, so nothing could tell a
    correct assignment apart from a scrambled one."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"{FIXTURE_PATH} not generated (run scripts/analysis/export-trace-graph-fixture.ts)")
    with FIXTURE_PATH.open("r") as fh:
        fixture = json.load(fh)
    graph = _graph_arrays_from_fixture(fixture)

    graphs_dir = root / "graphs"
    graphs_dir.mkdir()

    bio_binary = binfmt.encode_graph_binary(graph)
    bio_sha256 = binfmt.sha256_hex(bio_binary)
    bio_path = root / "biological.bin.gz"
    bio_path.write_bytes(gzip.compress(bio_binary))

    perturbed_magnitudes = graph.contact_magnitudes.copy()
    perturbed_magnitudes[0] = perturbed_magnitudes[0] * np.float32(1.5)
    rewired_graph = binfmt.GraphArrays(
        metadata=graph.metadata,
        biological_ids=graph.biological_ids,
        presynaptic_offsets=graph.presynaptic_offsets,
        postsynaptic_indices=graph.postsynaptic_indices,
        contact_magnitudes=perturbed_magnitudes,
        presynaptic_signs=graph.presynaptic_signs,
        input_channel_index=graph.input_channel_index,
        input_weight=graph.input_weight,
        output_population_index=graph.output_population_index,
        output_weight=graph.output_weight,
    )
    rewired_binary = binfmt.encode_graph_binary(rewired_graph)
    assert rewired_binary != bio_binary  # a perturbation that silently no-ops would defeat this fixture's own point
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
        # biological and rewired-0 are numerically distinct (see
        # _write_cli_fixture's doc comment) -- if this ever failed, every
        # assertion below it would pass vacuously regardless of whether
        # graph/sidecar attribution is actually correct.
        assert payload["graphs"]["biological"]["T"] != payload["graphs"]["rewired-0"]["T"]

        # The steady-state manifest is the round-1/round-2 dual-review fix
        # (I1/I2): every entry must tie the graph it was computed from
        # (`graphBinarySha256`) to the sidecar bytes actually on disk
        # (`sidecarSha256`), and the manifest itself must be exactly as
        # deterministic as `transfer.json` -- a worker-count-dependent
        # manifest would defeat the whole point of verifying it up front in
        # `regime-check.ts`.
        manifest1_path = root / "steady-state-1" / "manifest.json"
        manifest3_path = root / "steady-state-3" / "manifest.json"
        assert manifest1_path.read_bytes() == manifest3_path.read_bytes()
        manifest = json.loads(manifest1_path.read_text())
        assert manifest["version"] == 1
        assert manifest["rewireSourceSha256"] == "0" * 64
        assert set(manifest["graphs"].keys()) == {"biological", "disconnected", "rewired-0"}
        bio_sha256 = binfmt.sha256_hex(gzip.decompress(bio_path.read_bytes()))
        for graph_id, entry in manifest["graphs"].items():
            sidecar_path = root / "steady-state-1" / f"{graph_id}.steadystate.f64"
            assert entry["sidecarSha256"] == binfmt.sha256_hex(sidecar_path.read_bytes())
        assert manifest["graphs"]["biological"]["graphBinarySha256"] == bio_sha256
        assert manifest["graphs"]["disconnected"]["graphBinarySha256"] == bio_sha256
        index_payload = json.loads(index_path.read_text())
        rewired_sha256 = index_payload["seeds"][0]["binarySha256"]
        assert manifest["graphs"]["rewired-0"]["graphBinarySha256"] == rewired_sha256
        # And a sha mismatch really is caught: corrupt one entry and confirm
        # regime-check.ts's verifySteadyStateManifest would reject it (the
        # TS-side unit tests exercise verifySteadyStateManifest directly;
        # this only re-confirms the Python side actually wrote a sha that
        # differs from a tampered one, i.e. the check has something to catch).
        assert manifest["graphs"]["biological"]["graphBinarySha256"] != manifest["graphs"]["rewired-0"]["graphBinarySha256"]


def test_singular_system_is_flagged_and_serializable():
    # `A = diag(0.5, 0.0)` at `leakRate=0.5`, `globalGain=1.0`: neuron 0's
    # block (`0.5 - 1.0*0.5 = 0`) is exactly singular, and neuron 1 is
    # completely isolated (no edges at all) -- together the whole 2x2
    # `system_matrix` is exactly singular (`np.linalg.cond` returns `inf`,
    # `np.linalg.solve` raises `LinAlgError`), reproducing the round-2
    # dual-review finding: `condition_number` must become `None`, not the
    # raw `inf` `np.linalg.cond` returns, or `canonical_json_text`'s
    # `allow_nan=False` raises when the *whole batch's* result is written
    # (see `_compute_transfer`'s `singular` branch doc comment).
    matrices = DenseGraphMatrices(
        adjacency=np.diag([0.5, 0.0]),
        input_matrix=np.zeros((2, len(OBSERVATION_CHANNEL_INDEX))),
        output_matrix=np.zeros((len(OUTPUT_POPULATION_INDEX), 2)),
    )
    computation = _compute_transfer(matrices, leak_rate=0.5, global_gain=1.0, timestep_seconds=1.0 / 30.0, strict_shape=True)
    result = computation.result

    assert result["singular"] is True
    assert result["illConditioned"] is True
    assert result["T"] is None
    assert result["conditionNumber"] is None
    assert result["stable"] is False
    assert result["timeConstantSeconds"] is None
    assert result["turnGain"] is None
    assert result["approachGain"] is None
    assert np.isfinite(result["spectralAbscissa"])
    assert np.isfinite(result["discretizedSpectralRadius"])

    # The actual failure mode this test reproduces: canonical_json_text must
    # not raise on a singular graph's result.
    canonical_json_text(result)


def test_compute_transfer_strict_shape_raises_on_a_non_production_shape():
    # `_compute_transfer`'s `strict_shape=True` path (the CLI's own
    # `_one_graph`/`_one_disconnected`) is otherwise never exercised by any
    # test: the CLI fixture uses a production-shaped (3x8) graph precisely
    # so it does NOT hit this raise -- so nothing previously proved the
    # guard actually fires (a round-2 dual-review finding).
    graph = _make_three_neuron_graph(leak_rate=0.35, global_gain=0.5, a=1.5, b=0.75, c=2.0, w_in=1.0, w_out=1.0)
    matrices = build_dense_matrices(graph)
    with pytest.raises(ValueError, match=r"expected \(3, 8\)"):
        _compute_transfer(matrices, leak_rate=0.35, global_gain=0.5, timestep_seconds=1.0 / 30.0, strict_shape=True)


def test_transfer_cli_rejects_a_tampered_rewired_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bio_path, index_path, graphs_dir = _write_cli_fixture(root)
        victim = graphs_dir / "rewired-seed0.bin.gz"
        # Substitute a *valid but different* graph under the expected
        # filename, rather than corrupting bytes: a byte flip can land
        # anywhere in the DEFLATE stream and either break decompression
        # outright (any "Error" in stderr, not necessarily the sha check) or
        # -- if it lands in trailing garbage gzip tolerates -- decompress to
        # byte-identical content and catch nothing at all. A valid
        # substitute graph decompresses cleanly every time and is
        # deterministically caught by exactly one thing: the decompressed
        # sha256 not matching `index.json`'s `binarySha256` (a dual-review
        # finding -- the previous version of this test could pass on any
        # crash, not specifically because sha verification worked).
        substitute = _make_three_neuron_graph(leak_rate=0.35, global_gain=0.5, a=3.0, b=1.0, c=1.0, w_in=1.0, w_out=1.0)
        victim.write_bytes(gzip.compress(binfmt.encode_graph_binary(substitute)))

        out = root / "transfer-tampered.json"
        result = _run_transfer_cli(bio_path, index_path, graphs_dir, out, workers=1)
        assert result.returncode != 0
        assert "GraphVerificationError" in result.stderr
        assert "does not match expected" in result.stderr
        assert not out.exists()
