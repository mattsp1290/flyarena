"""`.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2 change
surface: "tests_python/test_features.py (new): Hand graphs with known path
lengths, motif counts, and reciprocity."

Primary fixture: a 4-neuron graph small enough to enumerate every walk,
motif, and reciprocal pair by hand (see this module's inline derivation next
to each assertion group). Neuron 0 is the only input neuron (channel
`foodBearing`); neuron 3 is the only output neuron (population `thrust`).
Edges (`pre -> post`, magnitude, `presynapticSigns[pre]`):

    0 -> 1  (mag 2, sign[0] = +1)
    1 -> 2  (mag 3, sign[1] = +1)
    1 -> 3  (mag 5, sign[1] = +1)
    2 -> 3  (mag 1, sign[2] = -1)
    3 -> 1  (mag 1, sign[3] = +1)
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import binfmt  # noqa: E402

from features import OBSERVATION_CHANNELS, OUTPUT_POPULATIONS, graph_features, motif_counts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_four_neuron_graph() -> "binfmt.GraphArrays":
    metadata = {
        "formatVersion": 1,
        "neuronCount": 4,
        "edgeCount": 5,
        "inputChannelCount": len(OBSERVATION_CHANNELS),
        "outputPopulationCount": len(OUTPUT_POPULATIONS),
        "timestepSeconds": 1.0 / 30.0,
        "leakRate": 0.35,
        "rateMin": -2.0,
        "rateMax": 2.0,
        "inputClampMin": -1.0,
        "inputClampMax": 1.0,
        "globalGain": 0.5,
    }
    # CSR rows: row0=[1] (0->1); row1=[2,3] (1->2, 1->3); row2=[3] (2->3);
    # row3=[1] (3->1). offsets = [0,1,3,4,5].
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.array([1, 2, 3, 4], dtype=np.uint64),
        presynaptic_offsets=np.array([0, 1, 3, 4, 5], dtype=np.uint32),
        postsynaptic_indices=np.array([1, 2, 3, 3, 1], dtype=np.uint32),
        contact_magnitudes=np.array([2, 3, 5, 1, 1], dtype=np.float32),
        presynaptic_signs=np.array([1, 1, -1, 1], dtype=np.int8),
        input_channel_index=np.array([OBSERVATION_CHANNELS.index("foodBearing"), -1, -1, -1], dtype=np.int32),
        input_weight=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        output_population_index=np.array([-1, -1, -1, OUTPUT_POPULATIONS.index("thrust")], dtype=np.int32),
        output_weight=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


@pytest.fixture(scope="module")
def features() -> dict:
    return graph_features(_make_four_neuron_graph())


def test_path_lengths(features):
    assert features["pathLengths"]["foodBearing->thrust"] == 2  # 0 -> 1 -> 3
    # Every other (channel, population) pair is unreachable: no input
    # neuron for any channel but foodBearing, no output neuron for any
    # population but thrust.
    for channel in OBSERVATION_CHANNELS:
        for population in OUTPUT_POPULATIONS:
            if (channel, population) == ("foodBearing", "thrust"):
                continue
            assert features["pathLengths"][f"{channel}->{population}"] is None
    assert features["meanPathLength"] == pytest.approx(2.0)
    assert features["reachablePairCount"] == 1
    assert features["totalPairCount"] == len(OBSERVATION_CHANNELS) * len(OUTPUT_POPULATIONS)


def test_signed_path_counts(features):
    # Walks 0 -> ... -> 3 (the only output neuron), length <= 3, split by
    # the sign product of every traversed presynaptic neuron:
    #   length 1: none (no direct edge 0 -> 3).
    #   length 2: 0 -> 1 -> 3 (edges (0,1),(1,3) both exist); sign product
    #     sign[0] * sign[1] = (+1)*(+1) = +1 (excitatory).
    #   length 3: 0 -> 1 -> 2 -> 3 (edges (0,1),(1,2),(2,3) all exist); sign
    #     product sign[0]*sign[1]*sign[2] = (+1)*(+1)*(-1) = -1 (inhibitory).
    #     (0 -> 1 -> 3 -> ? needs a self-loop at 3, which does not exist.)
    assert features["excitatoryPathCount"]["thrust"] == 1
    assert features["inhibitoryPathCount"]["thrust"] == 1
    assert features["excitatoryPathCount"]["yaw"] == 0
    assert features["inhibitoryPathCount"]["yaw"] == 0
    assert features["excitatoryPathCount"]["brake"] == 0
    assert features["inhibitoryPathCount"]["brake"] == 0


def test_reciprocity(features):
    # 5 directed edges total; (1,3) and (3,1) are each other's reverse (the
    # only reciprocal pair) -- 2 of the 5 edges have their reverse present.
    assert features["reciprocity"] == pytest.approx(2.0 / 5.0)


def test_weight_balance(features):
    # Neuron 3 (thrust) receives (1,3, mag 5, sign +1) -> +5 and
    # (2,3, mag 1, sign -1) -> -1: signed sum = 4.
    assert features["weightBalance"]["thrust"] == pytest.approx(4.0)
    assert features["weightBalance"]["yaw"] == pytest.approx(0.0)
    assert features["weightBalance"]["brake"] == pytest.approx(0.0)


def test_motif_counts(features):
    # 2-cycle: exactly the unordered pair {1, 3} (edges (1,3) and (3,1)).
    assert features["twoCycleCount"] == 1
    # Feed-forward triangle: exactly (a, b, c) = (1, 2, 3) -- edges
    # (1,2), (2,3), (1,3) all present. (0, ?, ?) has no candidate since 0's
    # only out-edge is to 1, and 1 -> 1 does not exist.
    assert features["feedForwardTriangleCount"] == 1


def test_weighted_in_degree(features):
    # Neuron 3 (thrust): unsigned in-degree = |5| + |1| = 6, mean over 1
    # neuron = 6.
    assert features["weightedInDegree"]["thrust"] == pytest.approx(6.0)
    assert features["weightedInDegree"]["yaw"] is None  # no neurons in this population
    assert features["weightedInDegree"]["brake"] is None


def test_edge_and_neuron_counts(features):
    assert features["edgeCount"] == 5
    assert features["neuronCount"] == 4


# ---------------------------------------------------------------------------
# motif_counts: self-loops must not be double-counted or fabricate a triangle
# ---------------------------------------------------------------------------


def test_motif_counts_excludes_self_loops():
    # 2 neurons: 0 -> 0 (self-loop), 0 -> 1, 1 -> 0 (a genuine 2-cycle).
    # Without excluding the diagonal, the self-loop could otherwise inflate
    # the 2-cycle count (E[0,0] & E[0,0].T is trivially true) or fabricate a
    # feed-forward triangle through b == a/b == c degenerate walks -- see
    # `motif_counts`'s own doc comment for why zeroing the diagonal is exact,
    # not approximate, for both corrections.
    edge_bool = np.array(
        [
            [True, True],  # row 0 (post=0): edges into neuron 0 from pre=0 (self-loop) and pre=1
            [True, False],  # row 1 (post=1): edge into neuron 1 from pre=0
        ]
    )
    result = motif_counts(edge_bool)
    assert result["twoCycleCount"] == 1
    assert result["feedForwardTriangleCount"] == 0


# ---------------------------------------------------------------------------
# CLI: determinism across --workers, disconnected control, sha256 verification
# ---------------------------------------------------------------------------


def _write_cli_fixture(root: Path) -> tuple[Path, Path, Path]:
    graphs_dir = root / "graphs"
    graphs_dir.mkdir()

    bio_graph = _make_four_neuron_graph()
    bio_binary = binfmt.encode_graph_binary(bio_graph)
    bio_sha256 = binfmt.sha256_hex(bio_binary)
    bio_path = root / "biological.bin.gz"
    bio_path.write_bytes(gzip.compress(bio_binary))

    rewired_graph = _make_four_neuron_graph()
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


def _run_features_cli(bio_path: Path, index_path: Path, graphs_dir: Path, out_path: Path, workers: int) -> subprocess.CompletedProcess:
    script = REPO_ROOT / "scripts" / "analysis" / "features.py"
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
            "--workers",
            str(workers),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_features_cli_is_byte_identical_across_worker_counts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bio_path, index_path, graphs_dir = _write_cli_fixture(root)

        out1 = root / "features-w1.json"
        result1 = _run_features_cli(bio_path, index_path, graphs_dir, out1, workers=1)
        assert result1.returncode == 0, result1.stderr

        out3 = root / "features-w3.json"
        result3 = _run_features_cli(bio_path, index_path, graphs_dir, out3, workers=3)
        assert result3.returncode == 0, result3.stderr

        assert out1.read_bytes() == out3.read_bytes()

        payload = json.loads(out1.read_text())
        assert set(payload["graphs"].keys()) == {"biological", "disconnected", "rewired-0"}
        # The disconnected control (built through disconnected_graph_arrays,
        # not a hand-maintained constant) has no edges at all.
        assert payload["graphs"]["disconnected"]["edgeCount"] == 0
        assert payload["graphs"]["disconnected"]["reciprocity"] == 0.0
        assert payload["graphs"]["biological"]["edgeCount"] == 5
