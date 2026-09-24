"""`scripts/data/rewire_batch.py` -- the batch entry point over
`rewire.rewire_graph` (`.agents/plans/rewiring-null/01-rewired-graph-generation.md`,
WP1).

Covers:
- seed 0 from the batch script is byte-identical to the shipped
  `public/data/malecns-arena-v1-rewired-seed0.bin.gz` control arm (both
  ultimately call the same `rewire.rewire_graph(graph, seed=0)` over the
  same committed source graph, so this mostly proves the batch script's own
  I/O plumbing -- gzip writing, path naming -- introduces no divergence from
  `rewire.py`'s own CLI),
- two runs over the same seed range produce byte-identical `index.json`,
- the batch script's own output preserves the rewiring invariants
  `rewire.py`/`rewire_graph` already claim (in/out degree, weight multiset,
  signs, edge count) -- the exhaustive version of these checks lives in
  `tests_python/test_compile.py`; this file only spot-checks that the batch
  wrapper hands back the same graphs `rewire_graph` itself produced, not a
  second implementation of the invariant suite.

Runs entirely against the committed, small `public/data/malecns-arena-v1.bin.gz`
compiled artifact -- no raw MaleCNS data required, matching
`test_compile.py`'s "no MaleCNS raw data required" convention.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import binfmt  # noqa: E402
import rewire  # noqa: E402
import rewire_batch  # noqa: E402

PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
SOURCE_ARTIFACT = PUBLIC_DATA_DIR / "malecns-arena-v1.bin.gz"
SHIPPED_REWIRED_SEED0 = PUBLIC_DATA_DIR / "malecns-arena-v1-rewired-seed0.bin.gz"


def _degree_sequences(graph: binfmt.GraphArrays) -> tuple[np.ndarray, np.ndarray]:
    """(out_degree_per_node, in_degree_per_node)."""
    out_degree = np.diff(graph.presynaptic_offsets.astype(np.int64))
    neuron_count = int(graph.metadata["neuronCount"])
    in_degree = np.zeros(neuron_count, dtype=np.int64)
    np.add.at(in_degree, graph.postsynaptic_indices.astype(np.int64), 1)
    return out_degree, in_degree


def test_seed0_is_byte_identical_to_the_shipped_rewired_control_arm(tmp_path):
    out_dir = tmp_path / "graphs"
    index = rewire_batch.run_batch(in_path=SOURCE_ARTIFACT, seeds=range(0, 1), out_dir=out_dir)

    assert len(index["seeds"]) == 1
    entry = index["seeds"][0]
    assert entry["seed"] == 0

    batch_gzip_bytes = (out_dir / entry["artifact"]).read_bytes()
    shipped_gzip_bytes = SHIPPED_REWIRED_SEED0.read_bytes()
    assert batch_gzip_bytes == shipped_gzip_bytes, (
        "rewire_batch.py's seed-0 output must be byte-identical to the shipped "
        "public/data/malecns-arena-v1-rewired-seed0.bin.gz control arm"
    )

    with gzip.open(SHIPPED_REWIRED_SEED0, "rb") as fh:
        shipped_binary = fh.read()
    assert entry["binarySha256"] == binfmt.sha256_hex(shipped_binary)

    manifest = json.loads((PUBLIC_DATA_DIR / "malecns-arena-v1.manifest.json").read_text())
    assert entry["binarySha256"] == manifest["rewiredArms"]["seed0"]["binarySha256"]
    assert entry["gzipSha256"] == manifest["rewiredArms"]["seed0"]["gzipSha256"]


def test_two_runs_over_the_same_seed_range_give_identical_index_json(tmp_path):
    out_dir_a = tmp_path / "run-a"
    out_dir_b = tmp_path / "run-b"
    index_a = rewire_batch.run_batch(in_path=SOURCE_ARTIFACT, seeds=range(0, 3), out_dir=out_dir_a)
    index_b = rewire_batch.run_batch(in_path=SOURCE_ARTIFACT, seeds=range(0, 3), out_dir=out_dir_b)

    serialized_a = json.dumps(index_a, indent=2, sort_keys=True)
    serialized_b = json.dumps(index_b, indent=2, sort_keys=True)
    assert serialized_a == serialized_b

    assert [entry["seed"] for entry in index_a["seeds"]] == [0, 1, 2]
    # Every seed's rewired binary is also byte-identical across the two runs,
    # not just the recorded hashes (a hash collision would be undetectable
    # from the JSON diff alone).
    for entry in index_a["seeds"]:
        artifact = entry["artifact"]
        assert (out_dir_a / artifact).read_bytes() == (out_dir_b / artifact).read_bytes()


def test_main_cli_writes_the_requested_seed_range_and_index_json(tmp_path):
    # Snapshotted *before* the CLI runs -- read after, this assertion would
    # trivially compare the file with itself and could never fail even if
    # the CLI rewrote the manifest.
    manifest_path = PUBLIC_DATA_DIR / "malecns-arena-v1.manifest.json"
    manifest_bytes_before = manifest_path.read_bytes()

    out_dir = tmp_path / "cli-graphs"
    exit_code = rewire_batch.main(
        [
            "--in-path",
            str(SOURCE_ARTIFACT),
            "--seeds",
            "0:3",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    written = sorted(p.name for p in out_dir.glob("malecns-arena-v1-rewired-seed*.bin.gz"))
    assert written == [
        "malecns-arena-v1-rewired-seed0.bin.gz",
        "malecns-arena-v1-rewired-seed1.bin.gz",
        "malecns-arena-v1-rewired-seed2.bin.gz",
    ]

    index_path = out_dir / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert [entry["seed"] for entry in index["seeds"]] == [0, 1, 2]
    assert index["sourceArtifact"] == SOURCE_ARTIFACT.name
    assert index["rewireSourceSha256"] == binfmt.sha256_hex(Path(rewire.__file__).read_bytes())
    assert index["binfmtSourceSha256"] == binfmt.sha256_hex(Path(binfmt.__file__).read_bytes())
    assert index["numpyVersion"] == np.__version__
    assert index["params"] == {"allowSelfLoops": False, "swapAttemptsMultiplier": rewire.DEFAULT_SWAP_ATTEMPTS_MULTIPLIER}

    # The CLI must never touch the shipped product manifest.
    assert "malecns-arena-v1.manifest.json" not in {p.name for p in out_dir.iterdir()}
    assert manifest_path.read_bytes() == manifest_bytes_before


def test_custom_index_out_path_is_honored(tmp_path):
    out_dir = tmp_path / "graphs"
    index_out = tmp_path / "elsewhere" / "custom-index.json"
    exit_code = rewire_batch.main(
        [
            "--in-path",
            str(SOURCE_ARTIFACT),
            "--seeds",
            "0:1",
            "--out-dir",
            str(out_dir),
            "--index-out",
            str(index_out),
        ]
    )
    assert exit_code == 0
    assert index_out.exists()
    assert not (out_dir / "index.json").exists()


@pytest.mark.parametrize("spec", ["0", "0:", ":5", "abc:5", "0:abc", "-1:3"])
def test_parse_seed_range_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        rewire_batch.parse_seed_range(spec)


def test_parse_seed_range_rejects_empty_or_reversed_ranges():
    with pytest.raises(ValueError):
        rewire_batch.parse_seed_range("5:5")
    with pytest.raises(ValueError):
        rewire_batch.parse_seed_range("5:2")


def test_parse_seed_range_is_end_exclusive():
    assert list(rewire_batch.parse_seed_range("0:500")) == list(range(500))
    assert list(rewire_batch.parse_seed_range("10:13")) == [10, 11, 12]


def test_batch_output_preserves_rewiring_invariants_for_every_seed(tmp_path):
    """Spot-check (not the exhaustive suite -- see this file's module doc)
    that the graphs the batch script writes to disk, once decoded back from
    their gzip artifacts, still satisfy the invariants `rewire_graph` claims
    to preserve: this exercises the batch script's own encode/gzip/decode
    round trip, not `rewire_graph` itself."""
    with gzip.open(SOURCE_ARTIFACT, "rb") as fh:
        source_binary = fh.read()
    source_graph = rewire.decode_graph_binary(source_binary)
    source_out_degree, source_in_degree = _degree_sequences(source_graph)

    out_dir = tmp_path / "graphs"
    index = rewire_batch.run_batch(in_path=SOURCE_ARTIFACT, seeds=range(0, 2), out_dir=out_dir)

    for entry in index["seeds"]:
        with gzip.open(out_dir / entry["artifact"], "rb") as fh:
            rewired_binary = fh.read()
        rewired_graph = rewire.decode_graph_binary(rewired_binary)
        binfmt.validate_graph(rewired_graph)

        out_degree, in_degree = _degree_sequences(rewired_graph)
        assert np.array_equal(out_degree, source_out_degree)
        assert np.array_equal(in_degree, source_in_degree)
        assert int(rewired_graph.metadata["edgeCount"]) == int(source_graph.metadata["edgeCount"])
        assert np.array_equal(rewired_graph.presynaptic_signs, source_graph.presynaptic_signs)
        assert np.array_equal(rewired_graph.biological_ids, source_graph.biological_ids)
        # Edge-weight multiset: a swap only ever changes an edge's target,
        # never its contactMagnitudes value, so the full set of weights
        # (irrespective of which row each now sorts into) is preserved --
        # the per-presynaptic-neuron version of this same invariant is
        # already exhaustively covered by
        # tests_python/test_compile.py::test_rewire_preserves_edge_weight_multiset_per_presynaptic_neuron.
        assert np.array_equal(
            np.sort(rewired_graph.contact_magnitudes), np.sort(source_graph.contact_magnitudes)
        )
        # Per-node I/O maps are copied through unmodified by rewire_graph.
        assert np.array_equal(rewired_graph.input_channel_index, source_graph.input_channel_index)
        assert np.array_equal(rewired_graph.input_weight, source_graph.input_weight)
        assert np.array_equal(rewired_graph.output_population_index, source_graph.output_population_index)
        assert np.array_equal(rewired_graph.output_weight, source_graph.output_weight)

        assert entry["stats"]["acceptedSwaps"] <= entry["stats"]["attempts"]
        # Any seed with acceptedSwaps < edgeCount is still included (never
        # silently dropped) -- see 01-rewired-graph-generation.md's risks.
        assert entry["stats"]["edgeCount"] == int(source_graph.metadata["edgeCount"])
