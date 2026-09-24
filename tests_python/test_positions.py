"""Tests for `scripts/data/positions.py`, driven entirely from tiny fixtures
(a fixture graph built from the existing compiler CSV fixtures, and a tiny
fixture feather table this module writes itself) -- no MaleCNS raw data
required, matching `tests_python/test_compile.py`'s pattern.

Covers WP1's acceptance criteria (`.agents/plans/anatomical-activity-view/
01-positions-artifact.md`):
- `bodyIds` order equals the fixture graph's `biological_ids`;
- fallback order is soma -> tosoma -> none;
- coverage counts sum to the neuron count;
- two runs produce byte-identical output;
- an annotations sha256 mismatch exits non-zero.
"""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import binfmt  # noqa: E402
import compile as compiler  # noqa: E402
import positions  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
POSITIONS_SCRIPT = REPO_ROOT / "scripts" / "data" / "positions.py"

DEFAULT_METADATA_PARAMS = {
    "timestepSeconds": 1.0 / 30.0,
    "leakRate": 0.35,
    "rateMin": -2.0,
    "rateMax": 2.0,
    "inputClampMin": -1.0,
    "inputClampMax": 1.0,
    "globalGain": 0.5,
}


def _load_csv_fixture(nodes_name: str, edges_name: str):
    """Mirrors test_compile.py's `_load_csv_fixture`: a tiny nodes/edges CSV
    pair into the inputs `compile_graph` expects."""
    nodes_path = FIXTURES_DIR / nodes_name
    edges_path = FIXTURES_DIR / edges_name

    node_ids: list[int] = []
    signs: dict[int, int] = {}
    input_assignment: dict[int, tuple[int, float]] = {}
    output_assignment: dict[int, tuple[int, float]] = {}

    with open(nodes_path, newline="") as fh:
        for row in csv.DictReader(fh):
            body = int(row["body_id"])
            node_ids.append(body)
            signs[body] = int(row["sign"])
            if row["input_channel"].strip() != "":
                input_assignment[body] = (int(row["input_channel"]), float(row["input_weight"]))
            if row["output_population"].strip() != "":
                output_assignment[body] = (int(row["output_population"]), float(row["output_weight"]))

    edges = pd.read_csv(edges_path)
    return node_ids, signs, input_assignment, output_assignment, edges


def _build_fixture_graph() -> binfmt.GraphArrays:
    """`dup_nodes.csv`/`dup_edges.csv` (also used by test_compile.py)
    exercise all three roles: 2001/2002 sensory (channels 0/1), 2003 bridge
    (no channel or population), 2004/2005 descending (populations 0/1)."""
    node_ids, signs, input_assignment, output_assignment, edges = _load_csv_fixture(
        "dup_nodes.csv", "dup_edges.csv"
    )
    graph, _stats = compiler.compile_graph(
        node_ids=node_ids,
        edges=edges,
        signs=signs,
        input_assignment=input_assignment,
        output_assignment=output_assignment,
        input_channel_count=2,
        output_population_count=2,
        metadata_params=DEFAULT_METADATA_PARAMS,
    )
    return graph


def _write_graph_gzip(graph: binfmt.GraphArrays, path: Path) -> None:
    binary = binfmt.encode_graph_binary(graph)
    binfmt.write_gzip_deterministic(binary, path)


def _write_annotations_feather(path: Path, rows: "list[dict]") -> None:
    """`rows` is a list of {"bodyId": int, "somaLocation": list[int]|None,
    "tosomaLocation": list[int]|None}. Round-trips through an actual
    feather file (not an in-memory DataFrame) so this exercises the exact
    same list<item: int64>-column read path as the real MaleCNS export
    (verified directly: a missing list cell round-trips as Python `None`,
    not NaN)."""
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, path)


# Body IDs shared by the fixture graph (from dup_nodes.csv) and the
# annotations fixtures below.
SENSORY_A, SENSORY_B = 2001, 2002
BRIDGE = 2003
DESCENDING_A, DESCENDING_B = 2004, 2005


def _default_annotation_rows() -> "list[dict]":
    """Covers the full fallback matrix in one fixture:
    - SENSORY_A: soma only -> "soma"
    - SENSORY_B: tosoma only -> "tosoma"
    - BRIDGE: neither -> "none"
    - DESCENDING_A: both soma and tosoma present -> "soma" wins
    - DESCENDING_B: tosoma only -> "tosoma"
    Also includes an unrelated body (9999) not present in the graph, to
    prove extra annotation rows never affect the join.
    """
    return [
        {"bodyId": SENSORY_A, "somaLocation": [1, 2, 3], "tosomaLocation": None},
        {"bodyId": SENSORY_B, "somaLocation": None, "tosomaLocation": [4, 5, 6]},
        {"bodyId": BRIDGE, "somaLocation": None, "tosomaLocation": None},
        {"bodyId": DESCENDING_A, "somaLocation": [7, 8, 9], "tosomaLocation": [99, 99, 99]},
        {"bodyId": DESCENDING_B, "somaLocation": None, "tosomaLocation": [10, 11, 12]},
        {"bodyId": 9999, "somaLocation": [1, 1, 1], "tosomaLocation": None},
    ]


def test_body_ids_order_matches_graph_biological_ids():
    graph = _build_fixture_graph()
    # Reversed row order: the annotations table's own row order must have no
    # effect on the emitted order. If build_positions ever iterated the
    # annotations table instead of the graph's biological_ids, this would
    # catch it (an ascending-both-fixtures test would not).
    annotations = pd.DataFrame(list(reversed(_default_annotation_rows())))

    fields = positions.build_positions(graph, annotations)

    assert fields["bodyIds"] == [str(b) for b in graph.biological_ids.tolist()]
    # Sorted ascending, per compile_graph's node ordering.
    assert fields["bodyIds"] == ["2001", "2002", "2003", "2004", "2005"]


def test_role_derivation_matches_graph_channel_and_population_assignment():
    graph = _build_fixture_graph()
    annotations = pd.DataFrame(_default_annotation_rows())

    fields = positions.build_positions(graph, annotations)

    index_of = {body: i for i, body in enumerate(fields["bodyIds"])}
    assert fields["role"][index_of[str(SENSORY_A)]] == "sensory"
    assert fields["role"][index_of[str(SENSORY_B)]] == "sensory"
    assert fields["role"][index_of[str(BRIDGE)]] == "bridge"
    assert fields["role"][index_of[str(DESCENDING_A)]] == "descending"
    assert fields["role"][index_of[str(DESCENDING_B)]] == "descending"
    assert fields["roleCounts"] == {"sensory": 2, "bridge": 1, "descending": 2}


def test_fallback_order_is_soma_then_tosoma_then_none():
    graph = _build_fixture_graph()
    annotations = pd.DataFrame(_default_annotation_rows())

    fields = positions.build_positions(graph, annotations)

    index_of = {body: i for i, body in enumerate(fields["bodyIds"])}

    def at(body: int):
        i = index_of[str(body)]
        return fields["positionSource"][i], fields["xyz"][i]

    assert at(SENSORY_A) == ("soma", [1, 2, 3])
    assert at(SENSORY_B) == ("tosoma", [4, 5, 6])
    assert at(BRIDGE) == ("none", None)
    # Both soma and tosoma present -> soma wins, tosoma value is ignored.
    assert at(DESCENDING_A) == ("soma", [7, 8, 9])
    assert at(DESCENDING_B) == ("tosoma", [10, 11, 12])


def test_coverage_counts_sum_to_neuron_count():
    graph = _build_fixture_graph()
    annotations = pd.DataFrame(_default_annotation_rows())

    fields = positions.build_positions(graph, annotations)
    neuron_count = int(graph.metadata["neuronCount"])

    assert sum(fields["coverage"].values()) == neuron_count
    assert fields["coverage"] == {"soma": 2, "tosoma": 2, "none": 1}


def test_both_channel_and_population_assigned_raises():
    graph = _build_fixture_graph()
    # Corrupt the fixture graph so one neuron is illegally both sensory and
    # descending, exercising build_positions' own defensive check (this can
    # never happen from a real compiled artifact, which compile.py's
    # assign_channels partitions disjointly).
    graph.output_population_index[0] = 0  # index 0 (body 2001) is already sensory (channel 0)
    annotations = pd.DataFrame(_default_annotation_rows())

    with pytest.raises(RuntimeError, match="both an input channel and an output population"):
        positions.build_positions(graph, annotations)


def test_missing_position_column_raises_instead_of_all_none():
    """A table missing somaLocation/tosomaLocation entirely must not
    silently produce an all-"none" artifact (exit 0) -- see
    docs/data-provenance.md's "never guess silently" policy."""
    graph = _build_fixture_graph()
    rows = [{k: v for k, v in row.items() if k != "tosomaLocation"} for row in _default_annotation_rows()]
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="missing required column"):
        positions.build_positions(graph, annotations)


def test_graph_body_with_no_annotation_row_raises():
    """Every compiled neuron's bodyId comes from this same annotations
    table's traced subset (compile.py's select_subgraph), so a compiled
    body missing from the table entirely is a graph/annotations divergence,
    not a legitimate "no data" case -- it must raise, not silently become
    positionSource "none"."""
    graph = _build_fixture_graph()
    rows = [row for row in _default_annotation_rows() if row["bodyId"] != SENSORY_A]
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="no row in the annotations table"):
        positions.build_positions(graph, annotations)


def test_body_id_dtype_mismatch_raises():
    """A bodyId column read back as strings (a schema-drift scenario) joins
    against nothing, and must raise rather than silently emitting an
    all-"none" artifact."""
    graph = _build_fixture_graph()
    rows = [{**row, "bodyId": str(row["bodyId"])} for row in _default_annotation_rows()]
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="no row in the annotations table"):
        positions.build_positions(graph, annotations)


def test_malformed_location_cell_raises():
    """A 3-element list containing a null component, or a list of the wrong
    length, is a malformed cell -- not the same as a legitimately-absent
    (None) cell -- and must raise with the offending bodyId, not silently
    fall back to the next source."""
    graph = _build_fixture_graph()
    rows = _default_annotation_rows()
    rows[0] = {**rows[0], "somaLocation": [1, None, 3]}  # SENSORY_A
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match=r"bodyId 2001: somaLocation has a missing component"):
        positions.build_positions(graph, annotations)


def test_wrong_length_location_cell_raises():
    graph = _build_fixture_graph()
    rows = _default_annotation_rows()
    rows[0] = {**rows[0], "somaLocation": [1, 2]}  # SENSORY_A
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match=r"bodyId 2001: somaLocation has shape"):
        positions.build_positions(graph, annotations)


def test_annotations_sha_mismatch_raises_with_specific_message(tmp_path):
    """In-process counterpart to test_annotations_sha_mismatch_exits_non_zero
    below: pins the exact failure branch (not just "some error happened")
    via load_annotations_verified directly."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / positions.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())

    with pytest.raises(RuntimeError, match="does not match the pinned"):
        positions.load_annotations_verified(raw_dir, expected_sha256="0" * 64)


def test_two_runs_produce_byte_identical_positions_json(tmp_path):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / positions.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = positions._sha256_of_file(annotations_path)

    out_dir_a = tmp_path / "out-a"
    out_dir_b = tmp_path / "out-b"

    for out_dir in (out_dir_a, out_dir_b):
        annotations, source_sha256 = positions.load_annotations_verified(
            raw_dir, expected_sha256=expected_sha256
        )
        gzip_bytes = graph_path.read_bytes()
        graph_sha256 = binfmt.sha256_hex(gzip_bytes)
        decoded = positions.rewire.decode_graph_binary(gzip.decompress(gzip_bytes))
        fields = positions.build_positions(decoded, annotations)
        payload = positions.render_positions_document(
            source_sha256=source_sha256, graph_sha256=graph_sha256, fields=fields
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "malecns-arena-v1.positions.json").write_bytes(payload.encode("utf-8"))

    bytes_a = (out_dir_a / "malecns-arena-v1.positions.json").read_bytes()
    bytes_b = (out_dir_b / "malecns-arena-v1.positions.json").read_bytes()
    assert bytes_a == bytes_b
    # Deterministic formatting: sorted keys, fixed separators, trailing
    # newline -- checked as a full round-trip so this doesn't hard-code
    # which key happens to sort first (a later WP could add one that sorts
    # before "bodyIds").
    assert bytes_a.endswith(b"\n")
    text_a = bytes_a.decode("utf-8")
    canonical = json.dumps(json.loads(text_a), indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    assert text_a == canonical


def _write_manifest_and_ledger_fixtures(
    out_dir: Path, graph_path: Path, *, selection_counts: "dict | None" = None
) -> "tuple[Path, Path]":
    """A manifest/ledger pair minimal enough for main()'s cross-checks: the
    manifest's gzipSha256 must match `graph_path`'s actual sha256 (I3's
    --graph/manifest guard), and an optional ledger `selectionCounts` block
    exercises the roleCounts cross-check."""
    manifest_path = out_dir / "malecns-arena-v1.manifest.json"
    ledger_path = out_dir / "malecns-arena-v1.ledger.json"
    graph_gzip_sha256 = binfmt.sha256_hex(graph_path.read_bytes())
    manifest_path.write_text(
        json.dumps({"artifact": "malecns-arena-v1.bin.gz", "gzipSha256": graph_gzip_sha256}) + "\n"
    )
    ledger = {"artifact": "malecns-arena-v1.bin.gz"}
    if selection_counts is not None:
        ledger["selectionCounts"] = selection_counts
    ledger_path.write_text(json.dumps(ledger) + "\n")
    return manifest_path, ledger_path


def test_main_writes_manifest_and_ledger_entries(tmp_path, monkeypatch):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / positions.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = positions._sha256_of_file(annotations_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path, ledger_path = _write_manifest_and_ledger_fixtures(
        out_dir,
        graph_path,
        selection_counts={
            "sensorySelectedCount": 2,
            "bridgeSelectedCount": 1,
            "descendingSelectedCount": 2,
        },
    )

    # main() always verifies against download.py's real pinned sha256 for
    # the real filename; monkeypatch it to this fixture's hash for this
    # end-to-end test only.
    monkeypatch.setattr(positions, "_pinned_annotations_sha256", lambda: expected_sha256)

    exit_code = positions.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--graph",
            str(graph_path),
            "--out-dir",
            str(out_dir),
            "--manifest-path",
            str(manifest_path),
            "--ledger-path",
            str(ledger_path),
        ]
    )

    assert exit_code == 0

    positions_json_path = out_dir / "malecns-arena-v1.positions.json"
    assert positions_json_path.exists()
    doc = json.loads(positions_json_path.read_text())
    assert doc["version"] == 1
    assert doc["roleCounts"] == {"sensory": 2, "bridge": 1, "descending": 2}
    assert doc["coverage"] == {"soma": 2, "tosoma": 2, "none": 1}
    assert doc["units"] == positions.UNITS

    manifest = json.loads(manifest_path.read_text())
    assert manifest["positions"]["artifact"] == "malecns-arena-v1.positions.json"
    assert manifest["positions"]["sha256"]
    assert manifest["positions"]["coverage"] == {"soma": 2, "tosoma": 2, "none": 1}

    ledger = json.loads(ledger_path.read_text())
    assert ledger["positionsCoverage"] == {"soma": 2, "tosoma": 2, "none": 1}


def test_main_raises_when_graph_does_not_match_manifest(tmp_path, monkeypatch):
    """--graph must describe the same compiled artifact the manifest does;
    otherwise a positions entry could be attached to the wrong graph."""
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / positions.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = positions._sha256_of_file(annotations_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "malecns-arena-v1.manifest.json"
    ledger_path = out_dir / "malecns-arena-v1.ledger.json"
    # Deliberately wrong gzipSha256: does not match graph_path's real hash.
    manifest_path.write_text(json.dumps({"artifact": "malecns-arena-v1.bin.gz", "gzipSha256": "0" * 64}) + "\n")
    ledger_path.write_text(json.dumps({"artifact": "malecns-arena-v1.bin.gz"}) + "\n")

    monkeypatch.setattr(positions, "_pinned_annotations_sha256", lambda: expected_sha256)

    with pytest.raises(RuntimeError, match="does not match .*gzipSha256"):
        positions.main(
            [
                "--raw-dir",
                str(raw_dir),
                "--graph",
                str(graph_path),
                "--out-dir",
                str(out_dir),
                "--manifest-path",
                str(manifest_path),
                "--ledger-path",
                str(ledger_path),
            ]
        )
    # Nothing should have been written: the check runs before any write.
    assert not (out_dir / "malecns-arena-v1.positions.json").exists()


def test_main_raises_when_role_counts_disagree_with_ledger_selection_counts(tmp_path, monkeypatch):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / positions.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = positions._sha256_of_file(annotations_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path, ledger_path = _write_manifest_and_ledger_fixtures(
        out_dir,
        graph_path,
        # Deliberately wrong: the fixture graph actually has 2 sensory.
        selection_counts={
            "sensorySelectedCount": 999,
            "bridgeSelectedCount": 1,
            "descendingSelectedCount": 2,
        },
    )

    monkeypatch.setattr(positions, "_pinned_annotations_sha256", lambda: expected_sha256)

    with pytest.raises(RuntimeError, match="does not match .* selectionCounts"):
        positions.main(
            [
                "--raw-dir",
                str(raw_dir),
                "--graph",
                str(graph_path),
                "--out-dir",
                str(out_dir),
                "--manifest-path",
                str(manifest_path),
                "--ledger-path",
                str(ledger_path),
            ]
        )
    assert not (out_dir / "malecns-arena-v1.positions.json").exists()


def test_annotations_sha_mismatch_exits_non_zero(tmp_path):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / positions.ANNOTATIONS_FILENAME
    # Content does not matter here: this fixture file's sha256 will not
    # match download.py's real pinned hash for the real filename, and
    # main() (invoked as a subprocess, exactly as a real run would be) must
    # refuse to proceed rather than silently joining against unverified
    # data.
    _write_annotations_feather(annotations_path, _default_annotation_rows())

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(POSITIONS_SCRIPT),
            "--raw-dir",
            str(raw_dir),
            "--graph",
            str(graph_path),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    # main() prints a "graph sha256(gzip)=..." progress line to stdout
    # before it ever checks the annotations, so a bare "sha256" in output"
    # assertion would pass for any failure after the graph loads -- assert
    # on the mismatch-specific message instead, so this test actually
    # proves the mismatch branch (not some unrelated crash) is what stopped
    # the run.
    assert "does not match the pinned" in combined
    assert "refusing to join positions" in combined
    assert not (out_dir / "malecns-arena-v1.positions.json").exists()
