"""Tests for `scripts/data/descending_types.py`, driven entirely from tiny
fixtures (a fixture graph built from the existing compiler CSV fixtures, and
a tiny fixture feather table this module writes itself) -- no MaleCNS raw
data required, matching `tests_python/test_positions.py`'s pattern.

Covers WP1c's acceptance criteria
(`.agents/plans/readout-attribution/01-archive-and-types.md`):
- neuron order matches the graph's ascending descending-neuron index order;
- population comes from the graph's own outputPopulationIndex, never the
  annotations table;
- a missing per-neuron annotation value is recorded as `None`, never
  inferred or filled in;
- two runs produce byte-identical output;
- an annotations sha256 mismatch is refused;
- a descending neuron's bodyId missing from the annotations table entirely
  is refused (graph/annotations divergence), unlike an ordinary missing
  cell value.
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
import descending_types  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DESCENDING_TYPES_SCRIPT = REPO_ROOT / "scripts" / "data" / "descending_types.py"

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
    """Mirrors test_compile.py's/test_positions.py's `_load_csv_fixture`."""
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
    """`dup_nodes.csv`/`dup_edges.csv` (shared with test_compile.py and
    test_positions.py) exercise all three roles: 2001/2002 sensory
    (channels 0/1), 2003 bridge (no channel or population), 2004/2005
    descending (populations 0/1 respectively)."""
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
    """Round-trips through an actual feather file (not an in-memory
    DataFrame), matching test_positions.py's rationale: this exercises the
    exact same nullable-column read path the real MaleCNS export uses."""
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, path)


# Body IDs shared by the fixture graph (from dup_nodes.csv) and the
# annotations fixtures below.
SENSORY_A, SENSORY_B = 2001, 2002
BRIDGE = 2003
DESCENDING_A, DESCENDING_B = 2004, 2005


def _default_annotation_rows() -> "list[dict]":
    """DESCENDING_A has every field populated; DESCENDING_B is missing
    `class` and `group` (a real annotation gap, matching the real
    MaleCNS table where `class` is null for every descending neuron) --
    both must round-trip as `None`, never inferred. Also includes rows for
    the non-descending bodies (required since `build_descending_types` only
    looks up descending bodies, but a realistic table has every compiled
    body) and one unrelated body (9999) not present in the graph, to prove
    extra annotation rows never affect the join."""
    return [
        {
            "bodyId": SENSORY_A,
            "type": "sensoryType",
            "class": "sensoryClass",
            "instance": "sensoryInstance",
            "group": 1.0,
            "somaSide": "L",
        },
        {
            "bodyId": SENSORY_B,
            "type": "sensoryType",
            "class": "sensoryClass",
            "instance": "sensoryInstance",
            "group": 1.0,
            "somaSide": "R",
        },
        {
            "bodyId": BRIDGE,
            "type": "bridgeType",
            "class": "bridgeClass",
            "instance": "bridgeInstance",
            "group": 3.0,
            "somaSide": "L",
        },
        {
            "bodyId": DESCENDING_A,
            "type": "DNa01",
            "class": "descendingClass",
            "instance": "DNa01_L",
            "group": 42.0,
            "somaSide": "L",
        },
        {
            "bodyId": DESCENDING_B,
            "type": "DNb02",
            "class": None,
            "instance": "DNb02_R",
            "group": None,
            "somaSide": "R",
        },
        {
            "bodyId": 9999,
            "type": "unrelatedType",
            "class": "unrelatedClass",
            "instance": "unrelatedInstance",
            "group": 9.0,
            "somaSide": "L",
        },
    ]


def test_neuron_order_and_population_match_graph_output_population_index():
    graph = _build_fixture_graph()
    # Reversed row order: the annotations table's own row order must have no
    # effect on the emitted order.
    annotations = pd.DataFrame(list(reversed(_default_annotation_rows())))

    neurons = descending_types.build_descending_types(graph, annotations)

    body_ids = [str(b) for b in graph.biological_ids.tolist()]
    index_of = {body: i for i, body in enumerate(body_ids)}

    assert [n["bodyId"] for n in neurons] == [str(DESCENDING_A), str(DESCENDING_B)]
    assert [n["index"] for n in neurons] == [index_of[str(DESCENDING_A)], index_of[str(DESCENDING_B)]]
    assert [n["population"] for n in neurons] == [0, 1]


def test_fields_copied_exactly_and_missing_values_are_none():
    graph = _build_fixture_graph()
    annotations = pd.DataFrame(_default_annotation_rows())

    neurons = descending_types.build_descending_types(graph, annotations)
    by_body = {n["bodyId"]: n for n in neurons}

    a = by_body[str(DESCENDING_A)]
    assert a["type"] == "DNa01"
    assert a["class"] == "descendingClass"
    assert a["instance"] == "DNa01_L"
    assert a["group"] == 42
    assert a["somaSide"] == "L"

    b = by_body[str(DESCENDING_B)]
    assert b["type"] == "DNb02"
    assert b["class"] is None  # never inferred/filled in
    assert b["instance"] == "DNb02_R"
    assert b["group"] is None
    assert b["somaSide"] == "R"


def test_group_is_emitted_as_a_python_int_not_a_float():
    graph = _build_fixture_graph()
    annotations = pd.DataFrame(_default_annotation_rows())

    neurons = descending_types.build_descending_types(graph, annotations)
    a = next(n for n in neurons if n["bodyId"] == str(DESCENDING_A))
    assert a["group"] == 42
    assert isinstance(a["group"], int)


def test_non_integral_group_value_raises():
    graph = _build_fixture_graph()
    rows = _default_annotation_rows()
    for row in rows:
        if row["bodyId"] == DESCENDING_A:
            row["group"] = 42.5
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match=r"bodyId 2004: group has a non-integral value"):
        descending_types.build_descending_types(graph, annotations)


def test_missing_required_column_raises_instead_of_all_none():
    """A table missing a required column entirely must not silently
    produce an all-null artifact (exit 0) -- see
    docs/data-provenance.md's "never guess silently" policy, applied here
    the same way `test_positions.py` applies it to soma columns."""
    graph = _build_fixture_graph()
    rows = [{k: v for k, v in row.items() if k != "instance"} for row in _default_annotation_rows()]
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="missing required column"):
        descending_types.build_descending_types(graph, annotations)


def test_descending_body_with_no_annotation_row_raises():
    """Every compiled neuron's bodyId comes from this same annotations
    table's traced subset, so a compiled descending body missing from the
    table entirely is a graph/annotations divergence, not a legitimate
    "no data" case -- it must raise, not silently emit an all-null row."""
    graph = _build_fixture_graph()
    rows = [row for row in _default_annotation_rows() if row["bodyId"] != DESCENDING_A]
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="no row in the annotations table"):
        descending_types.build_descending_types(graph, annotations)


def test_duplicate_body_id_raises():
    graph = _build_fixture_graph()
    rows = _default_annotation_rows() + [_default_annotation_rows()[3]]  # duplicate DESCENDING_A
    annotations = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="duplicate bodyId"):
        descending_types.build_descending_types(graph, annotations)


def test_annotations_sha_mismatch_raises_with_specific_message(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / descending_types.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())

    with pytest.raises(RuntimeError, match="does not match the pinned"):
        descending_types.load_annotations_verified(raw_dir, expected_sha256="0" * 64)


def test_two_runs_produce_byte_identical_descending_types_json(tmp_path):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / descending_types.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = descending_types._sha256_of_file(annotations_path)

    out_dir_a = tmp_path / "out-a"
    out_dir_b = tmp_path / "out-b"

    for out_dir in (out_dir_a, out_dir_b):
        annotations, source_sha256 = descending_types.load_annotations_verified(
            raw_dir, expected_sha256=expected_sha256
        )
        gzip_bytes = graph_path.read_bytes()
        graph_sha256 = binfmt.sha256_hex(gzip_bytes)
        decoded = descending_types.rewire.decode_graph_binary(gzip.decompress(gzip_bytes))
        neurons = descending_types.build_descending_types(decoded, annotations)
        payload = descending_types.render_descending_types_document(
            source_sha256=source_sha256, graph_sha256=graph_sha256, neurons=neurons
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "descending-types-v1.json").write_bytes(payload.encode("utf-8"))

    bytes_a = (out_dir_a / "descending-types-v1.json").read_bytes()
    bytes_b = (out_dir_b / "descending-types-v1.json").read_bytes()
    assert bytes_a == bytes_b
    assert bytes_a.endswith(b"\n")
    text_a = bytes_a.decode("utf-8")
    canonical = json.dumps(json.loads(text_a), indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    assert text_a == canonical


def test_main_writes_manifest_entry_touching_only_its_own_key(tmp_path, monkeypatch):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / descending_types.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = descending_types._sha256_of_file(annotations_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "malecns-arena-v1.manifest.json"
    graph_gzip_sha256 = binfmt.sha256_hex(graph_path.read_bytes())
    original_manifest = {
        "artifact": "malecns-arena-v1.bin.gz",
        "gzipSha256": graph_gzip_sha256,
        "binarySha256": "unrelated-binary-sha",
        "compilerSourceSha256": "unrelated-compiler-sha",
    }
    manifest_path.write_text(json.dumps(original_manifest, sort_keys=True) + "\n")

    monkeypatch.setattr(descending_types, "_pinned_annotations_sha256", lambda: expected_sha256)

    exit_code = descending_types.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--graph",
            str(graph_path),
            "--out-dir",
            str(out_dir),
            "--manifest-path",
            str(manifest_path),
        ]
    )

    assert exit_code == 0

    out_path = out_dir / "descending-types-v1.json"
    assert out_path.exists()
    doc = json.loads(out_path.read_text())
    assert doc["version"] == 1
    assert len(doc["neurons"]) == 2

    manifest = json.loads(manifest_path.read_text())
    assert manifest["descendingTypes"]["artifact"] == "descending-types-v1.json"
    assert manifest["descendingTypes"]["sha256"]
    assert manifest["descendingTypes"]["neuronCount"] == 2
    # Every other pre-existing key is untouched -- adding this key must
    # never perturb compilerSourceSha256/binarySha256/gzipSha256.
    assert manifest["binarySha256"] == "unrelated-binary-sha"
    assert manifest["compilerSourceSha256"] == "unrelated-compiler-sha"
    assert manifest["gzipSha256"] == graph_gzip_sha256


def test_main_skips_manifest_update_when_manifest_path_does_not_exist(tmp_path, monkeypatch):
    """A manifest-less environment (e.g. a hand-built fixture, or main()
    invoked before compile.py has ever run) must still write the sidecar
    itself and exit 0 -- it only skips the manifest merge, with a `[warn]`
    message, rather than crashing (mirrors positions.py's same tolerance for
    a missing manifest/ledger)."""
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / descending_types.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = descending_types._sha256_of_file(annotations_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Deliberately does not create a manifest.json at all.
    manifest_path = out_dir / "malecns-arena-v1.manifest.json"
    assert not manifest_path.exists()

    monkeypatch.setattr(descending_types, "_pinned_annotations_sha256", lambda: expected_sha256)

    exit_code = descending_types.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--graph",
            str(graph_path),
            "--out-dir",
            str(out_dir),
            "--manifest-path",
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    out_path = out_dir / "descending-types-v1.json"
    assert out_path.exists()
    doc = json.loads(out_path.read_text())
    assert len(doc["neurons"]) == 2
    # No manifest was created as a side effect of main() running.
    assert not manifest_path.exists()


def test_main_raises_when_graph_does_not_match_manifest(tmp_path, monkeypatch):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / descending_types.ANNOTATIONS_FILENAME
    _write_annotations_feather(annotations_path, _default_annotation_rows())
    expected_sha256 = descending_types._sha256_of_file(annotations_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "malecns-arena-v1.manifest.json"
    # Deliberately wrong gzipSha256: does not match graph_path's real hash.
    manifest_path.write_text(json.dumps({"artifact": "malecns-arena-v1.bin.gz", "gzipSha256": "0" * 64}) + "\n")

    monkeypatch.setattr(descending_types, "_pinned_annotations_sha256", lambda: expected_sha256)

    with pytest.raises(RuntimeError, match="does not match .*gzipSha256"):
        descending_types.main(
            [
                "--raw-dir",
                str(raw_dir),
                "--graph",
                str(graph_path),
                "--out-dir",
                str(out_dir),
                "--manifest-path",
                str(manifest_path),
            ]
        )
    assert not (out_dir / "descending-types-v1.json").exists()


def test_annotations_sha_mismatch_exits_non_zero(tmp_path):
    graph = _build_fixture_graph()
    graph_path = tmp_path / "malecns-arena-v1.bin.gz"
    _write_graph_gzip(graph, graph_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    annotations_path = raw_dir / descending_types.ANNOTATIONS_FILENAME
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
            str(DESCENDING_TYPES_SCRIPT),
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
    assert "does not match the pinned" in combined
    assert "refusing to join cell types" in combined
    assert not (out_dir / "descending-types-v1.json").exists()
