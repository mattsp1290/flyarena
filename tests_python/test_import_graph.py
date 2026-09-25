"""`graph_io.python_dependency_closure`'s real-Python-import-graph walker,
the structural fix for a thermo-fix-verification review finding (`reviews/
feat-g3an-null-explanation-thermo-fix-2026-09-25-544aae2/thermo-fix-
verification/01-critical-and-important.md`, finding 1): the old hand-
maintained `TRANSFER_SOURCE_FILENAMES`/`FEATURES_SOURCE_FILENAMES` flat
lists silently omitted `scripts/data/rewire.py`/`binfmt.py`, even though
`graph_io.load_verified_graph` calls `rewire.decode_graph_binary` on every
graph either producer loads -- the actual bytes-to-arrays decode feeding
every number in `transfer.json`/`features.json`.

Covers (1) against the real repository, `transfer.py`'s/`features.py`'s
walked closures genuinely include those previously-missing files, and (2)
against small, hand-built fixture module trees, the walker's module-
resolution rules and `source_identity_sha256`'s content-sensitivity (the
mutation-test requirement)."""

from __future__ import annotations

from pathlib import Path

import features
import graph_io
import transfer

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Real repository: transfer.py / features.py closures
# ---------------------------------------------------------------------------


def test_transfer_py_closure_includes_rewire_and_binfmt():
    dependencies = graph_io.python_dependency_closure(transfer.TRANSFER_ENTRY, REPO_ROOT, transfer.TRANSFER_SEARCH_DIRS)

    assert "scripts/analysis/transfer.py" in dependencies  # entry file itself
    assert "scripts/analysis/graph_io.py" in dependencies
    assert "scripts/analysis/env_guard.py" in dependencies
    # Previously missing (thermo-fix-verification finding 1): the actual
    # graph binary decode `graph_io.load_verified_graph` delegates to.
    assert "scripts/data/rewire.py" in dependencies
    assert "scripts/data/binfmt.py" in dependencies
    assert "scripts/data/fsutil.py" in dependencies
    assert dependencies == sorted(set(dependencies))


def test_features_py_closure_includes_rewire_and_binfmt():
    dependencies = graph_io.python_dependency_closure(features.FEATURES_ENTRY, REPO_ROOT, features.FEATURES_SEARCH_DIRS)

    assert "scripts/analysis/features.py" in dependencies
    assert "scripts/analysis/graph_io.py" in dependencies
    assert "scripts/analysis/env_guard.py" in dependencies
    assert "scripts/data/rewire.py" in dependencies
    assert "scripts/data/binfmt.py" in dependencies
    assert "scripts/data/fsutil.py" in dependencies


def test_transfer_and_features_producer_blocks_record_their_dependency_list():
    transfer_block = transfer.transfer_producer()
    features_block = features.features_producer()

    assert set(transfer_block["dependencies"]) == set(
        graph_io.python_dependency_closure(transfer.TRANSFER_ENTRY, REPO_ROOT, transfer.TRANSFER_SEARCH_DIRS)
    )
    assert set(features_block["dependencies"]) == set(
        graph_io.python_dependency_closure(features.FEATURES_ENTRY, REPO_ROOT, features.FEATURES_SEARCH_DIRS)
    )
    assert transfer_block["sourceSha256"] == graph_io.source_identity_sha256(REPO_ROOT, transfer_block["dependencies"])
    assert features_block["sourceSha256"] == graph_io.source_identity_sha256(
        REPO_ROOT, features_block["dependencies"]
    )


# ---------------------------------------------------------------------------
# Synthetic fixture trees: resolution rules
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_resolves_bare_module_import_within_entry_directory(tmp_path):
    _write(tmp_path / "a.py", "import b\n\nb.value\n")
    _write(tmp_path / "b.py", "value = 1\n")

    deps = graph_io.python_dependency_closure(tmp_path / "a.py", tmp_path, (tmp_path,))
    assert deps == ["a.py", "b.py"]


def test_resolves_from_import_across_a_second_search_dir(tmp_path):
    analysis_dir = tmp_path / "analysis"
    data_dir = tmp_path / "data"
    _write(analysis_dir / "entry.py", "from helper import thing\n\nthing\n")
    _write(data_dir / "helper.py", "thing = 1\n")

    deps = graph_io.python_dependency_closure(analysis_dir / "entry.py", tmp_path, (analysis_dir, data_dir))
    assert sorted(deps) == ["analysis/entry.py", "data/helper.py"]


def test_stdlib_and_third_party_imports_are_not_walked_into(tmp_path):
    _write(tmp_path / "a.py", "import os\nimport numpy as np\nfrom pathlib import Path\n")

    deps = graph_io.python_dependency_closure(tmp_path / "a.py", tmp_path, (tmp_path,))
    assert deps == ["a.py"]


def test_terminates_on_an_import_cycle(tmp_path):
    _write(tmp_path / "a.py", "import b\n")
    _write(tmp_path / "b.py", "import a\n")

    deps = graph_io.python_dependency_closure(tmp_path / "a.py", tmp_path, (tmp_path,))
    assert sorted(deps) == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# source_identity_sha256: content-sensitivity (the mutation-test requirement)
# ---------------------------------------------------------------------------


def test_source_identity_sha256_changes_when_a_dependency_file_is_mutated(tmp_path):
    _write(tmp_path / "a.py", "value = 1\n")
    _write(tmp_path / "rewire.py", "def decode_graph_binary(b):\n    return b\n")
    paths = ["a.py", "rewire.py"]

    before = graph_io.source_identity_sha256(tmp_path, paths)
    _write(tmp_path / "rewire.py", "def decode_graph_binary(b):\n    return b[::-1]\n")  # simulates a rewire.py edit
    after = graph_io.source_identity_sha256(tmp_path, paths)

    assert before != after


def test_source_identity_sha256_is_path_order_independent(tmp_path):
    _write(tmp_path / "a.py", "value = 1\n")
    _write(tmp_path / "b.py", "value = 2\n")

    forward = graph_io.source_identity_sha256(tmp_path, ["a.py", "b.py"])
    reverse = graph_io.source_identity_sha256(tmp_path, ["b.py", "a.py"])
    assert forward == reverse


def test_source_identity_sha256_distinguishes_same_named_files_in_different_dirs(tmp_path):
    _write(tmp_path / "dir1" / "x.py", "value = 1\n")
    _write(tmp_path / "dir2" / "x.py", "value = 1\n")  # identical bytes, different path

    sha1 = graph_io.source_identity_sha256(tmp_path, ["dir1/x.py"])
    sha2 = graph_io.source_identity_sha256(tmp_path, ["dir2/x.py"])
    assert sha1 != sha2
