"""WP1 of `.agents/plans/selection-robustness`: `compile.py`'s predeclared
alternative subgraph selections (`SELECTIONS`), the `--selection`/
`--artifact-name` CLI surface, and the refusal rule that keeps a variant
compile from ever landing in `public/data` or under the default artifact
name.

Everything here runs against small, purely synthetic `annotations`/`weights`
tables built in this file -- no MaleCNS raw data required, matching
`test_compile.py`'s "no download needed" design. The real, pinned-data
default-compile byte-identity check (Gate 1 of
`.agents/plans/selection-robustness/01-compiler-variants.md`) is a separate,
manual procedure run against `data/raw/`, not part of this suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import compile as compiler  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixture: one sensory neuron, one descending neuron, and a large
# pool of candidate bridge neurons all tied at identical degree (so
# `bridge_mode="degree-rank"`'s output is exactly "smallest N body ids",
# trivially checkable, and large enough -- comfortably above 1600 -- that
# `larger`/`smaller`/`default` bridge targets each produce a distinct,
# non-degenerately-capped count).
# ---------------------------------------------------------------------------

SENSORY_BODY = 1
DESCENDING_BODY = 2
BRIDGE_POOL_SIZE = 2000
BRIDGE_BODY_START = 100  # bridge candidates: 100..100+BRIDGE_POOL_SIZE-1


def _selection_fixture():
    bridge_bodies = list(range(BRIDGE_BODY_START, BRIDGE_BODY_START + BRIDGE_POOL_SIZE))

    annotations = pd.DataFrame(
        {
            "bodyId": [SENSORY_BODY, DESCENDING_BODY] + bridge_bodies,
            "status": ["Traced"] * (2 + BRIDGE_POOL_SIZE),
            "superclass": ["vnc_sensory", "descending_neuron"] + ["other"] * BRIDGE_POOL_SIZE,
            "class": ["mechanosensory_tactile", None] + [None] * BRIDGE_POOL_SIZE,
        }
    )
    # Every bridge candidate is a direct postsynaptic partner of the sensory
    # neuron and a direct presynaptic partner of the descending neuron (the
    # 1-hop-each-direction bridge definition), all at identical weight so
    # `_rank_by_degree` ties everything and falls back to its documented
    # ascending-body-id tie-break -- making "smallest N" a checkable ground
    # truth for `bridge_mode="degree-rank"`.
    weights = pd.DataFrame(
        {
            "body_pre": [SENSORY_BODY] * BRIDGE_POOL_SIZE + bridge_bodies,
            "body_post": bridge_bodies + [DESCENDING_BODY] * BRIDGE_POOL_SIZE,
            "weight": [5.0] * (2 * BRIDGE_POOL_SIZE),
        }
    )
    return annotations, weights


# ---------------------------------------------------------------------------
# select_subgraph: SELECTIONS["default"] matches the no-selection-argument
# call (backward compatibility for the unchanged default path).
# ---------------------------------------------------------------------------


def test_default_selection_matches_no_selection_argument():
    annotations, weights = _selection_fixture()
    explicit = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["default"])
    implicit = compiler.select_subgraph(annotations, weights)
    assert explicit == implicit


# ---------------------------------------------------------------------------
# larger / smaller: bridge_target threading, degree-rank mode unchanged.
# ---------------------------------------------------------------------------


def test_larger_and_smaller_give_expected_bridge_counts():
    annotations, weights = _selection_fixture()

    default_selection = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["default"])
    larger_selection = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["larger"])
    smaller_selection = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["smaller"])

    assert default_selection["counts"]["bridgeSelectedCount"] == compiler.BRIDGE_TARGET  # 800
    assert larger_selection["counts"]["bridgeSelectedCount"] == 1600
    assert smaller_selection["counts"]["bridgeSelectedCount"] == 400

    # All three are ties broken by ascending body id -> "smallest N".
    expected_smallest = lambda n: list(range(BRIDGE_BODY_START, BRIDGE_BODY_START + n))  # noqa: E731
    assert default_selection["bridge_ids"] == expected_smallest(compiler.BRIDGE_TARGET)
    assert larger_selection["bridge_ids"] == expected_smallest(1600)
    assert smaller_selection["bridge_ids"] == expected_smallest(400)

    # bridgeCandidateCount (the pool before truncation) is identical across
    # all three -- only the cutoff differs.
    assert (
        default_selection["counts"]["bridgeCandidateCount"]
        == larger_selection["counts"]["bridgeCandidateCount"]
        == smaller_selection["counts"]["bridgeCandidateCount"]
        == BRIDGE_POOL_SIZE
    )


# ---------------------------------------------------------------------------
# random-bridge: deterministic per seed, differs from the degree-rank
# default.
# ---------------------------------------------------------------------------


def test_random_bridge_is_deterministic_and_differs_from_default():
    annotations, weights = _selection_fixture()

    default_selection = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["default"])
    random_a = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["random-bridge"])
    random_b = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["random-bridge"])

    # Same seed -> byte-for-byte identical selection.
    assert random_a["bridge_ids"] == random_b["bridge_ids"]
    assert random_a["node_ids"] == random_b["node_ids"]

    # A seeded uniform sample of 800 out of 2000 sorted candidates is not
    # the "smallest 800 ids" the degree-rank default would pick.
    assert random_a["bridge_ids"] != default_selection["bridge_ids"]
    assert len(random_a["bridge_ids"]) == compiler.BRIDGE_TARGET
    assert set(random_a["bridge_ids"]).issubset(set(range(BRIDGE_BODY_START, BRIDGE_BODY_START + BRIDGE_POOL_SIZE)))


def test_random_bridge_different_seed_gives_different_selection():
    annotations, weights = _selection_fixture()
    seeded_20260927 = compiler.select_subgraph(annotations, weights, selection=compiler.SELECTIONS["random-bridge"])
    seeded_other = compiler.select_subgraph(
        annotations, weights, selection={"bridge_mode": "seeded-uniform", "seed": 1}
    )
    assert seeded_20260927["bridge_ids"] != seeded_other["bridge_ids"]


def test_select_subgraph_rejects_unknown_bridge_mode():
    annotations, weights = _selection_fixture()
    with pytest.raises(ValueError, match="unknown bridge_mode"):
        compiler.select_subgraph(annotations, weights, selection={"bridge_mode": "not-a-real-mode"})


def test_random_bridge_target_larger_than_pool_returns_every_candidate():
    # sample_size = min(bridge_target, len(candidates)); this exercises the
    # min() branch instead of always leaving it implicitly true at the
    # default 800-of-2000 target.
    annotations, weights = _selection_fixture()
    oversized = compiler.select_subgraph(
        annotations,
        weights,
        selection={"bridge_mode": "seeded-uniform", "seed": 20260927, "bridge_target": BRIDGE_POOL_SIZE + 500},
    )
    assert oversized["bridge_ids"] == list(range(BRIDGE_BODY_START, BRIDGE_BODY_START + BRIDGE_POOL_SIZE))
    assert oversized["counts"]["bridgeSelectedCount"] == BRIDGE_POOL_SIZE


# ---------------------------------------------------------------------------
# alt-sensory-mapping: keeps the node set, changes only the channel
# assignment (input_channel_index); descending/output assignment unchanged.
# ---------------------------------------------------------------------------


def test_alt_sensory_mapping_keeps_node_set_and_changes_only_input_channels():
    sensory_ids = list(range(1, 17))  # 16 sensory candidates, 8 channels -> 2 each
    descending_ids = [1001, 1002, 1003, 1004]  # 4 descending, 3 populations

    default_input, default_output = compiler.assign_channels(
        sensory_ids, descending_ids, selection=compiler.SELECTIONS["default"]
    )
    alt_input, alt_output = compiler.assign_channels(
        sensory_ids, descending_ids, selection=compiler.SELECTIONS["alt-sensory-mapping"]
    )

    # Same node set (keys) on both sides.
    assert set(default_input.keys()) == set(alt_input.keys()) == set(sensory_ids)

    # Descending/output assignment is untouched by channel_mode.
    assert default_output == alt_output

    # At least one body's assigned channel actually changes under the
    # seeded permutation (otherwise the permutation would have to be the
    # identity, astronomically unlikely for 16! orderings and disallowed by
    # the next determinism/difference assertions anyway).
    assert default_input != alt_input
    changed_channels = {body for body in sensory_ids if default_input[body][0] != alt_input[body][0]}
    assert len(changed_channels) > 0

    # Every input weight is untouched (only the channel index moves).
    for body in sensory_ids:
        assert alt_input[body][1] == default_input[body][1]


def test_alt_sensory_mapping_is_deterministic_per_seed():
    sensory_ids = list(range(1, 17))
    descending_ids = [1001, 1002, 1003, 1004]

    alt_a, _ = compiler.assign_channels(sensory_ids, descending_ids, selection=compiler.SELECTIONS["alt-sensory-mapping"])
    alt_b, _ = compiler.assign_channels(sensory_ids, descending_ids, selection=compiler.SELECTIONS["alt-sensory-mapping"])
    assert alt_a == alt_b


def test_assign_channels_rejects_unknown_channel_mode():
    with pytest.raises(ValueError, match="unknown channel_mode"):
        compiler.assign_channels([1, 2, 3], [100], selection={"channel_mode": "not-a-real-mode"})


# ---------------------------------------------------------------------------
# build_manifest_and_ledger: artifact_name / selection_id / selection_params
# threading (so a variant's manifest/ledger `artifact` field and new
# `selection` key match its actual file names and predeclared policy).
# ---------------------------------------------------------------------------


def _minimal_graph_and_stats():
    # Smallest possible valid graph: one node, no edges, no channel/
    # population assignment (compile_graph tolerates empty assignments).
    graph, stats = compiler.compile_graph(
        node_ids=[42],
        edges=pd.DataFrame({"pre": [], "post": [], "weight": []}),
        signs={42: 1},
        input_assignment={},
        output_assignment={},
        input_channel_count=1,
        output_population_count=1,
        metadata_params={
            "timestepSeconds": 1.0 / 30.0,
            "leakRate": 0.35,
            "rateMin": -2.0,
            "rateMax": 2.0,
            "inputClampMin": -1.0,
            "inputClampMax": 1.0,
            "globalGain": 0.5,
        },
    )
    return graph, stats


def _build_args(**overrides):
    graph, stats = _minimal_graph_and_stats()
    args = dict(
        graph=graph,
        stats=stats,
        selection={"counts": {"finalNodeCount": 1}},
        unknown_by_label={},
        binary_sha256="deadbeef",
        binary_gzip_sha256="cafef00d",
        binary_gzip_size=1,
        binary_size=1,
        compiler_source_sha256_value="abc123",
    )
    args.update(overrides)
    return args


def test_build_manifest_and_ledger_defaults_match_legacy_behavior():
    manifest, ledger = compiler.build_manifest_and_ledger(**_build_args())
    assert manifest["artifact"] == f"{compiler.ARTIFACT_NAME}.bin.gz"
    assert ledger["artifact"] == f"{compiler.ARTIFACT_NAME}.bin.gz"
    assert manifest["selection"] == {"id": "default", "params": {}}
    assert ledger["selection"] == {"id": "default", "params": {}}


def test_build_manifest_and_ledger_threads_artifact_name_and_selection():
    manifest, ledger = compiler.build_manifest_and_ledger(
        **_build_args(
            artifact_name="malecns-arena-random-bridge",
            selection_id="random-bridge",
            selection_params=compiler.SELECTIONS["random-bridge"],
        )
    )
    assert manifest["artifact"] == "malecns-arena-random-bridge.bin.gz"
    assert ledger["artifact"] == "malecns-arena-random-bridge.bin.gz"
    assert manifest["selection"] == {
        "id": "random-bridge",
        "params": {"bridge_mode": "seeded-uniform", "seed": 20260927},
    }
    assert ledger["selection"] == manifest["selection"]


# ---------------------------------------------------------------------------
# Refusal rule: a variant into public/data, or under the default artifact
# name, is refused before any raw data is loaded or any file is written.
# ---------------------------------------------------------------------------


def test_refuse_helper_allows_default_selection_into_public_data():
    assert compiler._refuse_unsafe_variant_target("default", compiler.ARTIFACT_NAME, compiler.PUBLIC_DATA_DIR) is None


def test_refuse_helper_allows_safe_variant_target(tmp_path):
    assert (
        compiler._refuse_unsafe_variant_target("random-bridge", "malecns-arena-random-bridge", tmp_path) is None
    )


def test_refuse_helper_rejects_default_artifact_name_for_a_variant(tmp_path):
    message = compiler._refuse_unsafe_variant_target("random-bridge", compiler.ARTIFACT_NAME, tmp_path)
    assert message is not None
    assert "artifact-name" in message


@pytest.mark.parametrize(
    "out_dir_suffix",
    ["", "subdir", "nested/deeper"],
)
def test_refuse_helper_rejects_out_dir_inside_public_data(out_dir_suffix):
    out_dir = compiler.PUBLIC_DATA_DIR if not out_dir_suffix else compiler.PUBLIC_DATA_DIR / out_dir_suffix
    message = compiler._refuse_unsafe_variant_target("random-bridge", "malecns-arena-random-bridge", out_dir)
    assert message is not None
    assert "public/data" in message or "public" in message.lower()


def test_refuse_helper_allows_safe_out_dir_that_merely_shares_a_prefix_with_public_data(tmp_path):
    # public/data2 is a sibling of public/data, not inside it -- a
    # string-prefix check would wrongly flag this; the real check must use
    # path-component containment.
    sibling = compiler.PUBLIC_DATA_DIR.parent / (compiler.PUBLIC_DATA_DIR.name + "2")
    assert compiler._refuse_unsafe_variant_target("random-bridge", "malecns-arena-random-bridge", sibling) is None


@pytest.mark.parametrize(
    "unsafe_artifact_name",
    [
        "public/data/malecns-arena-v1",  # dual-review finding: a slash lets artifact_name smuggle
        "../malecns-arena-v1",  # in extra directory components the out-dir check alone misses
        "..",
        ".",
        "",
    ],
)
def test_refuse_helper_rejects_artifact_name_that_is_not_a_plain_filename(tmp_path, unsafe_artifact_name):
    message = compiler._refuse_unsafe_variant_target("random-bridge", unsafe_artifact_name, tmp_path)
    assert message is not None


def test_refuse_helper_closes_the_path_separator_bypass_into_public_data():
    """Regression test for the dual-review Critical finding: an
    `--artifact-name` embedding `public/data/...` combined with an `--out-dir`
    that does *not* itself resolve inside `public/data` used to slip past the
    refusal, because only `out_dir` (not the real `out_dir / artifact_name`
    write path) was checked. Must now be refused."""
    message = compiler._refuse_unsafe_variant_target(
        "larger", "public/data/malecns-arena-v1", compiler.REPO_ROOT
    )
    assert message is not None
    real_write_target = (compiler.REPO_ROOT / "public/data/malecns-arena-v1").resolve()
    assert real_write_target == (compiler.PUBLIC_DATA_DIR / "malecns-arena-v1").resolve()  # confirms the exploit shape


# ---------------------------------------------------------------------------
# Regression tests for the thermo-methodology dual-review Critical finding:
# `_refuse_unsafe_variant_target`'s prior `if selection_id == "default": return
# None` short-circuit skipped the plain-filename and out-dir checks entirely
# for the default selection -- which is also what `--selection`'s argparse
# default gives when the flag is omitted. The reviewer confirmed
# `--artifact-name /tmp/absolute-evil` and `--artifact-name ../escape` both
# reached `load_annotations()` under `--selection default`. The only
# permitted way to write into `public/data` is now the exact canonical
# triple: selection "default" + `ARTIFACT_NAME` + `PUBLIC_DATA_DIR`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "evil_artifact_name",
    ["/tmp/absolute-evil", "../escape"],
)
def test_refuse_helper_rejects_absolute_or_traversal_artifact_name_for_default_selection(evil_artifact_name):
    # Guard-level reproduction of the reviewer's two bypass argv lists,
    # against the exact canonical (selection=default, out_dir=public/data)
    # combination that used to short-circuit past all validation.
    message = compiler._refuse_unsafe_variant_target("default", evil_artifact_name, compiler.PUBLIC_DATA_DIR)
    assert message is not None


def test_refuse_helper_rejects_default_selection_with_non_canonical_artifact_name_into_public_data():
    # default selection + public/data out-dir is only safe with the exact
    # default artifact name; any other name must still be refused.
    message = compiler._refuse_unsafe_variant_target("default", "malecns-arena-v1-fake", compiler.PUBLIC_DATA_DIR)
    assert message is not None


def test_refuse_helper_rejects_non_default_selection_with_canonical_artifact_name_into_public_data():
    # non-default selection + the exact default artifact name, still
    # targeting public/data -- refused for two independent reasons at once
    # (canonical-name reuse and public/data target); either is sufficient.
    message = compiler._refuse_unsafe_variant_target("larger", compiler.ARTIFACT_NAME, compiler.PUBLIC_DATA_DIR)
    assert message is not None


def test_main_refuses_absolute_artifact_name_with_selection_default(tmp_path, monkeypatch):
    """Regression test for the reviewer's first bypass argv list:
    `--selection default --artifact-name /tmp/absolute-evil --out-dir ...`
    used to reach `load_annotations()` -- past the point of no return --
    because the guard's default-selection branch skipped the plain-filename
    check. `pathlib`'s `/` makes an absolute right-hand operand replace the
    left one entirely, so this would have written to `/tmp/absolute-evil*`
    regardless of `--out-dir`."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("load_annotations must not be called when the refusal rule fires")

    monkeypatch.setattr(compiler, "load_annotations", _fail_if_called)

    exit_code = compiler.main(
        ["--selection", "default", "--artifact-name", "/tmp/absolute-evil", "--out-dir", str(tmp_path)]
    )
    assert exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_main_refuses_traversal_artifact_name_with_selection_omitted(tmp_path, monkeypatch):
    """Regression test for the reviewer's second bypass argv list:
    `--artifact-name ../escape --out-dir ...` with `--selection` omitted
    entirely (argparse's `default="default"` making this identical to the
    previous test's failure mode)."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("load_annotations must not be called when the refusal rule fires")

    monkeypatch.setattr(compiler, "load_annotations", _fail_if_called)

    exit_code = compiler.main(["--artifact-name", "../escape", "--out-dir", str(tmp_path)])
    assert exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_main_refuses_variant_before_loading_any_raw_data(tmp_path, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("load_annotations must not be called when the refusal rule fires")

    monkeypatch.setattr(compiler, "load_annotations", _fail_if_called)

    exit_code = compiler.main(
        [
            "--selection",
            "random-bridge",
            "--artifact-name",
            compiler.ARTIFACT_NAME,  # deliberately the default name -> refused
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert exit_code != 0
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_main_refuses_variant_targeting_public_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compiler,
        "load_annotations",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load raw data")),
    )
    exit_code = compiler.main(
        [
            "--selection",
            "smaller",
            "--artifact-name",
            "malecns-arena-smaller",
            "--out-dir",
            str(compiler.PUBLIC_DATA_DIR),
        ]
    )
    assert exit_code != 0


def test_main_rejects_unknown_selection_id():
    with pytest.raises(SystemExit):
        compiler.main(["--selection", "not-a-real-selection"])
