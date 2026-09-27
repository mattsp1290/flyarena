"""Predeclared alternative subgraph selections for `scripts/data/compile.py`
(`.agents/plans/selection-robustness`, WP1): the `SELECTIONS` policy table,
the two pure dispatch functions `select_subgraph()`/`assign_channels()` call
into for their `bridge_mode`/`channel_mode` overrides, a small helper that
dedupes the "no override given" default, and the refusal guard that keeps a
variant compile from ever landing in `public/data` or reusing the default
artifact name.

This is part of the compiler, not a post-processing sidecar: `resolve_bridge_ids`/
`resolve_sensory_order` directly determine which neurons and which channel
assignment end up in the compiled `.bin.gz`, unlike `positions.py`/
`descending_types.py`, which only annotate an already-compiled graph after
the fact. It is classified in `compile.py`'s `COMPILER_SOURCE_FILENAMES`,
not `NON_COMPILER_SIDECAR_FILENAMES`, and its bytes are part of
`compiler_source_sha256()`.

Deliberately does not import `compile.py` (that would invert the natural
dependency, since `compile.py` imports this module): `refuse_unsafe_variant_target`
takes the two `compile.py` constants it needs (`default_artifact_name`,
`public_data_dir`) as explicit keyword arguments instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

#: Predeclared alternative subgraph selections. `"default"` is
#: `compile.py`'s unchanged selection policy (degree-ranked bridge
#: population, contiguous-block channel assignment) and is what `main()`
#: uses when `--selection` is omitted -- every other entry is an explicit,
#: reproducible deviation from exactly one policy knob:
#:
#: - `larger`/`smaller`: only `bridge_target` changes (the same
#:   degree-ranked bridge selection, just a different population cutoff).
#: - `random-bridge`: bridge neurons are drawn with a seeded uniform sample
#:   instead of ranked by degree (`resolve_bridge_ids`'s `bridge_mode`).
#: - `alt-sensory-mapping`: sensory-to-channel assignment uses a seeded
#:   permutation instead of the contiguous body-id blocks
#:   (`resolve_sensory_order`'s `channel_mode`); descending-to-population
#:   assignment is always unchanged.
#:
#: Recorded verbatim (id + params) in the emitted ledger/manifest
#: `selection` key, the same way the rest of the selection policy already
#: is, so a variant artifact's provenance is reproducible from the ledger
#: alone.
SELECTIONS: Mapping[str, Mapping[str, object]] = {
    "default": {},
    "larger": {"bridge_target": 1600},
    "smaller": {"bridge_target": 400},
    "random-bridge": {"bridge_mode": "seeded-uniform", "seed": 20260927},
    "alt-sensory-mapping": {"channel_mode": "seeded-permutation", "seed": 20260927},
}


def default_if_none(selection: Mapping[str, object] | None) -> Mapping[str, object]:
    """`select_subgraph`/`assign_channels`/`build_manifest_and_ledger` all
    accept an optional `selection` override and fall back to
    `SELECTIONS["default"]` (an empty mapping) when none is given. A plain
    `selection: Mapping = SELECTIONS["default"]` default-argument would
    share one mutable dict object across every call site that never
    supplies an override; using `None` as the sentinel and normalizing here
    avoids that without each of the three duplicating this one-liner.
    """
    return selection if selection is not None else SELECTIONS["default"]


def resolve_bridge_ids(
    bridge_candidates: Iterable[int],
    degree: pd.Series,
    bridge_target: int,
    bridge_mode: str,
    seed: int | None,
    rank_by_degree: Callable[[pd.DataFrame, pd.Series], pd.DataFrame],
) -> list[int]:
    """The `bridge_mode` half of `select_subgraph`'s selection policy, over
    an already-computed `bridge_candidates` set and per-body `degree`
    series. `rank_by_degree` is `compile.py`'s `_rank_by_degree` (kept there
    as a general degree-ranking helper, not selection-variant-specific, and
    passed in rather than duplicated or imported back).

    - `"degree-rank"` (the unchanged default): rank `bridge_candidates` by
      `rank_by_degree` and keep the top `bridge_target`.
    - `"seeded-uniform"`: sample `bridge_target` candidates (or all of them,
      if fewer are available) without replacement via
      `numpy.random.default_rng(seed)`, drawn over the candidates sorted
      ascending by body id first -- so the input order to the RNG, and
      therefore its output, is itself deterministic and does not depend on
      `set` iteration order.

    Raises `ValueError` for an unknown `bridge_mode`, or for
    `"seeded-uniform"` with `seed is None` (a directly-called
    `select_subgraph(..., selection={"bridge_mode": "seeded-uniform"})`
    with no `"seed"` key -- every `SELECTIONS` entry that sets this mode
    also sets a seed, so this only guards direct API misuse, not the CLI).
    """
    if bridge_mode == "degree-rank":
        bridge_frame = pd.DataFrame({"bodyId": sorted(bridge_candidates)})
        bridge_ranked = rank_by_degree(bridge_frame, degree)
        return bridge_ranked["bodyId"].head(bridge_target).astype(np.int64).tolist()
    if bridge_mode == "seeded-uniform":
        if seed is None:
            raise ValueError("select_subgraph: bridge_mode 'seeded-uniform' requires selection['seed']")
        # Sort first so the array handed to the RNG -- and therefore which
        # indices it draws -- does not depend on `set` iteration order.
        sorted_candidates = np.array(sorted(bridge_candidates), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        sample_size = min(bridge_target, len(sorted_candidates))
        chosen_positions = rng.choice(len(sorted_candidates), size=sample_size, replace=False)
        return sorted(int(body) for body in sorted_candidates[chosen_positions])
    raise ValueError(f"select_subgraph: unknown bridge_mode {bridge_mode!r}")


def resolve_sensory_order(sensory_sorted: list[int], channel_mode: str, seed: int | None) -> list[int]:
    """The `channel_mode` half of `assign_channels`'s selection policy:
    returns the order sensory bodies are walked in before being split into
    contiguous channel blocks. `sensory_sorted` is already sorted ascending
    by body id (the unchanged default order).

    - `"contiguous-blocks"` (the unchanged default): the input order
      unchanged.
    - `"seeded-permutation"`: `sensory_sorted` permuted with
      `numpy.random.default_rng(seed)` before the same contiguous split, so
      the channel boundaries land on a seeded-random subset of bodies
      rather than a body-id-contiguous one. Descending-to-population
      assignment is never affected by `channel_mode` -- this function is
      only ever applied to the sensory side.

    Raises `ValueError` for an unknown `channel_mode`, or for
    `"seeded-permutation"` with `seed is None` (direct API misuse; see
    `resolve_bridge_ids`'s docstring for the analogous case).
    """
    if channel_mode == "contiguous-blocks":
        return sensory_sorted
    if channel_mode == "seeded-permutation":
        if seed is None:
            raise ValueError("assign_channels: channel_mode 'seeded-permutation' requires selection['seed']")
        rng = np.random.default_rng(int(seed))
        permuted_positions = rng.permutation(len(sensory_sorted))
        return [sensory_sorted[i] for i in permuted_positions.tolist()]
    raise ValueError(f"assign_channels: unknown channel_mode {channel_mode!r}")


def refuse_unsafe_variant_target(
    selection_id: str,
    artifact_name: str,
    out_dir: Path,
    *,
    default_artifact_name: str,
    public_data_dir: Path,
) -> str | None:
    """Returns an error message (and does not write anything) if the given
    `(selection_id, artifact_name, out_dir)` triple is unsafe; `None` means
    safe. Checked before any raw data is loaded or any file is written.

    The *only* permitted way to write into `public_data_dir` is the exact
    canonical default invocation: `selection_id == "default"` **and**
    `artifact_name == default_artifact_name`. Every other combination that
    would resolve into `public_data_dir` is refused, and `artifact_name`
    must always be a plain filename stem, regardless of `selection_id`.

    A prior version of this function special-cased `selection_id ==
    "default"` to return "safe" immediately, before either the
    plain-filename check or the out-dir check ran. Since `--selection`
    defaults to `"default"` in `main()`'s `argparse` setup, that let
    `--artifact-name` alone -- with `--selection` left at its default, or
    simply omitted -- smuggle an absolute path or a `..`-traversal path
    into the real write target (`out_dir / f"{artifact_name}.bin.gz"`;
    `pathlib`'s `/` operator makes an absolute right-hand operand *replace*
    the left one entirely, so `out_dir / "/tmp/evil"` is `Path("/tmp/evil")`
    regardless of `out_dir`). A dual review confirmed both
    `--artifact-name /tmp/absolute-evil` and `--artifact-name ../escape`
    reached `load_annotations()` -- past the point of no return -- under
    the default selection. Making both checks unconditional, with the
    canonical triple as the single explicit exception, closes this without
    reintroducing the asymmetry: there is one invariant ("only the exact
    canonical default call may land in `public_data_dir`; every other call
    must use a plain filename and land elsewhere"), not two independently
    maintained branches that happened to enforce different subsets of it.
    """
    if (
        not artifact_name
        or artifact_name in (".", "..")
        or Path(artifact_name).is_absolute()
        or Path(artifact_name).name != artifact_name
    ):
        return (
            f"--artifact-name {artifact_name!r} must be a plain filename stem (no path "
            "separators, '.', '..', and not absolute); refusing an ambiguous variant output path"
        )

    is_canonical_default = selection_id == "default" and artifact_name == default_artifact_name

    resolved_out_dir = out_dir.resolve()
    resolved_public_data = public_data_dir.resolve()
    targets_public_data = (
        resolved_out_dir == resolved_public_data or resolved_public_data in resolved_out_dir.parents
    )
    if targets_public_data and not is_canonical_default:
        return (
            f"--out-dir {resolved_out_dir} resolves inside {resolved_public_data}, and this is not "
            f"the canonical default compile (--selection default --artifact-name "
            f"{default_artifact_name!r}); refusing to write here"
        )

    if selection_id != "default" and artifact_name == default_artifact_name:
        return (
            f"--selection {selection_id!r} requires --artifact-name different from the "
            f"default ({default_artifact_name!r}); refusing to compile a variant under the default name"
        )

    return None
