"""Selection-robustness WP2 (`.agents/plans/selection-robustness/
02-per-selection-chain.md`) support for `scripts/analysis/explain.py`: the
public/docs refusal guard for a non-shipped-graph run, and the `exploratory`
output-field builder for a `--selection-mode` run with no
`--features-exploratory-unrestricted`.

Extracted out of `explain.py` -- adding these inline pushed that file past
this repo's own "do not let a file cross 1000 lines without a very strong
reason" rule (see `explain.py`'s doc comment for the precedent this repeats;
`scripts/null/null-report-variant.ts` was extracted out of `null-report.ts`
for the same reason). Deliberately does not import `explain.py` (one-
directional import boundary, matching `explain_provenance.py`/
`explain_stats.py`/`explain_report.py`'s own convention) -- every path this
module needs (`public_data_dir`/`docs_dir`) is passed in by the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Why `--selection-mode` runs never carry a `featureSixUnrestricted`
#: reading: the historical pre-adjudication snapshot is a pinned, stale-code
#: input by design (see `explain_provenance.py`'s doc comment) and has no
#: per-selection counterpart to regenerate against a selection's own rewired
#: graphs (`docs/null-explanation-report.md:201`).
EXPLORATORY_OMITTED_REASON = (
    "selection-robustness WP2 (--selection-mode): the historical pre-adjudication "
    "features-exploratory-unrestricted snapshot has no per-selection counterpart and cannot be "
    "regenerated against this selection's own rewired graphs (docs/null-explanation-report.md:201)"
)


def guard_selection_scratch_target(
    path: Path,
    flag_label: str,
    source_graph_sha256: str,
    *,
    public_data_dir: Path,
    docs_dir: Path,
) -> None:
    """The same "only the shipped graph may write into the shipped tree"
    refusal `scripts/null/null-report.ts`'s `guardSelectionScratchTarget`
    enforces on the TypeScript side, mirrored here for `explain.py`'s own
    `--out`/`--report-out`/`--manifest`. Keyed on `source_graph_sha256`
    (the artifact the run was actually scored against, e.g.
    `rewiring_null["sourceGraphSha256"]`) versus the *shipped* manifest's
    `binarySha256` under `public_data_dir` -- never the caller's own
    `--manifest`, which may already be a selection's own scratch manifest
    and would trivially "match itself". A directory-*prefix* check (like
    `null-report-variant.ts`'s `guardVariantOutPath` tree block), not an
    exact-path one: a differently-named file under `public/` or `docs/`
    must be refused too, not only the three shipped default filenames.
    Applies unconditionally -- selection mode does not relax it; this is
    what makes selection mode safe to add in the first place. Fails safe:
    an unreadable/missing shipped manifest is treated as "graph not
    shipped".
    """
    shipped_manifest_path = public_data_dir / "malecns-arena-v1.manifest.json"
    shipped_sha256: str | None = None
    if shipped_manifest_path.exists():
        with shipped_manifest_path.open("r") as fh:
            shipped_sha256 = json.load(fh).get("binarySha256")
    if shipped_sha256 is not None and shipped_sha256 == source_graph_sha256:
        return
    resolved = path.resolve()
    for shipped_dir in (public_data_dir.parent, docs_dir):
        shipped_dir = shipped_dir.resolve()
        if resolved == shipped_dir or shipped_dir in resolved.parents:
            described = (
                f" ({shipped_sha256})" if shipped_sha256 is not None else " (the shipped manifest could not be read)"
            )
            raise ValueError(
                f"explain: {flag_label} ({resolved}) resolves under {shipped_dir}, but this run was scored "
                f"against a graph with sha256 {source_graph_sha256}, which does not match the shipped biological "
                f"graph{described} -- refusing to write a non-shipped-graph artifact into a shipped tree. Pass an "
                "explicit scratch path outside public/ and docs/ (e.g. training/runs/selections/<id>/...)."
            )


def build_exploratory_field(
    features_exploratory_json: dict | None,
    exploratory_metrics: list | None,
    source_sha256: str | None,
) -> dict:
    """Returns the `exploratory`(`/exploratoryOmittedReason`) key(s) to merge
    into `explain.py`'s output dict. `None`/`exploratory_metrics=None` (only
    ever together -- `explain.main()` computes both from the same
    `features_exploratory_json is None` branch) means a `--selection-mode`
    run with the flag omitted; every other caller (the required-flag,
    non-selection-mode path) always has both.
    """
    if features_exploratory_json is None:
        return {"exploratory": None, "exploratoryOmittedReason": EXPLORATORY_OMITTED_REASON}
    return {
        "exploratory": {
            "featureSixUnrestricted": {"sourceSha256": source_sha256, "metrics": exploratory_metrics}
        }
    }
