"""Python-side mirror of `scripts/lib/import-graph.ts`'s
`collectRepoRelativeDependencies`, used only for verification: `explain_
provenance.py`'s `current_regime_source_sha256` calls this (never the real
`scripts/lib/import-graph.ts`, no TypeScript execution required) to
independently recompute the exact same repo-relative dependency set
`scripts/null/regime-check.ts`'s `regimeProducer()` stamped `regime.json`
with, and refuse a `regime.json` whose recorded `producer.sourceSha256`
doesn't match.

This is a regex-based scanner over raw `.ts` source text, deliberately not
a real TypeScript parser (matching this study's existing convention: `graph_
io.source_identity_sha256`/`scripts/data/compile.py`'s `compiler_source_
sha256` already hash raw bytes rather than executing anything, and `tests_
python/test_null_stats_cross_check.py` already established the "implement
the identical scheme once per language, then cross-check the two agree" for
this exact category of problem, just for a statistics function rather than
an import walker). `tests_python/test_ts_import_graph_cross_check.py`
(`npx tsx`, skipped rather than failed if Node isn't available -- the same
pattern) confirms this module and the real `collectRepoRelativeDependencies`
produce byte-identical file sets and hashes for `scripts/null/
regime-check.ts`'s own closure, so the two implementations cannot silently
drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

#: The exact same relative-import-specifier pattern `scripts/lib/
#: import-graph.ts`'s `RELATIVE_IMPORT_RE` uses: `from '...'`/`from "..."`
#: (covers both `import ... from '...'` and `export ... from '...'`,
#: including a multi-line `import {\n  a,\n  b\n} from '...'` clause, since
#: the `from '...'` always ends the statement on its own line regardless of
#: how many lines the specifier list spans), a dynamic `import('...')`, or a
#: `new URL('...', import.meta.url)` worker-path reference (`regime-check.
#: ts`'s own way of locating its forked `regime-worker.ts` -- see the TS
#: module's doc comment for why this third alternative is required).
_RELATIVE_IMPORT_RE = re.compile(
    r"""\bfrom\s+['"](\.[^'"]+)['"]|\bimport\(\s*['"](\.[^'"]+)['"]\s*\)|\bnew\s+URL\(\s*['"](\.[^'"]+)['"]"""
)

#: See `scripts/lib/import-graph.ts`'s `stripComments` doc comment -- this
#: repo's producer files routinely quote example import syntax inside their
#: own doc comments, which `_RELATIVE_IMPORT_RE` would otherwise mistake for
#: a real dependency; comments are stripped before the specifier scan, not
#: after, in both language's walkers identically (required for the two to
#: agree file-for-file).
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def _strip_comments(source: str) -> str:
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", source))

#: The same candidate resolution order `scripts/lib/import-graph.ts`'s
#: `RESOLUTION_CANDIDATES` tries, so both walkers resolve every specifier to
#: the identical file.
def _resolution_candidates(base: Path) -> "list[Path]":
    return [
        Path(f"{base}.ts"),
        Path(f"{base}.tsx"),
        base / "index.ts",
        base / "index.tsx",
        base,
    ]


def _extract_relative_import_specifiers(source: str) -> "list[str]":
    specifiers: list[str] = []
    for match in _RELATIVE_IMPORT_RE.finditer(_strip_comments(source)):
        specifier = match.group(1) or match.group(2) or match.group(3)
        if specifier:
            specifiers.append(specifier)
    return specifiers


def _resolve_relative_import(from_file: Path, specifier: str) -> Path:
    base = (from_file.parent / specifier).resolve()
    for candidate in _resolution_candidates(base):
        if candidate.is_file():
            return candidate
    raise ValueError(f"ts_import_graph: cannot resolve {specifier!r} (imported from {from_file})")


def collect_repo_relative_dependencies(entry_file: Path, repo_root: Path) -> "list[str]":
    """See `scripts/lib/import-graph.ts`'s `collectRepoRelativeDependencies`
    doc comment -- identical algorithm, restated in Python so `explain_
    provenance.py` can verify a `regime.json` producer sha without a
    TypeScript toolchain."""
    visited: set[Path] = set()
    stack = [entry_file.resolve()]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        source = current.read_text(encoding="utf-8")
        for specifier in _extract_relative_import_specifiers(source):
            resolved = _resolve_relative_import(current, specifier)
            if resolved not in visited:
                stack.append(resolved)
    return sorted(str(path.relative_to(repo_root)).replace("\\", "/") for path in visited)
