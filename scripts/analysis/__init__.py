"""Offline analysis CLIs for `.agents/plans/null-explanation/02-transfer-and-features.md`
(WP2): the linear transfer matrix, the linear-regime validity check inputs,
and the predeclared structural feature set, computed per graph with numpy on
the root `flyarena-graph-compiler` environment.

Deliberately outside `scripts/data/`: nothing here influences a compiled
`.bin`/`.bin.gz` artifact's bytes, so it is not part of `compile.py`'s
`compiler_source_sha256()` allowlist (`scripts/data/compile.py`'s
`COMPILER_SOURCE_FILENAMES`/`NON_COMPILER_SIDECAR_FILENAMES`) and never needs
classification there.
"""
