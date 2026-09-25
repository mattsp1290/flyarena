"""Offline analysis CLIs for `.agents/plans/null-explanation/02-transfer-and-features.md`
(WP2): the linear transfer matrix, the linear-regime validity check inputs,
and the predeclared structural feature set, computed per graph with numpy on
the root `flyarena-graph-compiler` environment.

Deliberately outside `scripts/data/`: nothing here influences a compiled
`.bin`/`.bin.gz` artifact's bytes, so it is not part of `compile.py`'s
`compiler_source_sha256()` allowlist (`scripts/data/compile.py`'s
`COMPILER_SOURCE_FILENAMES`/`NON_COMPILER_SIDECAR_FILENAMES`) and never needs
classification there.

Run these modules as scripts (`uv run python scripts/analysis/transfer.py
...`), not as `-m` modules: `transfer.py`/`features.py` resolve `env_guard`/
`graph_io`/`binfmt`/`rewire`/`fsutil` via a `sys.path.insert` at the top of
the file (matching `scripts/data/`'s own established convention), which only
runs when the file is executed directly or imported with `scripts/analysis/`
already on `sys.path` (as `tests_python/conftest.py` arranges for the test
suite) -- `python -m scripts.analysis.transfer` does not go through either
path and fails. This `__init__.py` exists so `scripts.analysis` is a valid
package name for tooling that inspects the directory tree, not to invite
`-m`-style invocation.
"""
