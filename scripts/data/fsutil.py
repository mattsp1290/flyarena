"""Filesystem helpers shared across `scripts/data/`'s scripts.

This is a **non-compiler sidecar** (`compile.py`'s `NON_COMPILER_SIDECAR_FILENAMES`):
it never reads or writes anything that influences a compiled `.bin`/`.bin.gz`
artifact's bytes, so adding or editing it must not change
`compiler_source_sha256()` -- see `compile.py`'s `COMPILER_SOURCE_FILENAMES`
docstring and `docs/data-provenance.md`'s "Compiler provenance" section.

Currently holds one canonical helper, `atomic_write_text`, previously
duplicated (with diverging robustness) as `rewire_batch.py`'s
`_write_text_atomic` and `positions.py`'s `_atomic_write_text`. Both now call
this single implementation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via a same-directory temp file (a unique,
    randomly-suffixed name from `tempfile.mkstemp`, not a fixed `.tmp`
    sibling) plus `os.replace`, so:

    - a process killed mid-write leaves `path` with its previous contents
      (or nothing, if it never existed), never something truncated or
      half-written;
    - the temp file is fsync'd before the rename, so its contents are
      durable on disk before `path` is made to point at them -- not just
      buffered in the OS page cache;
    - two concurrent writers targeting the same `path` never collide on the
      same temp filename (unlike a fixed `path.name + ".tmp"` sibling);
    - a write that raises partway through cleans up its own temp file
      rather than leaving a stray one behind.

    `path.parent` must already exist.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
