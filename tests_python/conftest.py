"""Session-wide test setup for `tests_python/`.

`scripts/analysis/env_guard.py`'s `assert_single_threaded_blas()` runs at
*import* time in `scripts/analysis/transfer.py`/`features.py` (before
`numpy` is ever imported in the process -- see that module's doc comment
for why). Setting these here, in `conftest.py`, guarantees they are already
in `os.environ` before pytest imports any test module that in turn imports
`transfer`/`features`, matching this study's own required run environment
(`.agents/plans/null-explanation/05-execution-handoff.md`:
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`,
`DD_IAST_ENABLED=false`, `PYTHONPATH` unset) without every test file having
to repeat it. `setdefault` so a real run's own environment (which already
sets these) is never overridden.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DD_IAST_ENABLED", "false")

_ANALYSIS_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "analysis")
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)
