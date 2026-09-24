"""Shared BLAS-thread-count guard for every `scripts/analysis/` CLI.

`.agents/plans/null-explanation/02-transfer-and-features.md`: "Every Python
analysis command runs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 DD_IAST_ENABLED=false` and `PYTHONPATH` unset. The CLIs
assert that the thread variables are set to 1 and refuse to run otherwise."
Multi-threaded BLAS would make dense `numpy.linalg.solve`/`eigvals`/`cond`
calls nondeterministic in exact bit pattern run-to-run (thread-count-
dependent reduction order), which would break this study's "two runs produce
byte-identical output" acceptance gate -- catch a misconfigured environment
before any computation starts rather than after, since a multi-hour batch
that silently used multiple threads would need to be entirely rerun once
noticed anyway.

Must be imported (and `assert_single_threaded_blas()` called) before `numpy`
is imported anywhere in the process: OpenBLAS/MKL/OMP read their thread-count
environment variables once, at the native library's own load time, which
happens on `import numpy` (or on first use, depending on the BLAS backend) --
setting the variable afterward has no effect on an already-initialized
thread pool. This module only reads `os.environ`, so it never triggers that
import itself.
"""

from __future__ import annotations

import os

#: Every environment variable this study's determinism guarantee depends on.
#: `DD_IAST_ENABLED=false` (Datadog's IAST instrumentation) is also required
#: by the plan but is not a thread-count variable and is not enforced here --
#: unlike a wrong thread count, a stray IAST instrumentation pass changes
#: nothing about the numeric result, only tracing overhead.
REQUIRED_SINGLE_THREADED_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def assert_single_threaded_blas(env: "os._Environ[str] | dict[str, str]" = os.environ) -> None:
    """Raise unless every variable in `REQUIRED_SINGLE_THREADED_ENV_VARS` is
    set to exactly `"1"` in `env`. `env` is injectable (rather than always
    reading the real `os.environ`) so `tests_python/` can exercise both the
    pass and fail paths without mutating the real process environment out
    from under other tests."""
    bad = {name: env.get(name) for name in REQUIRED_SINGLE_THREADED_ENV_VARS if env.get(name) != "1"}
    if bad:
        formatted = ", ".join(f"{name}={value!r}" for name, value in sorted(bad.items()))
        raise RuntimeError(
            "scripts/analysis: OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS must all be "
            f'set to "1" for reproducible dense linear algebra (see .agents/plans/null-explanation/'
            f"02-transfer-and-features.md); got {formatted}. Run with "
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1."
        )
