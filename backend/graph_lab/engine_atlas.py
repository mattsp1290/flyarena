"""Atlas engine: the atlas search job's single child process
(`.agents/plans/graph-lab/02-job-engines.md`'s atlas engine). `jobs.py`'s
`Runner` contract is one argv, one subprocess per job -- `service.py`'s
`default_runner` points that one subprocess at `python -m
graph_lab.engine_atlas <args.json>` (this module's own `main`), which then
runs the atlas pipeline's three real steps as its *own* nested children,
sequentially:

  1. Export the graph's arm bundle (`entry-export-arms.mjs`, node --
     `scripts/graph-lab/entry-export-arms.ts`'s own bundled output).
  2. The GPU MAP-Elites search (`python -m flyarena_training.atlas_cli`,
     unmodified -- possibly a *different* Python executable than the one
     running this module: see `atlasPythonBin` below).
  3. Re-evaluate the search's candidates in TypeScript
     (`entry-atlas-reeval.mjs`, node).

Every nested step is a plain `subprocess.run` with no `start_new_session`
of its own, so it stays in *this* process's process group -- the same
group `jobs.py`'s top-level `Popen(..., start_new_session=True)` already
established for this module's own process. A cancel or timeout's
`os.killpg` therefore reaches whichever nested step is running too; there
is no separate supervision to duplicate here.

This module reports its own progress/result/error as newline-delimited
JSON on stdout, the same protocol every other graph-lab entry uses
(`jobs.py`'s `_drain`) -- one `progress` line per stage (export/search/
reeval), matching `entry-lesion.ts`'s own accepted "correctness over
progress granularity" trade-off (its own doc comment) rather than
threading through `atlas_cli.py`'s own per-generation progress lines.

Never reimplements science: every step below runs the exact same code
(`export-arms.ts`'s `runExportArms`, `flyarena_training.atlas_cli`'s
`main`, `verify-search-graph.ts`'s `verifyAndEvaluateSearchGraph`) that
`training:export-arms`/`flyarena-atlas`/`atlas:publish` already run
everywhere else in this repo.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import py_scripts


def _progress(payload: "dict[str, Any]") -> None:
    print(json.dumps({"type": "progress", "progress": payload}), flush=True)


def _result(payload: "dict[str, Any]") -> None:
    print(json.dumps({"type": "result", "result": payload}), flush=True)


def _error(message: str) -> None:
    print(json.dumps({"type": "error", "message": message}), flush=True)


def run(args: "dict[str, Any]") -> "dict[str, Any]":
    job_dir = Path(args["jobDir"])
    node_bin = args["nodeBin"]
    bundle_dir = Path(args["bundleDir"])
    data_dir = Path(args["dataDir"])
    # `atlasPythonBin`: the Python executable that runs
    # `flyarena_training.atlas_cli` -- in the container, the *same*
    # interpreter running this module (one merged environment; see
    # `Dockerfile`'s own header comment), but locally/in tests, a
    # *different* executable pointing at `training/`'s own uv-managed venv
    # (which has `torch`; `graph_lab`'s own venv deliberately does not --
    # see `pyproject.toml`'s header comment). This module's own interpreter
    # never needs `torch` itself: it only shells out to this one.
    #
    # Deliberately NOT run with `-I`/`-P` (isolated mode): a review
    # suggestion to keep `job_dir` off this step's own import path, but
    # isolated mode also ignores `PYTHONPATH` -- and in the container,
    # `flyarena_training` is importable *only* via `PYTHONPATH=/opt/
    # graph-lab` (`Dockerfile`'s own `ENV`; it is a raw copied directory,
    # never `pip install`ed). `-I` here would make every atlas job fail
    # with `ModuleNotFoundError` in production. Accepted as a documented
    # trade-off: `job_dir` only ever contains server-named `.json`/
    # `.bin.gz` files and an `arms/` directory, nothing an attacker
    # controls the *name* of.
    atlas_python_bin = args["atlasPythonBin"]

    _progress({"stage": "export-arms"})
    export_args_path = job_dir / "export-arms-args.json"
    export_args_path.write_text(
        json.dumps(
            {
                "graphPath": args["graphPath"],
                "rewiredPath": args.get("rewiredPath"),
                "outDir": str(job_dir / "arms"),
            }
        )
    )
    export_stdout = py_scripts.run_step(
        [node_bin, str(bundle_dir / "entry-export-arms.mjs"), str(export_args_path)], cwd=job_dir
    )
    export_result = py_scripts.last_json_message(export_stdout, source="entry-export-arms")
    bundle_path = Path(export_result["outDir"]) / f"{args['arm']}.json"

    _progress({"stage": "search"})
    search_output_path = job_dir / "search.json"
    search_argv = [
        atlas_python_bin,
        "-m",
        "flyarena_training.atlas_cli",
        "--graph",
        str(bundle_path),
        "--output",
        str(search_output_path),
        "--device",
        args.get("device", "cuda"),
        "--seed",
        str(args["searchSeed"]),
        "--population",
        str(args["population"]),
        "--generations",
        str(args["generations"]),
        "--ticks",
        str(args["ticks"]),
    ]
    py_scripts.run_step(search_argv, cwd=job_dir)

    _progress({"stage": "reeval"})
    expected: "dict[str, Any]" = {"arm": args["arm"], "parentGzipSha256": args["graphArtifactSha256"]}
    # `expectedSha256` is `None` for `arm == "disconnected"` (`service.py`'s
    # `_atlas_argv` never computes one) -- the key must be OMITTED
    # entirely, not set to `null`: `verify-search-graph.ts`'s
    # `ExpectedGraphIdentity.binarySha256` is optional and its own check
    # only derives the disconnected identity from `verifiedBiologicalGraph`
    # when the field is `undefined` (`expected.binarySha256 === undefined`)
    # -- JSON has no `undefined`, so a literal `null` here would round-trip
    # as a *present* key whose value fails the real sha comparison instead
    # of taking the "derive it" branch, and every disconnected atlas job
    # would fail at re-eval (confirmed empirically before this fix).
    if args.get("expectedSha256") is not None:
        expected["binarySha256"] = args["expectedSha256"]
    reeval_args_path = job_dir / "reeval-args.json"
    reeval_args_path.write_text(
        json.dumps(
            {
                "dataDir": str(data_dir),
                "searchPath": str(search_output_path),
                "expected": expected,
                "biologicalGraphPath": args["biologicalGraphPath"],
                "biologicalExpectedSha256": args["biologicalExpectedSha256"],
            }
        )
    )
    reeval_stdout = py_scripts.run_step(
        [node_bin, str(bundle_dir / "entry-atlas-reeval.mjs"), str(reeval_args_path)], cwd=job_dir
    )
    return py_scripts.last_json_message(reeval_stdout, source="entry-atlas-reeval")


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(args)
    except subprocess.CalledProcessError as error:
        # `py_scripts.describe_step_failure`: prefers the failing step's
        # own already-sanitized `{"type":"error",...}` stdout message over
        # a bare exit code (a dual-review finding -- see that function's
        # own doc comment).
        _error(py_scripts.describe_step_failure(error))
        sys.exit(1)
    except Exception as error:  # noqa: BLE001 -- top-level job driver: always report, never crash silently. Anything else (e.g. OSError) can include an absolute server-side path in its own str() (a dual-review finding, matching engine_swapset.py's identical fix) -- keep the client-facing message generic, log the real exception server-side.
        logging.exception("graph-lab atlas job failed")
        _error("an internal error occurred while running the atlas job")
        sys.exit(1)
    else:
        _result(result)


if __name__ == "__main__":
    main()
