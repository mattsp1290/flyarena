"""Gate 2 of `.agents/plans/graph-lab/00-overview.md`: "each job type's
reproduction check (criterion 2) passes" -- the three checks
`.agents/plans/graph-lab/02-job-engines.md`'s Tests section names, marked
`@pytest.mark.spark` there ("They run on the Spark before the WP closes
(gate 2) and again on every `scripts/graph-lab.sh --build`"). Every check
here runs the real engines against the real, checked-in MaleCNS graph data
(`public/data/`) -- a real GPU (atlas), a real built JS bundle
(`npm run graph-lab:bundle`), and (atlas only) `training/`'s own uv-managed
`torch` venv. None of that is available in the `graph-lab` CI job
(CPU-only, no Node setup, no GPU), so this whole module skips itself
(`_require_repro_environment`) rather than failing when the environment is
missing -- run for real on a dev machine or the Spark:

    npm run graph-lab:bundle
    cd backend/graph_lab && uv run pytest -m spark -v

Reuses the existing verification paths throughout -- never reimplements
`runEpisode`/the MAP-Elites search/`swap_ops`'s swap primitives; every
check below runs the *actual* production code path (or, for the swap-set
check, the actual `swap_ops.random_class_swaps` control generator
`engine_swapset.py` itself calls) against the real data and compares
against the real published artifacts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graph_lab import py_scripts
from graph_lab.service import create_app

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "public" / "data"
BUNDLE_DIR = REPO_ROOT / "backend" / "graph_lab" / "js"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TRAINING_PYTHON_BIN = REPO_ROOT / "training" / ".venv" / "bin" / "python3"

TOKEN = "graph-lab-test-token-0123456789"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
ALLOWED_ORIGIN = "http://127.0.0.1:5173"
NODE_BIN = shutil.which("node")

# `scripts/analysis/transfer.py`'s own module-level reproducibility guard
# (`env_guard.assert_single_threaded_blas`) requires these before it can
# even be imported -- see `py_scripts.load_transfer`'s doc comment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _require_repro_environment() -> None:
    if NODE_BIN is None:
        pytest.skip("node is not on PATH")
    if not (BUNDLE_DIR / "bundle.json").is_file():
        pytest.skip(f"{BUNDLE_DIR} has no bundle.json -- run `npm run graph-lab:bundle` first")


def _require_training_venv() -> None:
    if not TRAINING_PYTHON_BIN.is_file():
        pytest.skip(f"{TRAINING_PYTHON_BIN} not found -- see training/README.md's setup instructions")


def _manifest() -> dict:
    return json.loads((DATA_DIR / "malecns-arena-v1.manifest.json").read_text())


def _load_and_verify_biological_binary() -> bytes:
    manifest = _manifest()
    with gzip.open(DATA_DIR / manifest["artifact"], "rb") as fh:
        binary = fh.read()
    if hashlib.sha256(binary).hexdigest() != manifest["binarySha256"]:
        raise AssertionError("biological graph binary does not match the manifest's recorded sha256")
    return binary


def _load_and_verify_json(path: Path, expected_sha256_keypath: "tuple[str, ...]") -> dict:
    """Read a published data-file JSON, verifying its own raw bytes against
    the value the manifest records for it at `expected_sha256_keypath`
    (e.g. `("pathwayInterventions", "sha256")`) -- never trust a
    reproduction check's own comparison target without first checking it
    is the file this repo actually ships, not a stale or hand-edited copy."""
    manifest = _manifest()
    node = manifest
    for key in expected_sha256_keypath:
        node = node[key]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != node:
        raise AssertionError(f"{path} does not match manifest.{'.'.join(expected_sha256_keypath)}")
    return json.loads(raw)


def _wait_for_terminal_status(client: TestClient, identifier: str, *, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    status = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
    while status["status"] in ("queued", "running", "cancelling") and time.monotonic() < deadline:
        time.sleep(0.5)
        status = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
    return status


@pytest.mark.spark
class LesionReproductionTests(unittest.TestCase):
    """`02-job-engines.md`'s lesion reproduction check: "the `output` set
    on biological with seeds 30001-30010 and T 1800 equals `episode.ts`
    lesion scores computed offline (exact, same host)". "Computed offline"
    here means the exact same production code (`entry-lesion.mjs`, which
    calls `episode.ts`'s `runEpisode` via `worker-lesion.ts`), invoked
    directly as a standalone `node` process outside the graph-lab HTTP
    service -- proving the job-service/subprocess-supervision layer
    introduces zero discrepancy versus that direct invocation, without
    reimplementing `runEpisode` a second time in this test."""

    def setUp(self) -> None:
        _require_repro_environment()
        os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN

    def test_output_set_matches_offline_entry_lesion_exactly(self) -> None:
        manifest = _manifest()
        _load_and_verify_biological_binary()
        # `behavior-atlas-v1.json` ships its own sibling manifest
        # (`behavior-atlas-v1.manifest.json`'s own `sha256`), separate from
        # `malecns-arena-v1.manifest.json` -- verified against that one,
        # not skipped.
        behavior_atlas_manifest = json.loads((DATA_DIR / "behavior-atlas-v1.manifest.json").read_text())
        behavior_atlas_raw = (DATA_DIR / "behavior-atlas-v1.json").read_bytes()
        self.assertEqual(hashlib.sha256(behavior_atlas_raw).hexdigest(), behavior_atlas_manifest["sha256"])
        behavior_atlas = json.loads(behavior_atlas_raw)
        output_indices = behavior_atlas["outputNeuronIndices"]
        self.assertEqual(len(output_indices), len(set(output_indices)), "output indices must be unique")

        offline_args = {
            "dataDir": str(DATA_DIR),
            "mode": "biological",
            "graphPath": str(DATA_DIR / manifest["artifact"]),
            "expectedSha256": manifest["binarySha256"],
            "sets": [output_indices],
            "seedStart": 30001,
            "seedCount": 10,
            "ticks": 1800,
        }
        with tempfile.TemporaryDirectory() as tmp:
            args_path = Path(tmp) / "args.json"
            args_path.write_text(json.dumps(offline_args))
            completed = subprocess.run(
                [NODE_BIN, str(BUNDLE_DIR / "entry-lesion.mjs"), str(args_path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=True,
            )
        offline_result = py_scripts.last_json_message(completed.stdout, source="entry-lesion (offline)")

        app = create_app(
            token=TOKEN,
            data_dir=str(DATA_DIR),
            bundle_dir=str(BUNDLE_DIR),
            node_bin=NODE_BIN,
            scripts_dir=str(SCRIPTS_DIR),
        )
        body = {
            "kind": "lesion",
            "graph": "biological",
            "sets": [output_indices],
            "seedStart": 30001,
            "seedCount": 10,
            "ticks": 1800,
        }
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 202, response.text)
            status = _wait_for_terminal_status(client, response.json()["id"], timeout=300)
        self.assertEqual(status["status"], "completed", status)
        # Exact: the live job's result must equal the offline reference
        # byte-for-byte (every float compared with Python `==`, via dict
        # equality) -- not "close", not "same sign" -- the same numbers.
        self.assertEqual(status["result"], offline_result)


@pytest.mark.spark
class AtlasReproductionTests(unittest.TestCase):
    """`02-job-engines.md`'s atlas reproduction check: "on biological with
    seed 1729, P 64, G 24, and T 900, the TS-rebinned cells equal
    `behavior-atlas-v1.json` `cells` (cell index and quality)". Runs the
    real GPU search (`flyarena_training.atlas_cli`, `training/`'s own
    uv-managed torch venv) -- a real, ~1-2 minute GPU job, acceptable
    alongside a long-running training job on the shared Spark per this
    WP's own instructions."""

    def setUp(self) -> None:
        _require_repro_environment()
        _require_training_venv()
        os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN

    def test_seed_1729_matches_published_cells_exactly(self) -> None:
        _load_and_verify_biological_binary()
        app = create_app(
            token=TOKEN,
            data_dir=str(DATA_DIR),
            bundle_dir=str(BUNDLE_DIR),
            node_bin=NODE_BIN,
            scripts_dir=str(SCRIPTS_DIR),
            atlas_python_bin=str(TRAINING_PYTHON_BIN),
            atlas_device="cuda",
            # A real `torch.cuda.mem_get_info()` check would also work here
            # (this WP's own environment note: short GPU jobs are fine
            # alongside the long-running training job) -- injected anyway
            # so this test's pass/fail never depends on how much GPU
            # memory happens to be free at the moment it runs.
            gpu_free_bytes=lambda: 999 * 1024**3,
        )
        body = {
            "kind": "atlas",
            "graph": "biological",
            "searchSeed": 1729,
            "population": 64,
            "generations": 24,
            "ticks": 900,
        }
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 202, response.text)
            status = _wait_for_terminal_status(client, response.json()["id"], timeout=900)
        self.assertEqual(status["status"], "completed", status)
        result = status["result"]

        published = json.loads((DATA_DIR / "behavior-atlas-v1.json").read_text())
        self.assertEqual(len(result["cells"]), len(published["cells"]))
        published_by_cell = {cell["cell"]: (cell["id"], cell["quality"]) for cell in published["cells"]}
        result_by_cell = {cell["cell"]: (cell["id"], cell["quality"]) for cell in result["cells"]}
        self.assertEqual(result_by_cell, published_by_cell)


@pytest.mark.spark
class SwapsetReproductionTests(unittest.TestCase):
    """`02-job-engines.md`'s swap-set reproduction check: "the engine's
    control generator with class M's definition, seed 1000, and k = 6 must
    reproduce the published score of graph M1000 exactly on the published
    seeds".

    **Plan deviation, documented per the plan's own instruction ("Confirm
    that ordering against `scripts/null/intervention-report.ts`'s arm
    construction before relying on it, and fail if the ordering can't be
    proven")**: the plan additionally claims "`controls.M.scores` is
    sorted by graph id `M1000...M1099`, so M1000 is index 0". Checked
    against the actual source: `scripts/null/intervention-report.ts`'s
    `armDistribution` (the function that builds `controls.M`) sorts its
    `scores` array **by numeric value** (`[...meanScores].sort((a, b) =>
    a - b)`), not by graph id -- confirmed both by reading that function
    and empirically, since `pathway-interventions-v1.json`'s own
    `controls.M.scores[0]` is the numeric minimum of the 100 M scores, not
    a per-id designation. There is also no id-keyed raw statistics file
    checked into this repo to look M1000 up in directly (only the final,
    aggregated `pathway-interventions-v1.json` is published). The ordering
    cannot be proven, so this test does not rely on it: it recomputes
    M1000 directly (seed 1000, class M's exact masks, k = P's realized
    swap count) and asserts the recomputed score is an **exact member**
    of `controls.M.scores` -- still a real, provable, exact reproduction
    of "the published score of graph M1000", just located by value instead
    of by a false positional assumption.

    Also note: the swap-set job's own public HTTP contract only ever
    generates control seeds `0..controls-1` (`service.py`'s
    `_swapset_argv`, documented there as this WP's own seed convention,
    since the plan specifies no seed scheme of its own for arbitrary
    user-submitted swap sets) -- seed 1000 specifically is
    `interventions.py`'s own study-specific convention
    (`M_SEED_BASE = 1000`), not reachable through that public numbering.
    This test therefore calls the engine's own control-generator primitive
    (`swap_ops.random_class_swaps`, the exact function
    `engine_swapset.build_candidate_and_controls` itself calls) and the
    real `entry-swapset.mjs` scoring step directly, at seed 1000 -- same
    engine code, same science, just reached directly rather than through
    the public API's own seed range for this one specific, published
    seed.
    """

    def setUp(self) -> None:
        _require_repro_environment()
        os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN

    def test_m1000_score_is_an_exact_member_of_published_controls_m(self) -> None:
        swap_ops = py_scripts.load_swap_ops(SCRIPTS_DIR)
        rewire = py_scripts.load_rewire(SCRIPTS_DIR)
        transfer = py_scripts.load_transfer(SCRIPTS_DIR)

        biological_binary = _load_and_verify_biological_binary()
        bio_graph = rewire.decode_graph_binary(biological_binary)

        pathway = _load_and_verify_json(
            DATA_DIR / "pathway-interventions-v1.json", ("pathwayInterventions", "sha256")
        )
        k = pathway["interventions"]["P"]["swaps"]
        self.assertEqual(k, 6, "expected P's published swap count to be 6 -- update this test if it changes")

        # `interventions.py`'s own class-M masks (`_greedy_targeted_swaps`'s
        # callers, `interventions.py:748-750`): every input-labeled neuron
        # is the source class, `thrust`-population neurons are the target
        # class, and every neither-input-nor-output neuron is a bridge --
        # reused verbatim, not re-derived from anything else.
        input_mask = bio_graph.input_channel_index >= 0
        thrust_idx = transfer.OUTPUT_POPULATION_INDEX["thrust"]
        thrust_mask = bio_graph.output_population_index == thrust_idx
        bridge_mask = swap_ops.bridge_mask_of(bio_graph)

        m1000_graph, stats = swap_ops.random_class_swaps(bio_graph, k, 1000, input_mask, thrust_mask, bridge_mask)
        self.assertEqual(stats["seed"], 1000)
        self.assertEqual(stats["acceptedSwaps"], k)
        m1000_binary = rewire.binfmt.encode_graph_binary(m1000_graph)
        m1000_sha256 = hashlib.sha256(m1000_binary).hexdigest()
        self.assertNotEqual(m1000_sha256, hashlib.sha256(biological_binary).hexdigest())

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            m1000_path = tmp_path / "M1000.bin.gz"
            rewire.binfmt.write_gzip_deterministic(m1000_binary, m1000_path)
            entry_args = {
                "dataDir": str(DATA_DIR),
                "graphs": [{"graphId": "M1000", "path": str(m1000_path), "expectedSha256": m1000_sha256}],
                "seedStart": 30001,
                "seedCount": 100,
                "ticks": 1800,
            }
            args_path = tmp_path / "entry-swapset-args.json"
            args_path.write_text(json.dumps(entry_args))
            completed = subprocess.run(
                [NODE_BIN, str(BUNDLE_DIR / "entry-swapset.mjs"), str(args_path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=True,
            )
        scoring_result = py_scripts.last_json_message(completed.stdout, source="entry-swapset (M1000)")
        m1000_mean = next(entry["mean"] for entry in scoring_result["scores"] if entry["graphId"] == "M1000")

        published_scores = pathway["controls"]["M"]["scores"]
        self.assertEqual(len(published_scores), 100)
        self.assertIn(m1000_mean, published_scores)


if __name__ == "__main__":
    unittest.main()
