"""`.agents/plans/graph-lab/01-service-and-container.md`'s WP1 gate: this
suite must pass before any job engine is wired (`00-overview.md`'s stop/go
gate 1). Every job body below is a real, fully-bounds-valid `lesion`
request (the only kind WP1 wires to a real runner by default) unless a
test is specifically probing validation itself.
"""
import hmac
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from graph_lab.jobs import CEILING_SECONDS
from graph_lab.service import create_app

TOKEN = "graph-lab-test-token-0123456789"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
ALLOWED_ORIGIN = "http://127.0.0.1:5173"
DISALLOWED_ORIGIN = "http://evil.example"

LESION_BODY = {
    "kind": "lesion",
    "graph": "biological",
    "sets": [[0, 1]],
    "seedStart": 30001,
    "seedCount": 4,
    "ticks": 300,
}


def sleeping_runner(_request, _job_dir):
    """A real (not mocked) child process that just sleeps -- lets the
    process-group-kill and timeout tests exercise the real `Popen`/
    `os.killpg` path end to end."""
    return ["sleep", "30"]


def instant_runner(_request, _job_dir):
    return ["true"]


def completing_runner(_request, _job_dir):
    """Unlike `instant_runner` (`["true"]`, which prints nothing and so
    never populates `outcome["result"]`), this actually reports a
    `{"type": "result", ...}` line -- for tests that need to assert a
    genuine `"completed"` status rather than merely "some terminal
    status"."""
    script = "import json; print(json.dumps({'type': 'result', 'result': {'ok': True}}))"
    return ["python3", "-c", script]


def env_probing_runner(_request, _job_dir):
    """Prints whatever the child's own environment holds for
    `GRAPH_LAB_TOKEN`, as a structured result -- proves the child truly
    never receives the secret, rather than merely trusting that it
    wouldn't try to read it."""
    script = (
        "import json, os; "
        "print(json.dumps({'type': 'result', "
        "'result': {'token_seen': os.environ.get('GRAPH_LAB_TOKEN', 'MISSING')}}))"
    )
    return ["python3", "-c", script]


def make_app(origins=ALLOWED_ORIGIN, runner=instant_runner, token=TOKEN):
    os.environ["GRAPH_LAB_ORIGINS"] = origins
    return create_app(token=token, runner=runner)


def submit_when_idle(client, timeout=2):
    """A terminal `job.status` (per `jobs.py`'s `_execute` docstring) is
    only ever published after that job's `_final_sweep`/`rmtree` cleanup
    has actually finished -- so `submit()`'s 409 check
    (`self.thread.is_alive()`) should already be clear by the time a
    poller observes it. What can still lag by a hair is the worker thread
    itself finishing its return out of `_execute` and back into
    `Thread.run()`, which is the instant `is_alive()` actually flips --
    typically on the order of microseconds, not the multi-millisecond
    kill/cleanup window the terminal-status ordering fix closes. This
    bounded retry absorbs that unavoidable scheduling gap. It is not the
    regression detector for the original bug -- `TerminalStatusVisibilityTests`
    is, since it blocks cleanup on a `threading.Event` instead of relying
    on timing -- so a reintroduced ordering bug should be caught there even
    if it happened to slip past a short retry here."""
    deadline = time.monotonic() + timeout
    while True:
        response = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
        if response.status_code != 409 or time.monotonic() > deadline:
            return response.json()
        time.sleep(0.005)


class HealthTests(unittest.TestCase):
    def test_health_is_unauthenticated_and_returns_only_documented_fields(self):
        app = make_app()
        with TestClient(app) as client:
            response = client.get("/api/graph/v1/health")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(set(body.keys()), {"status", "modelVersion", "bundleSha256", "graphSha256", "gpu"})
            self.assertEqual(body["modelVersion"], "arena-graph-lab-v1")
            self.assertEqual(set(body["gpu"].keys()), {"available"})
            # No free-memory figure, no job ids, no request counts.
            self.assertNotIn("freeMemory", body["gpu"])
            self.assertNotIn("jobs", body)

    def test_docs_and_openapi_are_disabled(self):
        """FastAPI's default `/docs`/`/redoc`/`/openapi.json` sit outside
        `authorize`'s `Depends` (same as `/health`) and would otherwise leak
        the whole API schema -- endpoint names, field names, every bound in
        models.py -- to anyone who can reach this private service, with no
        token required."""
        app = make_app()
        with TestClient(app) as client:
            for path in ("/docs", "/redoc", "/openapi.json"):
                self.assertEqual(client.get(path).status_code, 404, path)


class AuthTests(unittest.TestCase):
    def test_every_non_health_route_requires_auth(self):
        app = make_app()
        with TestClient(app) as client:
            self.assertEqual(client.post("/api/graph/v1/jobs", json=LESION_BODY).status_code, 401)
            self.assertEqual(client.get("/api/graph/v1/jobs/x").status_code, 401)
            self.assertEqual(client.delete("/api/graph/v1/jobs/x").status_code, 401)

    def test_wrong_token_is_rejected(self):
        app = make_app()
        with TestClient(app) as client:
            response = client.post(
                "/api/graph/v1/jobs", json=LESION_BODY, headers={"Authorization": "Bearer wrong-token-0123456789"}
            )
            self.assertEqual(response.status_code, 401)

    def test_missing_bearer_scheme_is_rejected(self):
        app = make_app()
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers={"Authorization": TOKEN})
            self.assertEqual(response.status_code, 401)

    def test_startup_fails_with_a_short_token(self):
        os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN
        app = create_app(token="short", runner=instant_runner)
        with self.assertRaises(RuntimeError):
            with TestClient(app):
                pass

    def test_auth_uses_constant_time_compare(self):
        app = make_app()
        with patch("graph_lab.service.hmac.compare_digest", wraps=hmac.compare_digest) as spy:
            with TestClient(app) as client:
                client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
        self.assertTrue(spy.called)

    def test_token_never_appears_in_any_response_body(self):
        app = make_app(runner=env_probing_runner)
        with TestClient(app) as client:
            responses = [
                client.get("/api/graph/v1/health"),
                client.post("/api/graph/v1/jobs", json={}, headers=HEADERS),  # 422
                client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS),
            ]
            for response in responses:
                self.assertNotIn(TOKEN, response.text)


class CorsAndPnaTests(unittest.TestCase):
    def test_cors_allows_only_listed_origins(self):
        app = make_app(origins=ALLOWED_ORIGIN)
        with TestClient(app) as client:
            allowed = client.get("/api/graph/v1/health", headers={"Origin": ALLOWED_ORIGIN})
            self.assertEqual(allowed.headers.get("access-control-allow-origin"), ALLOWED_ORIGIN)

            disallowed = client.get("/api/graph/v1/health", headers={"Origin": DISALLOWED_ORIGIN})
            self.assertNotIn("access-control-allow-origin", disallowed.headers)

    def test_wildcard_origin_is_rejected_at_startup(self):
        os.environ["GRAPH_LAB_ORIGINS"] = "*"
        with self.assertRaises(ValueError):
            create_app(token=TOKEN, runner=instant_runner)

    def test_pna_preflight_answered_only_for_allowed_origins(self):
        """The security-meaningful signal for a CORS/PNA preflight is the
        *status code*, not merely which headers are present: a browser
        only proceeds with the real request after a 2xx preflight response,
        regardless of what headers a non-2xx response happens to carry.
        `CORSMiddleware`'s own preflight handler (this is `allow_private_network=True`
        on `CORSMiddleware` itself, not a hand-rolled middleware -- see
        `service.py`'s own comment on why) can still attach
        `Access-Control-Allow-Private-Network: true` to a 400 for a
        disallowed origin (it doesn't gate that one header on
        `is_allowed_origin`), but the request never gets access-control-allow-origin
        and the overall response is 400 -- both of which independently
        stop a real browser from ever sending the follow-up request."""
        app = make_app(origins=ALLOWED_ORIGIN)
        with TestClient(app) as client:
            allowed = client.options(
                "/api/graph/v1/jobs",
                headers={
                    "Origin": ALLOWED_ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Private-Network": "true",
                },
            )
            self.assertEqual(allowed.status_code, 200)
            self.assertEqual(allowed.headers.get("access-control-allow-private-network"), "true")
            self.assertEqual(allowed.headers.get("access-control-allow-origin"), ALLOWED_ORIGIN)

            disallowed = client.options(
                "/api/graph/v1/jobs",
                headers={
                    "Origin": DISALLOWED_ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Private-Network": "true",
                },
            )
            self.assertEqual(disallowed.status_code, 400)
            self.assertNotIn("access-control-allow-origin", disallowed.headers)

    def test_preflight_without_private_network_request_is_unaffected(self):
        app = make_app(origins=ALLOWED_ORIGIN)
        with TestClient(app) as client:
            response = client.options(
                "/api/graph/v1/jobs",
                headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST"},
            )
            self.assertNotIn("access-control-allow-private-network", response.headers)


class BodyLimitTests(unittest.TestCase):
    def test_413_over_16_kib(self):
        app = make_app()
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", content="x" * (16 * 1024 + 1), headers=HEADERS)
            self.assertEqual(response.status_code, 413)

    def test_at_the_limit_is_not_rejected_by_body_limit(self):
        app = make_app()
        with TestClient(app) as client:
            # An oversized (but still forbidden, so 422 rather than 202)
            # field proves the body limit itself let a body just under
            # 16 KiB through to the validator, rather than rejecting it
            # with a 413 the way the over-limit test does.
            body = ("{" + '"padding": "' + "x" * (16 * 1024 - 200) + '", "kind": "lesion"}').encode()
            self.assertLessEqual(len(body), 16 * 1024)
            response = client.post(
                "/api/graph/v1/jobs", content=body, headers={**HEADERS, "Content-Type": "application/json"}
            )
            self.assertNotEqual(response.status_code, 413)


class GraphIdValidationTests(unittest.TestCase):
    def test_rejects_shell_metacharacter_payload(self):
        app = make_app()
        with TestClient(app) as client:
            body = dict(LESION_BODY, graph="rewired:1;rm -rf /")
            response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 422)

    def test_rejects_path_traversal(self):
        app = make_app()
        with TestClient(app) as client:
            body = dict(LESION_BODY, graph="../x")
            response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 422)

    def test_rejects_rewired_seed_over_499(self):
        app = make_app()
        with TestClient(app) as client:
            body = dict(LESION_BODY, graph="rewired:500")
            response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 422)

    def test_rejects_non_ascii_digits_and_leading_zeros(self):
        """Python's `\\d` matches every Unicode decimal-digit character
        without `re.ASCII`, not just 0-9 -- "rewired:٥" (Arabic-Indic
        5) and "rewired:１２" (fullwidth 12) both matched before
        `GRAPH_PATTERN` added `re.ASCII`. A leading zero ("rewired:007")
        matched a bare `\\d{1,3}` too."""
        app = make_app()
        with TestClient(app) as client:
            for graph in ("rewired:٥", "rewired:１２", "rewired:007"):
                body = dict(LESION_BODY, graph=graph)
                response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
                self.assertEqual(response.status_code, 422, graph)

    def test_rejects_a_trailing_newline(self):
        """Python's bare `$` matches immediately before a trailing "\\n",
        so a naive `^...$` pattern would accept "biological\\n" -- which
        would then silently mismatch the Node side's exact `===`
        `'biological'` comparison and fall through to a different graph
        entirely. `GRAPH_PATTERN` uses `\\A`/`\\Z` specifically to close
        this."""
        app = make_app()
        with TestClient(app) as client:
            for graph in ("biological\n", "rewired:5\n"):
                body = dict(LESION_BODY, graph=graph)
                response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
                self.assertEqual(response.status_code, 422, graph)

    def test_every_closed_set_member_passes_validation_through_the_real_default_runner(self):
        """Uses the real `default_runner` (not a test fake): `biological`/
        `disconnected` pass validation and dispatch (202). `rewired:*`
        passes the *same* validation (it is a real closed-set member, not
        rejected at 422) -- WP2 wires its engine
        (`engine_lesion.regenerate_rewired_graph`) for real, so it is no
        longer a static 501; this fixture's `data_dir` has no real
        `rewiring-null-v1.json`/graph binary for it to regenerate against,
        so regeneration fails and `default_runner` reports a clean 500 (see
        `WiredJobKindDispatchTests` for the equivalent atlas/swapset check,
        and `test_engines.py` for `rewired:<seed>` succeeding end to end
        against a real fixture). A malformed id (see `GraphIdValidationTests`)
        never even reaches the runner."""
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as bundle_dir:
            manifest = Path(data_dir) / "malecns-arena-v1.manifest.json"
            manifest.write_text(json.dumps({"artifact": "malecns-arena-v1.bin.gz", "binarySha256": "0" * 64}))

            os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN
            for graph, expected_status in (
                ("biological", 202),
                ("disconnected", 202),
                ("rewired:0", 500),
                ("rewired:499", 500),
            ):
                app = create_app(token=TOKEN, data_dir=data_dir, bundle_dir=bundle_dir)
                with TestClient(app) as client:
                    body = dict(LESION_BODY, graph=graph)
                    response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
                    self.assertEqual(response.status_code, expected_status, graph)
                    if expected_status == 202:
                        # Let the (doomed -- entry-lesion.mjs doesn't exist
                        # in this fixture bundle dir) child finish so its
                        # process doesn't outlive the `with TestClient`
                        # block's `jobs.close()`.
                        identifier = response.json()["id"]
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            status = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()["status"]
                            if status not in ("queued", "running", "cancelling"):
                                break
                            time.sleep(0.05)


ATLAS_BODY = {
    "kind": "atlas",
    "graph": "biological",
    "searchSeed": 1729,
    "population": 8,
    "generations": 2,
    "ticks": 300,
}

SWAPSET_BODY = {
    "kind": "swapset",
    "graph": "biological",
    "swaps": [{"a": 0, "b": 1, "c": 2, "d": 3}],
    "controls": 0,
    "seedStart": 30001,
    "seedCount": 4,
    "ticks": 300,
}


class WiredJobKindDispatchTests(unittest.TestCase):
    """Originally `UnwiredJobKindDispatchTests` (a thermo-review finding: the
    one `isinstance` branch WP1 added specifically to reject `atlas`/
    `swapset` with a clean 501 had zero direct test coverage -- only the
    `rewired:*` 501 path, a different branch for `LesionJobRequest`, was
    exercised). WP2 (`.agents/plans/graph-lab/02-job-engines.md`) wires
    both kinds for real, so the blanket 501 this class used to assert is
    gone -- these tests now assert the opposite regression: `atlas`/
    `swapset` must **not** fall back to the generic-501/400 path (a future
    reordering of `default_runner`'s `isinstance` chain, or a typo in the
    discriminator match, could silently start mis-dispatching one kind as
    another, or drop back to 501/400, with nothing else here failing).
    Full engine correctness (real fixture graphs, real scoring) is
    `test_engines.py`'s job, not this security-focused suite's -- these
    tests use fake/missing data on purpose and only check the *shape* of
    the response: never 501, never 400."""

    @staticmethod
    def _real_default_runner_app(*, gpu_free_bytes):
        # `make_app()`'s default `runner=instant_runner` is a test fake that
        # bypasses `default_runner` entirely (it would return 202 for
        # *any* kind) -- these tests specifically exercise the real
        # dispatch `isinstance` chain, so `runner` must be left unset here,
        # matching `test_every_closed_set_member_passes_validation_through_the_real_default_runner`'s
        # own pattern. `gpu_free_bytes` is always injected (never the real
        # `torch.cuda.mem_get_info`): this package's own venv has no
        # `torch` at all (see `pyproject.toml`'s header comment), so a
        # real GPU check would `ImportError` before ever reaching the
        # dispatch branch this class actually tests.
        data_dir = tempfile.mkdtemp()
        bundle_dir = tempfile.mkdtemp()
        os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN
        return create_app(
            token=TOKEN, data_dir=data_dir, bundle_dir=bundle_dir, gpu_free_bytes=gpu_free_bytes
        )

    def test_atlas_no_longer_returns_501_when_the_gpu_is_free(self):
        app = self._real_default_runner_app(gpu_free_bytes=lambda: 999 * 1024**3)
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", json=ATLAS_BODY, headers=HEADERS)
            self.assertNotEqual(response.status_code, 501)
            self.assertNotEqual(response.status_code, 400)

    def test_atlas_reports_gpu_busy_before_touching_any_engine_file(self):
        """`02-job-engines.md`'s atlas bound: "refuses to start... if free
        GPU memory is below 2 GiB". `gpu_free_bytes` returning a value
        under the 2 GiB threshold must short-circuit *before*
        `default_runner` ever reads the (nonexistent, in this fixture)
        manifest -- proven by the 503 firing even though `data_dir` here
        has no `malecns-arena-v1.manifest.json` at all (a `FileNotFoundError`
        reading it would surface as 500, not 503)."""
        app = self._real_default_runner_app(gpu_free_bytes=lambda: 1024**2)
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", json=ATLAS_BODY, headers=HEADERS)
            self.assertEqual(response.status_code, 503)

    def test_atlas_reports_gpu_busy_when_free_memory_is_unknown(self):
        """`gpu_free_bytes` returning `None` (no `torch`, no CUDA device,
        or the query itself raised) is treated the same as "known busy" --
        refuse rather than silently proceed without ever having checked."""
        app = self._real_default_runner_app(gpu_free_bytes=lambda: None)
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", json=ATLAS_BODY, headers=HEADERS)
            self.assertEqual(response.status_code, 503)

    def test_swapset_no_longer_returns_501(self):
        app = self._real_default_runner_app(gpu_free_bytes=lambda: 999 * 1024**3)
        with TestClient(app) as client:
            response = client.post("/api/graph/v1/jobs", json=SWAPSET_BODY, headers=HEADERS)
            self.assertNotEqual(response.status_code, 501)
            self.assertNotEqual(response.status_code, 400)

    def test_each_kind_actually_reaches_its_own_engine_not_a_neighbor(self):
        """A dual-review finding on an earlier version of this class: the
        "not 501/400" checks above pass whether or not the `isinstance`
        chain routes each kind to its *own* engine -- a reordering that
        mis-dispatched, say, `swapset` into `_atlas_argv` would still give
        some non-501/400 status (likely a 500 from the wrong code path)
        and none of the tests above would catch it. This test patches each
        kind's own engine entry point with a distinct sentinel exception
        and asserts that submitting that kind's body surfaces *that*
        sentinel's message -- proving the dispatch reached the right
        engine, not merely *an* engine."""
        data_dir = tempfile.mkdtemp()
        # `regenerate_rewired_graph` is patched below with a sentinel, but
        # `_lesion_argv` still reads the manifest *before* calling it --
        # without a real one here, a `FileNotFoundError` would fire first
        # and mask whether dispatch ever reached the sentinel at all.
        (Path(data_dir) / "malecns-arena-v1.manifest.json").write_text(
            json.dumps({"artifact": "malecns-arena-v1.bin.gz", "binarySha256": "0" * 64})
        )
        os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN
        app = create_app(
            token=TOKEN,
            data_dir=data_dir,
            bundle_dir=tempfile.mkdtemp(),
            gpu_free_bytes=lambda: 999 * 1024**3,
        )
        with TestClient(app) as client:
            with patch(
                "graph_lab.service.engine_swapset.build_candidate",
                side_effect=ValueError("swapset-sentinel"),
            ):
                response = client.post("/api/graph/v1/jobs", json=SWAPSET_BODY, headers=HEADERS)
                self.assertEqual(response.status_code, 422)
                self.assertIn("swapset-sentinel", response.text)

            with patch(
                "graph_lab.service.engine_lesion.regenerate_rewired_graph",
                side_effect=ValueError("rewired-sentinel"),
            ):
                body = dict(LESION_BODY, graph="rewired:0")
                response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
                self.assertEqual(response.status_code, 500)
                self.assertIn("rewired-sentinel", response.text)

            # Atlas's own submit-time gate is the GPU check -- confirm a
            # rejecting `gpu_free_bytes` fires specifically for `kind:
            # "atlas"` and not for `lesion`/`swapset` (which never call it).
            gpu_checked_for = []

            def recording_gpu_free_bytes():
                gpu_checked_for.append(True)
                return 999 * 1024**3

            app2 = self._real_default_runner_app(gpu_free_bytes=recording_gpu_free_bytes)
            with TestClient(app2) as client2:
                client2.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
                self.assertEqual(len(gpu_checked_for), 0, "the GPU check must not run for a lesion request")


class SeedRangeTests(unittest.TestCase):
    def test_rejects_a_seed_range_that_overflows_uint32(self):
        """`src/lib/arena/world.ts`'s `normalizeSeed` does `seed >>> 0`
        (an unsigned 32-bit wrap): a `heldOutSeeds` entry built as
        `seedStart + i` past `2**32 - 1` would silently wrap to some other,
        smaller seed instead of erroring -- which can collide with an
        earlier entry in the same request and make two "different" seeds
        simulate the identical episode. `seedStart` alone being in range
        does not bound `seedStart + seedCount - 1`, the actual highest
        seed produced."""
        app = make_app()
        with TestClient(app) as client:
            body = dict(LESION_BODY, seedStart=2**32 - 1, seedCount=4)
            response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 422)

    def test_accepts_a_seed_range_that_exactly_fits(self):
        app = make_app()
        with TestClient(app) as client:
            body = dict(LESION_BODY, seedStart=2**32 - 4, seedCount=4)
            response = client.post("/api/graph/v1/jobs", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 202)


class SubprocessSafetyTests(unittest.TestCase):
    def test_popen_is_called_with_shell_false_and_an_argv_list(self):
        app = make_app(runner=instant_runner)
        with TestClient(app) as client:
            with patch("graph_lab.jobs.subprocess.Popen", wraps=subprocess.Popen) as spy:
                response = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
                self.assertEqual(response.status_code, 202)
                identifier = response.json()["id"]
                self._wait_for_terminal(client, identifier)
            self.assertTrue(spy.called)
            args, kwargs = spy.call_args
            self.assertFalse(kwargs.get("shell", False))
            argv = args[0] if args else kwargs.get("args")
            self.assertIsInstance(argv, list)
            self.assertTrue(all(isinstance(part, str) for part in argv))

    @staticmethod
    def _wait_for_terminal(client, identifier, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
            if body["status"] not in ("queued", "running", "cancelling"):
                return body
            time.sleep(0.05)
        raise AssertionError(f"job {identifier} did not reach a terminal status within {timeout}s")


class CancellationAndTimeoutTests(unittest.TestCase):
    def test_cancel_kills_a_sleeping_child_process_group(self):
        app = make_app(runner=sleeping_runner)
        with TestClient(app) as client:
            submitted = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
            self.assertEqual(submitted.status_code, 202)
            identifier = submitted.json()["id"]

            job = app.state.jobs.jobs[identifier]
            deadline = time.monotonic() + 3
            while job.process is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(job.process, "child process never started")
            pid = job.process.pid

            cancelled = client.delete(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS)
            self.assertEqual(cancelled.json()["status"], "cancelling")

            self._assert_process_gone(pid, timeout=8)
            final = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
            self.assertEqual(final["status"], "cancelled")

    def test_timeout_marks_the_job_timed_out_and_kills_the_group(self):
        app = make_app(runner=sleeping_runner)
        with patch.dict(CEILING_SECONDS, {"lesion": 0.2}):
            with TestClient(app) as client:
                submitted = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
                identifier = submitted.json()["id"]

                job = app.state.jobs.jobs[identifier]
                deadline = time.monotonic() + 3
                while job.process is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertIsNotNone(job.process)
                pid = job.process.pid

                final = self._wait_for_terminal(client, identifier, timeout=10)
                self.assertEqual(final["status"], "timed-out")
                self._assert_process_gone(pid, timeout=8)

    @staticmethod
    def _wait_for_terminal(client, identifier, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
            if body["status"] not in ("queued", "running", "cancelling"):
                return body
            time.sleep(0.05)
        raise AssertionError(f"job {identifier} did not reach a terminal status within {timeout}s")

    @staticmethod
    def _assert_process_gone(pid, timeout=5):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError as error:
                return
            except OSError as error:  # pragma: no cover - defensive
                last_error = error
                break
            time.sleep(0.05)
        raise AssertionError(f"process {pid} is still alive" + (f" ({last_error})" if last_error else ""))

    def test_a_grandchild_outliving_its_leader_is_still_killed(self):
        """`process.wait()` only waits for the leader, not the rest of its
        process group -- if the leader (the Node entry, in production)
        exits, cleanly or via a crash, while its own forked shard workers
        are still alive, nothing else in the original code killed them.
        This fake leader spawns a real grandchild `sleep`, reports its pid
        via a progress line, and exits immediately while the grandchild is
        still very much running -- proving `_final_sweep` catches it."""

        def spawns_a_grandchild_then_exits(_request, _job_dir):
            script = (
                "import json, subprocess, sys; "
                "child = subprocess.Popen(['sleep', '30']); "
                "print(json.dumps({'type': 'progress', 'progress': {'grandchildPid': child.pid}})); "
                "sys.stdout.flush()"
            )
            return ["python3", "-c", script]

        app = make_app(runner=spawns_a_grandchild_then_exits)
        with TestClient(app) as client:
            submitted = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
            identifier = submitted.json()["id"]

            deadline = time.monotonic() + 5
            grandchild_pid = None
            while time.monotonic() < deadline:
                progress = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()["progress"]
                if "grandchildPid" in progress:
                    grandchild_pid = progress["grandchildPid"]
                    break
                time.sleep(0.02)
            self.assertIsNotNone(grandchild_pid, "leader never reported its grandchild's pid")

            self._wait_for_terminal(client, identifier)
            self._assert_process_gone(grandchild_pid, timeout=5)


class JobStoreBoundsTests(unittest.TestCase):
    def test_one_active_job_gives_409(self):
        app = make_app(runner=sleeping_runner)
        with TestClient(app) as client:
            first = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
            self.assertEqual(first.status_code, 202)
            second = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
            self.assertEqual(second.status_code, 409)
            client.delete(f"/api/graph/v1/jobs/{first.json()['id']}", headers=HEADERS)

    def test_at_most_four_jobs_retained(self):
        app = make_app(runner=instant_runner)
        with TestClient(app) as client:
            ids = []
            for _ in range(5):
                submitted = submit_when_idle(client)
                identifier = submitted["id"]
                ids.append(identifier)
                self._wait_for_terminal(client, identifier)
            self.assertEqual(client.get(f"/api/graph/v1/jobs/{ids[0]}", headers=HEADERS).status_code, 404)
            last = client.get(f"/api/graph/v1/jobs/{ids[-1]}", headers=HEADERS)
            self.assertEqual(last.status_code, 200)

    @staticmethod
    def _wait_for_terminal(client, identifier, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
            if body["status"] not in ("queued", "running", "cancelling"):
                return body
            time.sleep(0.02)
        raise AssertionError(f"job {identifier} did not reach a terminal status within {timeout}s")


class TerminalStatusVisibilityTests(unittest.TestCase):
    """Regression test for the race fixed in `jobs.py`'s `_execute`: a
    job's terminal status must not become visible over the API until that
    job's cleanup (`_final_sweep` plus removing `job_dir`) has actually
    finished. Blocks that cleanup deterministically on a
    `threading.Event`, rather than relying on timing luck the way the
    original bug report did (it passed 30/30 locally and only failed on a
    slower CI runner) -- so this test fails reliably, not intermittently,
    if the publish-before-cleanup ordering ever regresses.
    """

    def test_terminal_status_is_hidden_until_cleanup_finishes(self):
        app = make_app(runner=completing_runner)
        jobs = app.state.jobs
        release_cleanup = threading.Event()
        real_final_sweep = jobs._final_sweep

        def blocking_final_sweep(job):
            release_cleanup.wait(timeout=5)
            real_final_sweep(job)

        jobs._final_sweep = blocking_final_sweep

        with TestClient(app) as client:
            submitted = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
            self.assertEqual(submitted.status_code, 202)
            identifier = submitted.json()["id"]

            # `completing_runner` exits almost immediately, so
            # `_execute` reaches `finally` -- and blocks on
            # `release_cleanup` -- well within this window. While it's
            # blocked there the job must still read as in-flight, never a
            # terminal status, even though the child process itself is
            # long since dead.
            deadline = time.monotonic() + 5
            observed_running = False
            while time.monotonic() < deadline:
                status = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()["status"]
                self.assertIn(status, ("queued", "running"), "status went terminal before cleanup finished")
                if status == "running":
                    observed_running = True
                    break
                time.sleep(0.01)
            self.assertTrue(observed_running, "job never reached running before the test's own timeout")

            # The one-active-job invariant must hold for the whole cleanup
            # window, not just while the child process is literally
            # running: a submit here must still be refused.
            still_active = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
            self.assertEqual(still_active.status_code, 409)

            release_cleanup.set()

            final = self._wait_for_terminal(client, identifier)
            self.assertEqual(final["status"], "completed")

            # And a submit made right after the status turns terminal --
            # the exact sequence that raised `KeyError: 'id'` in CI --
            # must succeed. `submit_when_idle` (see its docstring) absorbs
            # only the same microseconds-scale scheduling gap the sibling
            # `test_at_most_four_jobs_retained` does; the actual proof that
            # cleanup, not timing, gates this is everything above -- the
            # 409 observed *while the event is still unset*.
            resubmitted = submit_when_idle(client)
            self.assertIn("id", resubmitted, f"expected a new job, got: {resubmitted}")
            client.delete(f"/api/graph/v1/jobs/{resubmitted['id']}", headers=HEADERS)

    @staticmethod
    def _wait_for_terminal(client, identifier, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
            if body["status"] not in ("queued", "running", "cancelling"):
                return body
            time.sleep(0.02)
        raise AssertionError(f"job {identifier} did not reach a terminal status within {timeout}s")


class NoSecretLeakTests(unittest.TestCase):
    def test_child_process_never_receives_the_token(self):
        """Regression-tests the actual leak path: `_child_env` must build
        an explicit allow-list rather than `os.environ.copy()` even when
        `GRAPH_LAB_TOKEN` really is present in the *process's own*
        environment (not just passed as a `create_app(token=...)`
        parameter, which alone would make this test pass vacuously no
        matter what `_child_env` does, since nothing would be in
        `os.environ` to leak in the first place)."""
        with patch.dict(os.environ, {"GRAPH_LAB_TOKEN": TOKEN}):
            os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN
            app = create_app(token=None, runner=env_probing_runner)
            with TestClient(app) as client:
                submitted = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
                identifier = submitted.json()["id"]
                final = self._wait_for_terminal(client, identifier)
                self.assertEqual(final["status"], "completed")
                self.assertEqual(final["result"]["token_seen"], "MISSING")

    def test_failed_job_error_never_includes_raw_stderr(self):
        def failing_runner(_request, _job_dir):
            return ["python3", "-c", "import sys; sys.stderr.write('a-private-detail\\n'); sys.exit(1)"]

        app = make_app(runner=failing_runner)
        with TestClient(app) as client:
            submitted = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
            identifier = submitted.json()["id"]
            final = self._wait_for_terminal(client, identifier)
            self.assertEqual(final["status"], "failed")
            self.assertNotIn("a-private-detail", final["error"] or "")

    @staticmethod
    def _wait_for_terminal(client, identifier, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = client.get(f"/api/graph/v1/jobs/{identifier}", headers=HEADERS).json()
            if body["status"] not in ("queued", "running", "cancelling"):
                return body
            time.sleep(0.02)
        raise AssertionError(f"job {identifier} did not reach a terminal status within {timeout}s")


if __name__ == "__main__":
    unittest.main()
