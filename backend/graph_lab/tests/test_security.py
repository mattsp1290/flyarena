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
        `disconnected` pass validation and dispatch (202) -- `rewired:*`
        passes the *same* validation (it is a real closed-set member, not
        rejected at 422) but its engine is not wired until WP2, so
        `default_runner` itself rejects it with a clean 501. A malformed id
        (see `GraphIdValidationTests`) never even reaches the runner."""
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as bundle_dir:
            manifest = Path(data_dir) / "malecns-arena-v1.manifest.json"
            manifest.write_text(json.dumps({"artifact": "malecns-arena-v1.bin.gz", "binarySha256": "0" * 64}))

            os.environ["GRAPH_LAB_ORIGINS"] = ALLOWED_ORIGIN
            for graph, expected_status in (
                ("biological", 202),
                ("disconnected", 202),
                ("rewired:0", 501),
                ("rewired:499", 501),
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
                submitted = client.post("/api/graph/v1/jobs", json=LESION_BODY, headers=HEADERS)
                identifier = submitted.json()["id"]
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
