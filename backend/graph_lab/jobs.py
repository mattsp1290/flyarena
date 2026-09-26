"""Job store with subprocess supervision.

Modeled on `flyarena_lab/service.py`'s `Jobs` (one active job, at most 4
retained, a background thread per job), but the compute itself always runs
in a **child process** rather than in-process: `runner(request, job_dir)`
returns an argv list, and this module owns everything about running it
safely -- `subprocess.Popen(argv, shell=False, ...)` (never a shell, never
a string command), a fresh process group per job (`start_new_session=True`)
so cancellation and timeout can kill the whole tree with one `os.killpg`,
an explicit minimal child environment (never `os.environ` wholesale, so a
child can never inherit `GRAPH_LAB_TOKEN` or any other secret the parent
process holds), and a wall-clock ceiling per job kind
(`models.CEILING_SECONDS`).

Progress/result protocol: the child prints newline-delimited JSON objects
to stdout. `{"type": "progress", "progress": {...}}` updates the job's
live progress; `{"type": "result", "result": {...}}` is the final answer;
`{"type": "error", "message": "..."}` is a clean, structured failure. A
line that isn't valid JSON, or that doesn't match one of these shapes, is
ignored rather than crashing the monitor thread -- the child's own exit
code and stderr are still available for the failure path, but nothing from
stderr is ever surfaced verbatim in a job's `error` field (see `_execute`'s
own comment) since stderr can contain interpreter tracebacks that quote
argv, and argv can carry data derived from the request body.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from fastapi import HTTPException

from .models import CEILING_SECONDS, GRAPH_PATTERN, MAX_REWIRED_SEED, JobRequest

Runner = Callable[[object, Path], Sequence[str]]

MAX_RETAINED_JOBS = 4
KILL_GRACE_SECONDS = 5
POLL_INTERVAL_SECONDS = 0.2
# Truncate any string surfaced in a job's `error` field or the process
# reader's captured stderr tail -- neither is ever unbounded, so a runaway
# child can't grow a retained job's memory footprint by printing endlessly.
MAX_ERROR_CHARS = 500


def parse_graph_id(graph: str) -> tuple[str, int | None]:
    """Re-validate `graph` independently of Pydantic having already run --
    this is the only function `jobs.py` itself trusts to turn a `graph`
    string into `(mode, seed)`; nothing else in this module ever slices or
    concatenates the raw string into a path or argv entry. Raises
    `ValueError` (never returns a partially-validated value) for anything
    that isn't an exact match, including a value a looser check might have
    let through, such as extra whitespace or a second `rewired:` prefix."""
    match = GRAPH_PATTERN.match(graph)
    if not match:
        raise ValueError(f"unrecognized graph id: {graph!r}")
    seed_text = match.group(2)
    if seed_text is None:
        return graph, None
    seed = int(seed_text)
    if seed > MAX_REWIRED_SEED:
        raise ValueError(f"rewired seed {seed} exceeds {MAX_REWIRED_SEED}")
    return "rewired", seed


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    process: subprocess.Popen | None = None
    pgid: int | None = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress.copy(),
            "result": self.result,
            "error": self.error,
        }


class Jobs:
    def __init__(self, runner: Runner, scratch_dir: str | None = None):
        self.runner = runner
        self.scratch_dir = scratch_dir
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.thread: threading.Thread | None = None
        self.closing = False

    def submit(self, request: JobRequest) -> dict:
        # The whole check-build-start sequence runs under one lock
        # acquisition (matching `flyarena_lab/service.py`'s `Jobs.submit`),
        # so two concurrent submits can't both pass the "one active job"
        # check. `self.runner(request, job_dir)` runs here -- synchronously,
        # before any 202 is returned -- rather than inside the background
        # thread: a runner that rejects a request (e.g. `default_runner`'s
        # 501 for a job kind/graph its WP1 engine doesn't cover yet) must
        # surface as a real HTTP response on the `POST`, not as an opaque
        # "failed" job discovered later by polling -- `HTTPException`
        # raised from inside a background thread would never reach the
        # client at all.
        with self.lock:
            if self.closing or (self.thread and self.thread.is_alive()):
                raise HTTPException(409, "A job is already active or the service is stopping")

            job_dir = Path(tempfile.mkdtemp(prefix="graph-lab-job-", dir=self.scratch_dir))
            try:
                argv = list(self.runner(request, job_dir))
                if not argv or not all(isinstance(part, str) for part in argv):
                    raise RuntimeError("runner produced an invalid argv")
            except HTTPException:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise
            except Exception as error:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(500, "Failed to prepare job") from error

            while len(self.jobs) >= MAX_RETAINED_JOBS:
                del self.jobs[next(iter(self.jobs))]
            job = Job(id=uuid.uuid4().hex, kind=request.kind)
            self.jobs[job.id] = job
            self.thread = threading.Thread(target=self._execute, args=(job, argv, job_dir), daemon=False)
            self.thread.start()
            return {"id": job.id, "status": job.status}

    def get(self, identifier: str, cancel: bool = False) -> dict:
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise HTTPException(404, "Unknown or expired job")
            should_kill = False
            if cancel and job.status in ("queued", "running"):
                job.status = "cancelling"
                should_kill = True
            snapshot = job.snapshot()
        if should_kill:
            self._kill_group(job)
        return snapshot

    def close(self) -> None:
        with self.lock:
            self.closing = True
            active = [job for job in self.jobs.values() if job.status in ("queued", "running")]
            for job in active:
                job.status = "cancelling"
        for job in active:
            self._kill_group(job)
        if self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("Worker did not stop within 30 seconds")

    # -- internals --------------------------------------------------------

    @staticmethod
    def _child_env(extra: dict[str, str]) -> dict[str, str]:
        """An explicit allow-list, never `os.environ.copy()`: the parent
        process may hold `GRAPH_LAB_TOKEN` (read once at startup into a
        local variable -- see `service.py`), and a child built from
        arbitrary request-derived argv should never be able to observe it,
        whether by design or by an unrelated future bug that logs its own
        environment. `NODE_OPTIONS` is explicitly cleared rather than
        omitted, so a host-level `--require` injection (a tracing agent,
        for example) already set in the parent's environment is not
        silently inherited by every job's Node child either.

        `DD_IAST_ENABLED`/`DD_TRACE_ENABLED`/`PYTHONUNBUFFERED` are
        included in the base allow-list (not only in a per-invocation
        `extra`) even though WP1's only wired kind (`lesion`) launches a
        Node child that doesn't read them: a Datadog host-level injection
        (confirmed active in dev/CI sandboxes akin to this one, via
        `/etc/ld.so.preload`, independent of this process's own
        `DD_TRACE_ENABLED=false`) breaks `flyarena_training`'s tensor ops
        under IAST (reproduced running the GPU atlas smoke test -- see the
        Dockerfile's own note), so a future WP2 python child using this
        same `_child_env` should not have to remember to add these itself.
        None of these are secrets."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": "/tmp",
            "NODE_OPTIONS": "",
            "PYTHONUNBUFFERED": "1",
            "DD_IAST_ENABLED": "false",
            "DD_TRACE_ENABLED": "false",
        }
        env.update(extra)
        return env

    def _execute(self, job: Job, argv: list[str], job_dir: Path) -> None:
        try:
            with self.lock:
                if job.status == "cancelling":
                    job.status = "cancelled"
                    return
                job.status = "running"

            process = subprocess.Popen(
                argv,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(job_dir),
                env=self._child_env(
                    {
                        "GRAPH_LAB_DATA_DIR": os.environ.get("GRAPH_LAB_DATA_DIR", ""),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torch-cache",
                        "TRITON_CACHE_DIR": "/tmp/triton-cache",
                    }
                ),
                text=True,
            )
            with self.lock:
                job.process = process
                try:
                    job.pgid = os.getpgid(process.pid)
                except ProcessLookupError:
                    job.pgid = None
                deadline = time.monotonic() + CEILING_SECONDS.get(job.kind, 20 * 60)
                # `get(..., cancel=True)` may have run between the
                # "cancelling? return before spawning" check above and this
                # point (it only needed `self.lock`, which this thread had
                # briefly released while `Popen` itself ran) -- re-check
                # under the same lock that just recorded `pgid`, so a
                # cancel landing in that narrow window still kills the
                # process it applies to instead of racing ahead unkilled.
                already_cancelling = job.status == "cancelling"
            if already_cancelling:
                self._kill_group(job)

            outcome: dict = {}
            stderr_tail: list[str] = []
            reader = threading.Thread(target=self._drain, args=(process, job, outcome, stderr_tail), daemon=True)
            reader.start()

            # `killing`/`give_up_deadline`: once either a cancel or a
            # timeout has fired `_kill_group` (SIGTERM, then SIGKILL after
            # `KILL_GRACE_SECONDS`), this loop still only learns the
            # process is actually gone via `process.wait()` succeeding --
            # with no bound on that wait, a process group SIGKILL somehow
            # can't reap (e.g. stuck in uninterruptible I/O) would loop
            # here forever, keeping `self.thread` alive and therefore
            # `submit()`'s "one active job" check permanently 409ing every
            # future request (an Important finding from review: "a job
            # that can't be killed blocks the service"). Giving up after a
            # bounded extra wait accepts a documented residual risk (a
            # truly unkillable process's resources leak until the
            # container itself restarts) in exchange for the service
            # itself staying usable for new jobs.
            killing = False
            give_up_deadline = 0.0
            while True:
                try:
                    process.wait(timeout=POLL_INTERVAL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if killing:
                        if time.monotonic() > give_up_deadline:
                            logging.error(
                                "graph-lab job %s (%s): process group %s did not exit after SIGKILL; "
                                "giving up waiting (it may still be running)",
                                job.id,
                                job.kind,
                                job.pgid,
                            )
                            break
                        continue
                    with self.lock:
                        cancelling = job.status == "cancelling"
                    if cancelling:
                        killing = True
                    elif time.monotonic() > deadline:
                        with self.lock:
                            job.status = "timed-out"
                        self._kill_group(job)
                        killing = True
                    if killing:
                        give_up_deadline = time.monotonic() + KILL_GRACE_SECONDS + 10
            reader.join(timeout=2)

            with self.lock:
                if job.status == "timed-out":
                    job.error = "Job exceeded its time limit and was terminated"
                elif job.status == "cancelling":
                    job.status = "cancelled"
                elif process.returncode == 0 and "result" in outcome:
                    job.result = outcome["result"]
                    job.status = "completed"
                else:
                    job.status = "failed"
                    # Never surface the child's own stderr in the API
                    # response (it can quote argv, which can carry
                    # request-derived data): only a structured
                    # `{"type": "error", ...}` message the child itself
                    # chose to report, or a generic fallback. The captured
                    # stderr tail still goes to the server's own log, for
                    # an operator to read from the container's logs -- it
                    # never reaches a client.
                    job.error = (outcome.get("error") or "Job process exited without a result")[:MAX_ERROR_CHARS]
                    if stderr_tail:
                        logging.error("graph-lab job %s (%s) stderr: %s", job.id, job.kind, "".join(stderr_tail))
        except Exception:
            logging.exception("graph-lab job failed")
            with self.lock:
                job.status = "failed"
                job.error = "Job failed; check backend logs"
        finally:
            # Unconditional final sweep, on every exit path (normal
            # completion, cancel, timeout, or the generic exception handler
            # above): `process.wait()` only waits for the *leader* --
            # if it exits (cleanly or via a crash) while its own forked
            # shard workers (`child_process.fork()` in the Node entries)
            # are still alive, those workers are never otherwise killed
            # and would keep computing, unbounded, after this job is
            # already marked done and a new submission has been accepted.
            # Confirmed reproducible in review before this was added.
            self._final_sweep(job)
            shutil.rmtree(job_dir, ignore_errors=True)

    def _final_sweep(self, job: Job) -> None:
        """SIGKILL `job.pgid` one more time, unconditionally. Idempotent
        and safe to call whether or not anything in the group is still
        alive: `killpg` on an already-empty or already-reaped group just
        raises `ProcessLookupError`, ignored here."""
        with self.lock:
            pgid = job.pgid
        if pgid is None:
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def _drain(process: subprocess.Popen, job: "Job", outcome: dict, stderr_tail: list[str]) -> None:
        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                message_type = message.get("type")
                if message_type == "progress" and isinstance(message.get("progress"), dict):
                    # A single attribute assignment (not a mutation of the
                    # existing dict), so a concurrent `Job.snapshot()` read
                    # from the API thread sees either the old or the new
                    # dict, never a torn one -- no separate lock needed for
                    # this specific write.
                    job.progress = message["progress"]
                elif message_type == "result":
                    outcome["result"] = message.get("result")
                elif message_type == "error":
                    outcome["error"] = str(message.get("message", "job failed"))[:MAX_ERROR_CHARS]

        def read_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                if len(stderr_tail) < 50:
                    stderr_tail.append(line)

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        stdout_thread.join()
        stderr_thread.join()

    def _kill_group(self, job: Job) -> None:
        with self.lock:
            pgid = job.pgid
        if pgid is None:
            return

        def killer() -> None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return
            time.sleep(KILL_GRACE_SECONDS)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        threading.Thread(target=killer, daemon=True).start()
