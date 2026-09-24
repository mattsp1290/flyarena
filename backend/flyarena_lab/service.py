"""Single-operator, ephemeral job API. One process, one bounded compute worker."""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import os
import threading
import uuid
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from .experiment import Options, run, Cancelled


@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self):
        return {"id": self.id, "status": self.status, "progress": self.progress.copy(),
                "result": self.result, "error": self.error}


class Jobs:
    def __init__(self, engine=run):
        self.engine = engine
        self.lock = threading.Lock()
        self.jobs = {}
        self.thread = None
        self.closing = False

    def submit(self, options):
        with self.lock:
            if self.closing or (self.thread and self.thread.is_alive()):
                raise HTTPException(409, "A job is already active or the service is stopping")
            while len(self.jobs) >= 4:
                del self.jobs[next(iter(self.jobs))]
            job = Job(uuid.uuid4().hex)
            self.jobs[job.id] = job
            self.thread = threading.Thread(target=self.execute, args=(job, options), daemon=False)
            self.thread.start()
            return {"id": job.id, "status": job.status}

    def execute(self, job, options):
        def progress(value):
            with self.lock:
                job.progress = value.copy()
        try:
            with self.lock:
                job.status = "running"
            result = self.engine(options, progress, job.cancel)
            with self.lock:
                if job.cancel.is_set():
                    job.status = "cancelled"
                else:
                    job.result = result
                    job.status = "completed"
        except Cancelled:
            with self.lock:
                job.status = "cancelled"
        except Exception:
            import logging
            logging.exception("Experiment failed")
            with self.lock:
                job.status = "cancelled" if job.cancel.is_set() else "failed"
                job.error = "Experiment failed; check backend logs and device availability"

    def get(self, identifier, cancel=False):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise HTTPException(404, "Unknown or expired job")
            if cancel and job.status in ("queued", "running", "cancelling"):
                job.cancel.set()
                job.status = "cancelling"
            return job.snapshot()

    def close(self):
        with self.lock:
            self.closing = True
            for job in self.jobs.values():
                job.cancel.set()
        if self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("Worker did not stop within 30 seconds")


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > 4096:
                return await JSONResponse({"detail": "Request body exceeds 4096 bytes"}, 413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        async def buffered():
            if chunks:
                return chunks.pop(0)
            return await receive()
        await self.app(scope, buffered, send)


def create_app(token=None, engine=run):
    secret = token if token is not None else os.environ.get("LAB_TOKEN", "")
    jobs = Jobs(engine)

    @asynccontextmanager
    async def lifespan(app):
        if len(secret) < 16:
            raise RuntimeError("Set LAB_TOKEN to at least 16 characters")
        yield
        jobs.close()

    app = FastAPI(title="FlyArena authored circuit lab", lifespan=lifespan)
    app.state.jobs = jobs
    app.add_middleware(BodyLimit)
    origins = os.environ.get("LAB_ORIGINS", "http://127.0.0.1:5173,http://127.0.0.1:4173").split(",")
    if "*" in origins:
        raise ValueError("LAB_ORIGINS must list explicit origins")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Authorization", "Content-Type"])

    def authorize(authorization: str = Header(default="")):
        if not secret or not hmac.compare_digest(authorization.encode(), f"Bearer {secret}".encode()):
            raise HTTPException(401, "A valid bearer token is required")

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "model": "synthetic-foraging-v1"}

    @app.post("/api/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    def submit(options: Options):
        return jobs.submit(options)

    @app.get("/api/v1/jobs/{identifier}", dependencies=[Depends(authorize)])
    def status(identifier: str):
        return jobs.get(identifier)

    @app.delete("/api/v1/jobs/{identifier}", dependencies=[Depends(authorize)])
    def cancel(identifier: str):
        return jobs.get(identifier, cancel=True)

    return app
