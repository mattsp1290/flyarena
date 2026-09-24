# Verified ARM64 GB10 runtime. Reuses an existing local image; no running LLM service is used.
FROM nvcr.io/nvidia/vllm@sha256:98494fdaea68ad892c107180b0be41c99caa3c77683175becc4e202b2cb75b27
WORKDIR /app
COPY pyproject.toml ./
COPY flyarena_lab ./flyarena_lab
RUN python -m pip install --no-cache-dir --no-deps .
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=/tmp/torch-cache TRITON_CACHE_DIR=/tmp/triton-cache DD_TRACE_ENABLED=false DD_CIVISIBILITY_ENABLED=false DD_INSTRUMENTATION_TELEMETRY_ENABLED=false
USER 65532:65532
ENTRYPOINT ["python", "-m", "uvicorn", "flyarena_lab.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "35"]
