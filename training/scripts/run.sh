#!/usr/bin/env bash
# Wrapper that disables this host's global Datadog APM auto-injection (which
# breaks `torch` — see training/README.md's "Environment note") and execs
# `uv run "$@"` from training/, so WP3's future `flyarena-train` CLI and any
# other automation (a cron job, a CI runner) can invoke one command instead
# of relying on the three DD_* env vars being copy-pasted correctly by hand
# every time. Harmless no-op on a host without the injection.
#
# Usage: training/scripts/run.sh pytest -v
#        training/scripts/run.sh python -m flyarena_training.cli ...
set -euo pipefail

export DD_TRACE_ENABLED=false
export DD_IAST_ENABLED=false
export DD_APPSEC_ENABLED=false

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run "$@"
