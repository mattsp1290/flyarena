#!/usr/bin/env bash
# Build and publish a static release. Endpoints belong only in the ignored .env.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

die() { printf 'deploy: %s\n' "$*" >&2; exit 1; }

# Cleanup registered in $cleanup_paths runs once, on any exit (success,
# `die`, or an unexpected failure) -- WP2 of `.agents/plans/findings-tour`
# (`02-verified-redeploy.md`): "release the lock with a trap on EXIT,
# including on failure." A single trap function (rather than the narrower
# per-block traps this script used before) is required so the lock release
# survives every exit path, including ones that happen before or after the
# temp-file cleanup blocks further down. See scripts/deploy-trap.sh (also
# sourced directly by scripts/verify/deploy-lock.test.sh, so that test
# exercises this exact trap rather than a hand-maintained duplicate).
# shellcheck source=scripts/deploy-trap.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/deploy-trap.sh"

mode=${1:---deploy}
[[ $# -le 1 ]] || die 'Expected at most one option.'
case "$mode" in
  --package-backend)
    mkdir -p dist
    tar -czf dist/flyarena-backend.tar.gz \
      backend/Dockerfile backend/pyproject.toml \
      backend/flyarena_lab/__init__.py backend/flyarena_lab/cli.py \
      backend/flyarena_lab/experiment.py backend/flyarena_lab/model.py \
      backend/flyarena_lab/service.py scripts/lab.sh
    printf 'Packaged dist/flyarena-backend.tar.gz; no remote actions performed.\n'
    exit 0 ;;
  --deploy|--build-only) ;;
  --help) printf 'Usage: ./scripts/deploy.sh [--deploy|--build-only|--package-backend]\nBackend packaging is offline and needs no .env. Configure .env for static build/deploy. Default: build, upload, activate, verify.\n'; exit 0 ;;
  *) die 'Unknown option; use --help.' ;;
esac
[[ -f .env ]] || die 'Copy .env.example to .env and configure deployment.'
# This is a trusted local configuration file, not input from the server.
source .env
base=${DEPLOY_BASE:-/fly/}
[[ "$base" =~ ^/([a-zA-Z0-9_-]+/)+$ ]] || die 'DEPLOY_BASE must be an absolute URL path ending in /.'
# Shell functions that lazy-load nvm are not inherited by this Bash process.
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    source "$NVM_DIR/nvm.sh" --no-use
    nvm use --silent 22 || die 'Install Node 22 with `nvm install 22`, then retry.'
  fi
fi
for command in node npm; do
  command -v "$command" >/dev/null || die "Missing command: $command. Install Node.js 22 (at least 22.22.2) and npm, or run nvm install 22."
done
command -v tar >/dev/null || die 'Missing command: tar'

if [[ "$mode" == --deploy ]]; then
  # (thermo review, Critical) Must be the very first thing this branch does,
  # before any other check: see scripts/deploy-trap.sh's own doc comment on
  # deploy_refuse_test_overrides for why a leftover test-only override must
  # never be allowed to silently reach a real deploy.
  deploy_refuse_test_overrides
  for command in ssh scp curl cmp; do
    command -v "$command" >/dev/null || die "Missing command: $command"
  done
  [[ ${DEPLOY_SSH:-} =~ ^[a-zA-Z0-9_-]+@[a-zA-Z0-9.-]+$ ]] || die 'Set DEPLOY_SSH to user@host in .env.'
  [[ ${DEPLOY_ROOT:-} =~ ^/(srv|var/www|opt|home)/[a-zA-Z0-9_/-]+$ ]] || die 'Set DEPLOY_ROOT to a dedicated app directory below /srv, /var/www, /opt, or /home.'
  [[ "$DEPLOY_ROOT" != *//* && "$DEPLOY_ROOT" != */ ]] || die 'DEPLOY_ROOT must not contain // or end in /.'
  [[ "${DEPLOY_ROOT##*/}" == flyarena ]] || die 'Use a dedicated DEPLOY_ROOT ending in /flyarena.'
  [[ ${DEPLOY_URL:-} =~ ^https?://[^/[:space:]]+${base%/}/?$ ]] || die 'DEPLOY_URL must match DEPLOY_BASE.'
  ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
  # Refuse to replace an existing directory or another application's symlink.
  ssh "${ssh_options[@]}" "$DEPLOY_SSH" "bash -s -- '$DEPLOY_ROOT'" <<'REMOTE'
set -euo pipefail
root=$1
if [[ -e "$root/current" || -L "$root/current" ]]; then
  [[ -L "$root/current" && $(readlink "$root/current") == releases/* ]] || {
    echo 'deploy: current must be a managed releases/ symlink.' >&2; exit 1;
  }
fi
mkdir -p "$root/releases"
test -w "$root/releases"
REMOTE

  # The release id is decided now (before the build), not after it, so the
  # lock's owner file can record which deploy this run is, and so the same
  # id is later reused, unchanged, for the remote release directory and for
  # matching `current` on rollback.
  release="$(date -u +%Y%m%dT%H%M%SZ)-$(node -e 'console.log(require("node:crypto").randomBytes(6).toString("hex"))')"

  # In-script, stale-aware concurrency guard (`02-verified-redeploy.md` step
  # 2): no caller can bypass this because it lives inside deploy.sh itself,
  # after the SSH preflight above. See scripts/deploy-lock.sh for the lock,
  # marker-check, and rollback implementation (sourced, not executed, so it
  # shares this shell's `set -euo pipefail`, `die`, `ssh_options`, and the
  # `release` id just generated). (thermo review, Suggestion S2/maintainability)
  # Sourcing this also silently overrides scripts/deploy-trap.sh's no-op
  # `deploy_lock_release()` stub with the real implementation below.
  # shellcheck source=scripts/deploy-lock.sh
  source "$(dirname -- "${BASH_SOURCE[0]}")/deploy-lock.sh"
  deploy_lock_acquire "$release"
  # Record the pre-deploy active release for rollback (step 3): the live
  # `current` target, which this same call also confirms equals the last
  # `Release:` marker in .agents/deployment.md -- otherwise an unrecorded
  # deploy could silently become this run's rollback target.
  previous_release=$(deploy_check_release_marker)
fi

# Never export deployment settings to Vite; only the public base path is needed.
npm ci --no-audit --no-fund
npm run check
npm run build -- --base "$base"
[[ -f dist/index.html ]] || die 'Build did not produce dist/index.html.'
# Create outside dist: adding the archive there while tar reads '.' changes its
# directory metadata and can make GNU tar fail before upload.
archive=$(mktemp)
cleanup_paths+=("$archive")
tar -czf "$archive" --exclude=flyarena.tar.gz -C dist .
mv -- "$archive" dist/flyarena.tar.gz
printf 'Built dist/ and dist/flyarena.tar.gz for %s\n' "$base"
[[ "$mode" == --deploy ]] || exit 0

# $release was already decided above (before the build), so the lock's
# owner file could record it.
remote_release="$DEPLOY_ROOT/releases/$release"
ssh "${ssh_options[@]}" "$DEPLOY_SSH" "mkdir '$remote_release'"
scp "${ssh_options[@]}" dist/flyarena.tar.gz "$DEPLOY_SSH:$remote_release/package.tar.gz"
ssh "${ssh_options[@]}" "$DEPLOY_SSH" "bash -s -- '$DEPLOY_ROOT' '$release'" <<'REMOTE'
set -euo pipefail
root=$1
release=$2
tar -xzf "$root/releases/$release/package.tar.gz" -C "$root/releases/$release"
test -s "$root/releases/$release/index.html"
rm -- "$root/releases/$release/package.tar.gz"
chmod -R a+rX "$root/releases/$release"
ln -s "releases/$release" "$root/.current-$release"
mv -Tf "$root/.current-$release" "$root/current"
REMOTE

# Compare public HTML and every emitted asset with the exact local build.
# Unlike the earlier `die`-on-mismatch version, a failure here does not exit
# immediately: it returns nonzero so the caller can roll back while still
# holding the lock (`02-verified-redeploy.md` step 7).
verify_dir=$(mktemp -d)
cleanup_paths+=("$verify_dir")
# (thermo review, Important I4/ops-safety) `--show-error` deliberately
# omitted on every curl call below (here and in report_failure_and_roll_back):
# it prints curl's own native diagnostic -- which includes the request
# target -- to stderr on a real connection failure, independent of and in
# addition to this script's own fixed, secret-free failure messages below.
# `--silent` alone suppresses both the progress meter and that diagnostic;
# every failure path here already has its own message, so nothing is lost.
verify_assets() {
  curl --fail --silent --location --connect-timeout 10 --max-time 60 \
    -H 'Cache-Control: no-cache' "${DEPLOY_URL%/}" -o "$verify_dir/response" || {
    printf 'deploy: public entry URL request failed.\n' >&2
    return 1
  }
  cmp -s dist/index.html "$verify_dir/response" || {
    printf 'deploy: public entry URL does not serve this release.\n' >&2
    return 1
  }
  while IFS= read -r -d '' file; do
    relative=${file#dist/}
    url="${DEPLOY_URL%/}/$relative"
    curl --fail --silent --location --connect-timeout 10 --max-time 60 \
      -H 'Cache-Control: no-cache' "$url" -o "$verify_dir/response" || {
      printf 'deploy: public content request failed: %s\n' "$relative" >&2
      return 1
    }
    cmp -s "$file" "$verify_dir/response" || {
      printf 'deploy: public content mismatch: %s\n' "$relative" >&2
      return 1
    }
  done < <(find dist -type f ! -name flyarena.tar.gz -print0)
  return 0
}

# Live smoke check (`02-verified-redeploy.md` step 5): a small
# Playwright/Chromium script that opens $DEPLOY_URL, waits for ready,
# expands Findings, and checks step 1's sentence and the ledger render.
# Requires devDependencies already installed by `npm ci` above (tsx,
# @playwright/test) plus a Chromium browser (`npx playwright install
# chromium`) on the machine running deploy.sh -- not installed here, since
# that is a one-time, potentially large download better left to the
# operator/CI image setup than to every deploy run.
run_live_smoke() {
  local bin="$PWD/node_modules/.bin/tsx"
  if [[ ! -x "$bin" ]]; then
    printf 'deploy: node_modules/.bin/tsx not found (expected after npm ci); cannot run the live smoke check.\n' >&2
    return 1
  fi
  DEPLOY_URL="$DEPLOY_URL" "$bin" scripts/verify/live-smoke.ts
}

# On verification or smoke failure: roll back only if `current` still
# equals this run's release (still holding the lock), then re-verify with
# an HTTP 200 on the entry URL plus the smoke check, best-effort.
report_failure_and_roll_back() {
  local reason=$1
  printf 'deploy: verification failed (%s). Attempting rollback to previous release %s.\n' "$reason" "$previous_release" >&2
  if ! deploy_rollback "$release" "$previous_release"; then
    printf 'deploy: rollback did not complete. Investigate manually; do not assume current is healthy.\n' >&2
    return
  fi
  printf 'deploy: rolled back current -> %s.\n' "$previous_release" >&2
  if curl --fail --silent --location --connect-timeout 10 --max-time 60 \
    -H 'Cache-Control: no-cache' "${DEPLOY_URL%/}" -o /dev/null; then
    printf 'deploy: post-rollback check: entry URL returns HTTP 200.\n' >&2
  else
    printf 'deploy: post-rollback check FAILED: entry URL did not return HTTP 200 after rollback. Investigate immediately.\n' >&2
  fi
  if run_live_smoke; then
    printf 'deploy: post-rollback smoke check passed.\n' >&2
  else
    printf 'deploy: post-rollback smoke check did not pass (or could not run); the HTTP 200 check above is authoritative for rollback health, this is best-effort.\n' >&2
  fi
}

if ! verify_assets; then
  report_failure_and_roll_back 'byte-for-byte asset verification failed'
  die 'Deployment failed verification; the previous release was restored where possible. See rollback output above. Leave this bean open and report.'
fi
if ! run_live_smoke; then
  report_failure_and_roll_back 'live smoke check failed'
  die 'Deployment failed the live smoke check; the previous release was restored where possible. See rollback output above. Leave this bean open and report.'
fi
printf 'Deployment verified. Release: %s\nThe static arena/workbench require no backend. Optional DGX sandbox placement is BACKEND_SSH in .env.\n' "$release"
