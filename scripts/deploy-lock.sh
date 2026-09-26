# Sourced library (not executed directly): `scripts/deploy.sh`'s stale-aware
# deploy lock, `Release:` marker check, and post-verification rollback.
#
# `.agents/plans/findings-tour/02-verified-redeploy.md` (WP2): the lock and
# marker check live *inside* `deploy.sh` itself so no caller can bypass them.
# This file is sourced by `deploy.sh` (not run as a separate process), so it
# shares `deploy.sh`'s `set -euo pipefail`, its `die()` helper, and the
# `ssh_options`/`DEPLOY_SSH`/`DEPLOY_ROOT`/`release`/`previous_release`
# variables `deploy.sh` already has in scope -- and
# `scripts/verify/deploy-lock.test.sh` sources it the same way, standing in
# its own `die`/`ssh_options`, to exercise the exact same functions without
# needing `deploy.sh`'s other preflight requirements (a real `.env`, node/npm,
# or a real SSH destination).
#
# `DEPLOY_LOCK_ROOT_OVERRIDE`, when set, makes every "remote" operation below
# a local filesystem operation against that directory instead of an ssh call
# to `$DEPLOY_SSH` -- the test hook `deploy-lock.test.sh` uses to run
# entirely locally, with no SSH and no real `DEPLOY_ROOT`.

DEPLOY_LOCK_STALE_SECONDS=1800
DEPLOY_LOCK_HELD=0

# The directory operations below actually target: the override in tests,
# `DEPLOY_ROOT` in production.
_deploy_lock_root() {
  printf '%s' "${DEPLOY_LOCK_ROOT_OVERRIDE:-$DEPLOY_ROOT}"
}

# Runs a small, literal (single-quoted heredoc source) bash script against
# the lock host: locally when `DEPLOY_LOCK_ROOT_OVERRIDE` is set, otherwise
# over ssh against `$DEPLOY_SSH` using deploy.sh's own `$ssh_options` --
# "the same SSH mechanism deploy.sh already uses" (same `bash -s --` pattern
# as the preflight/activation remote calls in deploy.sh).
_deploy_lock_run() {
  local script=$1
  shift
  if [[ -n "${DEPLOY_LOCK_ROOT_OVERRIDE:-}" ]]; then
    bash -c "$script" _deploy_lock_run "$@"
  else
    local quoted
    quoted=$(printf '%q ' "$@")
    ssh "${ssh_options[@]}" "$DEPLOY_SSH" "bash -s -- $quoted" <<SCRIPT
$script
SCRIPT
  fi
}

# (dual review, Important) The first line of stdout is always one of
# ACQUIRED / HELD / ERROR, so `deploy_lock_acquire` below can tell "the lock
# is genuinely held by someone else" (HELD, with the owner file on the
# following lines) apart from "this run's own `mkdir -- \"\$lock\"` failed
# for an unrelated reason" (ERROR -- permission denied, a read-only or full
# filesystem, a bad path) -- without this, both looked identical (a nonzero
# exit with no usable output) and an infrastructure failure was reported to
# the operator as a fabricated "stale lock," with wrong recovery advice.
_DEPLOY_LOCK_ACQUIRE_SCRIPT='
set -euo pipefail
root=$1; release=$2; ts=$3; host=$4; pid=$5
lock="$root/.deploy.lock"
owner_file="$root/.deploy.lock.owner"
mkdir_err=$(mktemp)
trap "rm -f -- \"\$mkdir_err\"" EXIT
if mkdir -- "$lock" 2>"$mkdir_err"; then
  {
    printf "timestamp=%s\n" "$ts"
    printf "hostname=%s\n" "$host"
    printf "pid=%s\n" "$pid"
    printf "release=%s\n" "$release"
  } > "$owner_file"
  printf "ACQUIRED\n"
  exit 0
fi
if grep -qi "file exists" -- "$mkdir_err"; then
  printf "HELD\n"
  if [[ -f "$owner_file" ]]; then
    cat -- "$owner_file"
  else
    printf "timestamp=0\n"
  fi
  exit 1
fi
printf "ERROR\n" >&2
printf "deploy: lock host mkdir failed (not lock contention):\n" >&2
cat -- "$mkdir_err" >&2
exit 2
'

_DEPLOY_LOCK_RELEASE_SCRIPT='
set -euo pipefail
root=$1
rm -f -- "$root/.deploy.lock.owner"
rmdir -- "$root/.deploy.lock" 2>/dev/null || true
'

# (dual review, Important) Read-only: the current owner file content, or
# nothing if it does not exist. Used by `deploy_lock_release` to confirm
# this run still owns the lock before deleting anything -- see that
# function's own doc comment.
_DEPLOY_LOCK_READ_OWNER_SCRIPT='
set -euo pipefail
root=$1
if [[ -f "$root/.deploy.lock.owner" ]]; then
  cat -- "$root/.deploy.lock.owner"
fi
'

_DEPLOY_LOCK_CURRENT_SCRIPT='
set -euo pipefail
root=$1
if [[ -L "$root/current" ]]; then
  readlink -- "$root/current"
fi
'

# (dual review, Important) `rm -f` the fixed temp name first: only one
# deploy.sh process can ever reach this script while holding the deploy
# lock, so this is not a race, only idempotency -- without it, a rollback
# interrupted between `ln -s` and `mv -Tf` (a dropped SSH connection, a
# killed process) leaves an orphaned `.rollback-current` that permanently
# blocks every later rollback attempt's own `ln -s` ("File exists"), unlike
# the activation step in deploy.sh, which sidesteps this by using a
# release-unique temp name (`.current-$release`) instead of a fixed one.
_DEPLOY_LOCK_ROLLBACK_SCRIPT='
set -euo pipefail
root=$1; previous=$2
rm -f -- "$root/.rollback-current"
ln -s -- "releases/$previous" "$root/.rollback-current"
mv -Tf -- "$root/.rollback-current" "$root/current"
'

# Atomically acquires the deploy lock (mkdir'd directory as the atomic
# primitive, a sibling `.deploy.lock.owner` file recording timestamp,
# hostname, pid, and release id) or dies:
#   - a fresh lock (owner timestamp < 30 minutes old): abort, no changes.
#   - a stale lock: abort, reported as stale, with the manual-clear command.
#     Never broken automatically.
#   - (dual review, Important) anything else -- SSH/connectivity failure,
#     permission denied, a full or read-only filesystem on the lock host --
#     is its own distinct die message, never misreported as a stale lock.
#     `_DEPLOY_LOCK_ACQUIRE_SCRIPT`'s first stdout line is the only signal
#     trusted for this: exactly "HELD" means genuine contention (with the
#     owner file on the following lines); anything else (including no
#     output at all, e.g. ssh itself never reached the remote host) means
#     the attempt did not actually determine the lock's state.
# Sets DEPLOY_LOCK_HELD=1 and DEPLOY_LOCK_ACQUIRED_RELEASE on success so
# `deploy_lock_release` (registered on deploy.sh's EXIT trap) knows it owns
# the lock and must release it, including on failure.
deploy_lock_acquire() {
  local release_id=$1
  local ts host pid output status root
  ts=$(date -u +%s)
  host=$(hostname 2>/dev/null || printf 'unknown-host')
  pid=$$
  root=$(_deploy_lock_root)
  set +e
  output=$(_deploy_lock_run "$_DEPLOY_LOCK_ACQUIRE_SCRIPT" "$root" "$release_id" "$ts" "$host" "$pid")
  status=$?
  set -e
  local first_line=${output%%$'\n'*}
  if [[ $status -eq 0 && "$first_line" == "ACQUIRED" ]]; then
    DEPLOY_LOCK_HELD=1
    DEPLOY_LOCK_ACQUIRED_RELEASE=$release_id
    return 0
  fi
  if [[ "$first_line" != "HELD" ]]; then
    die "Could not determine the deploy lock's state at $root/.deploy.lock: the lock check itself failed (SSH connectivity, permissions, or disk space on the deploy host -- see any error output above this line), not confirmed lock contention. Aborting with no changes."
  fi
  local owner=${output#*$'\n'}
  local owner_ts now age
  owner_ts=$(printf '%s\n' "$owner" | sed -n 's/^timestamp=//p' | head -n1)
  owner_ts=${owner_ts:-0}
  now=$(date -u +%s)
  age=$(( now - owner_ts ))
  if (( age < DEPLOY_LOCK_STALE_SECONDS )); then
    die "Deploy lock is held (age ${age}s, under the 1800s staleness threshold). Another deploy is in progress; aborting with no changes. Owner:
$owner"
  else
    die "Deploy lock at $root/.deploy.lock is STALE (age ${age}s). Refusing to break it automatically. Owner:
$owner
After confirming no deploy is actually running, clear it manually: ssh $DEPLOY_SSH \"rm -f '$root/.deploy.lock.owner'; rmdir '$root/.deploy.lock'\" (or, run directly on the host: rm -f '$root/.deploy.lock.owner'; rmdir '$root/.deploy.lock')."
  fi
}

# Releases the lock, best-effort, if this run holds it. Safe to call
# unconditionally from an EXIT trap (including on failure, and even if the
# lock was never acquired -- e.g. an earlier preflight `die`).
#
# (dual review, Important) Verifies the owner file still records *this
# run's* `DEPLOY_LOCK_ACQUIRED_RELEASE` before removing anything. The
# documented stale-lock manual-clear command is an operator escape hatch;
# if it is used against a lock this run still legitimately holds (operator
# error, or a race with this run's own trap), a *different* run may acquire
# it before this one exits -- this run's own trap must never delete a lock
# it no longer owns.
deploy_lock_release() {
  [[ "$DEPLOY_LOCK_HELD" == 1 ]] || return 0
  local root current_owner
  root=$(_deploy_lock_root)
  current_owner=$(_deploy_lock_run "$_DEPLOY_LOCK_READ_OWNER_SCRIPT" "$root" 2>/dev/null) || current_owner=""
  if [[ -n "$current_owner" ]] && ! grep -q "^release=${DEPLOY_LOCK_ACQUIRED_RELEASE}\$" <<<"$current_owner"; then
    printf 'deploy: warning: the deploy lock at %s/.deploy.lock is no longer owned by this run (release %s); NOT releasing it -- another deploy may hold it now.\n' \
      "$root" "$DEPLOY_LOCK_ACQUIRED_RELEASE" >&2
    DEPLOY_LOCK_HELD=0
    return 0
  fi
  if ! _deploy_lock_run "$_DEPLOY_LOCK_RELEASE_SCRIPT" "$root" >/dev/null; then
    printf 'deploy: warning: failed to release the deploy lock at %s/.deploy.lock; clear it manually once confirmed idle.\n' "$root" >&2
  fi
  DEPLOY_LOCK_HELD=0
}

# The release id the live `current` symlink currently targets (basename
# under `releases/`), or empty if `current` does not exist / is not a
# managed symlink.
deploy_current_release() {
  local root target
  root=$(_deploy_lock_root)
  target=$(_deploy_lock_run "$_DEPLOY_LOCK_CURRENT_SCRIPT" "$root") || target=""
  printf '%s' "${target#releases/}"
}

# The release id in the last `Release:` marker line of `.agents/deployment.md`
# (always the local, git-tracked file -- never remote). Empty if the file or
# marker is missing.
deploy_last_release_marker() {
  local doc=${DEPLOY_DEPLOYMENT_DOC:-.agents/deployment.md}
  [[ -f "$doc" ]] || { printf ''; return 0; }
  # `|| true`: under `set -o pipefail`, grep finding no marker at all (exit 1)
  # would otherwise propagate as this pipeline's exit status and, because
  # the whole thing is the value side of `marker=$(deploy_last_release_marker)`,
  # trip the caller's `set -e` -- a legitimate "no marker yet" case must
  # return an empty string, not abort the script here.
  grep -oE '^Release: [^[:space:]]+' -- "$doc" | tail -n1 | sed -E 's/^Release: //' || true
}

# Confirms the live `current` target equals the last recorded `Release:`
# marker (an unrecorded deploy would otherwise silently become this run's
# rollback target). On success, echoes that release id -- also the pre-deploy
# active release to record for rollback (`02-verified-redeploy.md` step 3).
# Aborts on any mismatch or missing marker.
deploy_check_release_marker() {
  local live marker
  live=$(deploy_current_release)
  marker=$(deploy_last_release_marker)
  if [[ -z "$marker" ]]; then
    die "No 'Release:' marker found in .agents/deployment.md -- cannot confirm the live release is the last recorded one. Backfill a marker for the current live release, then retry."
  fi
  if [[ "$live" != "$marker" ]]; then
    die "Live current ($live) does not match the last recorded Release: marker ($marker) in .agents/deployment.md -- an unrecorded deploy happened. Aborting with no changes; investigate before deploying."
  fi
  printf '%s' "$live"
}

# Rolls the live `current` symlink back to `$previous_release`, via the
# documented `ln -s releases/<previous> .rollback-current && mv -Tf
# .rollback-current current`, but only if `current` still equals
# `$this_release` (the release this run published) -- if another deploy has
# happened, refuses and reports rather than touching it. Must be called
# while still holding the lock. Returns nonzero if refused or if the
# rollback commands themselves fail.
deploy_rollback() {
  local this_release=$1 previous_release=$2
  local live root
  live=$(deploy_current_release)
  if [[ "$live" != "$this_release" ]]; then
    printf 'deploy: rollback refused -- current is "%s", not the release this run published ("%s"). Another deploy has happened; leaving it untouched. Investigate manually.\n' "$live" "$this_release" >&2
    return 1
  fi
  root=$(_deploy_lock_root)
  if [[ -z "$previous_release" ]]; then
    printf 'deploy: rollback refused -- no previous release was recorded for this run.\n' >&2
    return 1
  fi
  _deploy_lock_run "$_DEPLOY_LOCK_ROLLBACK_SCRIPT" "$root" "$previous_release"
}
