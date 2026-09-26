# shellcheck shell=bash
# (thermo review, Suggestion S2/ops-safety) this file has no shebang -- it is
# always sourced, never executed -- so shellcheck can't infer the target
# shell on its own (SC2148) without this directive.
#
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
    # (thermo review, Suggestion S4/ops-safety) `%q`'s round-trip quoting
    # guarantee is specifically a *bash* guarantee; the string is parsed
    # remotely by whatever shell sshd invokes for the deploy account's login
    # shell, not necessarily bash. Every value that actually flows through
    # here today is already constrained (a validated `DEPLOY_ROOT`, a
    # generated release id, numeric `ts`/`pid`, a `hostname` value), so this
    # is not exploitable currently -- noted for a future caller that might
    # pass a less-constrained value through this same helper.
    local quoted
    quoted=$(printf '%q ' "$@")
    # shellcheck disable=SC2154 # ssh_options is supplied by every caller (deploy.sh, deploy-lock.test.sh) -- see this file's own header comment.
    # shellcheck disable=SC2087 # intentional: this heredoc's body is the single token $script and IS meant to expand client-side, substituting in the actual script text before it reaches ssh's stdin -- quoting the delimiter would send the literal 7 characters "$script" instead.
    ssh "${ssh_options[@]}" "$DEPLOY_SSH" "bash -s -- $quoted" <<SCRIPT
$script
SCRIPT
  fi
}

# (thermo review, Important I2/ops-safety) Sentinel lines, not "the first
# line of stdout": a stray banner or profile-hook line some sshd
# configurations emit on a non-interactive `bash -s` session could otherwise
# land ahead of the real ACQUIRED/HELD/ERROR marker. `deploy_lock_acquire`
# below scans the *whole* output for a line matching one of these sentinels
# exactly, wherever it falls, rather than assuming position 1 -- this
# matters most for ACQUIRED: if banner noise made a genuinely successful
# mkdir+owner-write look unrecognized, `DEPLOY_LOCK_HELD` would never get
# set, and the lock this run actually holds on the remote host would never
# be released by the EXIT trap.
_DEPLOY_LOCK_SENTINEL_ACQUIRED='##DEPLOY-LOCK:ACQUIRED##'
_DEPLOY_LOCK_SENTINEL_HELD='##DEPLOY-LOCK:HELD##'

# (thermo review, Important I1/ops-safety) The owner file is written to a
# temp name inside $root and renamed into place -- never written in place --
# so a concurrent HELD reader can never observe "the lock directory exists
# but the owner file does not yet" (a `mkdir`-then-write TOCTOU window) and
# default to treating a lock that is genuinely milliseconds old as though it
# were fully absent. This narrows that window a great deal but a reader can
# still land inside it (or the owning run can die -- an SSH drop, `kill -9`
# -- between `mkdir` and the owner write, orphaning the lock with no owner
# file at all, permanently, unless something can still age it out).
#
# (thermo review follow-up, regression fix) When the owner file is missing,
# the fallback below is the lock *directory's own mtime* (`stat -c %Y`,
# read on the same host as the timestamp -- see `deploy_lock_acquire`'s own
# comment on why), not a fixed "indeterminate, retry" with no way out. This
# gives an owner-file-missing lock the same staleness math and the same
# manual-clear recovery path as a normal lock, via the `owner_missing=1`
# marker `deploy_lock_acquire` checks below -- so a crash exactly in this
# window still eventually reports STALE with the clear command, instead of
# blocking every future deploy forever with no recovery path at all.
_DEPLOY_LOCK_ACQUIRE_SCRIPT='
set -euo pipefail
root=$1; release=$2; ts=$3; host=$4; pid=$5
lock="$root/.deploy.lock"
owner_file="$root/.deploy.lock.owner"
mkdir_err=$(mktemp)
trap "rm -f -- \"\$mkdir_err\"" EXIT
if mkdir -- "$lock" 2>"$mkdir_err"; then
  owner_tmp=$(mktemp "$root/.deploy.lock.owner.XXXXXX")
  {
    printf "timestamp=%s\n" "$ts"
    printf "hostname=%s\n" "$host"
    printf "pid=%s\n" "$pid"
    printf "release=%s\n" "$release"
  } > "$owner_tmp"
  mv -- "$owner_tmp" "$owner_file"
  printf "SENTINEL_ACQUIRED\n"
  exit 0
fi
if grep -qi "file exists" -- "$mkdir_err"; then
  printf "SENTINEL_HELD\n"
  if [[ -f "$owner_file" ]]; then
    cat -- "$owner_file"
  else
    dir_mtime=$(stat -c %Y -- "$lock" 2>/dev/null || printf "0")
    printf "timestamp=%s\n" "$dir_mtime"
    printf "owner_missing=1\n"
  fi
  exit 1
fi
printf "deploy: lock host mkdir failed (not lock contention):\n" >&2
cat -- "$mkdir_err" >&2
exit 2
'
_DEPLOY_LOCK_ACQUIRE_SCRIPT=${_DEPLOY_LOCK_ACQUIRE_SCRIPT//SENTINEL_ACQUIRED/$_DEPLOY_LOCK_SENTINEL_ACQUIRED}
_DEPLOY_LOCK_ACQUIRE_SCRIPT=${_DEPLOY_LOCK_ACQUIRE_SCRIPT//SENTINEL_HELD/$_DEPLOY_LOCK_SENTINEL_HELD}

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
#   - the owner file is missing (the narrow TOCTOU window between `mkdir`
#     and the owner file's rename-into-place, or the owning run crashing
#     inside that window -- an SSH drop, `kill -9`): the lock directory's
#     own mtime stands in for the owner timestamp (see
#     `_DEPLOY_LOCK_ACQUIRE_SCRIPT`'s own comment), so this still resolves
#     to the normal fresh/stale math and, if stale, the same manual-clear
#     recovery path -- a crash in this exact window must never orphan the
#     lock forever with no way to recover it.
#   - anything else -- SSH/connectivity failure, permission denied, a full
#     or read-only filesystem on the lock host -- is its own distinct die
#     message, never misreported as a stale lock. The only signal trusted
#     for "genuine contention" is the `_DEPLOY_LOCK_SENTINEL_HELD` line
#     appearing *anywhere* in stdout (not assumed to be line 1 -- see that
#     sentinel's own comment on why); anything else (including no output at
#     all, e.g. ssh itself never reached the remote host) means the attempt
#     did not actually determine the lock's state.
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
  if [[ $status -eq 0 ]] && grep -qxF "$_DEPLOY_LOCK_SENTINEL_ACQUIRED" <<<"$output"; then
    DEPLOY_LOCK_HELD=1
    DEPLOY_LOCK_ACQUIRED_RELEASE=$release_id
    return 0
  fi
  if ! grep -qxF "$_DEPLOY_LOCK_SENTINEL_HELD" <<<"$output"; then
    die "Could not determine the deploy lock's state at $root/.deploy.lock: the lock check itself failed (SSH connectivity, permissions, or disk space on the deploy host -- see any error output above this line), not confirmed lock contention. Aborting with no changes."
  fi
  # Everything after the HELD sentinel line is the owner payload, wherever
  # that sentinel actually fell in the output (see the sentinel's comment).
  local owner
  owner=$(awk -v sentinel="$_DEPLOY_LOCK_SENTINEL_HELD" 'found{print} $0==sentinel{found=1}' <<<"$output")
  local owner_ts owner_missing now age manual_clear
  owner_ts=$(printf '%s\n' "$owner" | sed -n 's/^timestamp=//p' | head -n1)
  owner_missing=$(printf '%s\n' "$owner" | sed -n 's/^owner_missing=//p' | head -n1)
  if [[ -z "$owner_ts" || ! "$owner_ts" =~ ^[0-9]+$ ]]; then
    die "Deploy lock at $root/.deploy.lock exists, but its state could not be read at all (no usable timestamp, not even the lock directory's own mtime). Not confirmed stale -- investigate manually before clearing. Owner:
$owner"
  fi
  now=$(date -u +%s)
  age=$(( now - owner_ts ))
  manual_clear="After confirming no deploy is actually running, clear it manually, either directly on the host or over SSH to the configured deploy destination: rm -f '$root/.deploy.lock.owner'; rmdir '$root/.deploy.lock'"
  if [[ "$owner_missing" == "1" ]]; then
    # (thermo review follow-up, regression fix) `$age` here comes from the
    # lock directory's own mtime, not an owner file (there isn't one) --
    # still enough to tell "probably still mid-acquire" from "orphaned, the
    # owning run never got to write its owner file at all" apart, and,
    # critically, still gives a recovery path once 30 minutes have passed
    # either way, instead of blocking every future deploy forever.
    if (( age >= DEPLOY_LOCK_STALE_SECONDS )); then
      die "Deploy lock at $root/.deploy.lock is STALE (age ${age}s, based on the lock directory's own mtime -- its owner file is missing, most likely an interrupted acquire, e.g. a dropped SSH connection between mkdir and the owner write). Refusing to break it automatically.
$manual_clear"
    else
      die "Deploy lock at $root/.deploy.lock exists, but its owner file is missing (age ${age}s, based on the lock directory's own mtime) -- likely a concurrent acquire in progress; retry shortly rather than clearing it now. If this persists beyond ${DEPLOY_LOCK_STALE_SECONDS}s, it is an orphaned lock.
$manual_clear"
    fi
  elif (( age < DEPLOY_LOCK_STALE_SECONDS )); then
    die "Deploy lock is held (age ${age}s, under the 1800s staleness threshold). Another deploy is in progress; aborting with no changes. Owner:
$owner"
  else
    die "Deploy lock at $root/.deploy.lock is STALE (age ${age}s). Refusing to break it automatically. Owner:
$owner
$manual_clear"
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
#
# (thermo review, Critical C1, both reviewers) The pattern is anchored to
# the *exact* shape `deploy.sh` generates -- `date -u +%Y%m%dT%H%M%SZ`
# (8 digits, "T", 6 digits, "Z") + "-" + 12 lowercase hex characters (a
# 6-byte `node:crypto` `randomBytes(6).toString("hex")`), confirmed against
# `scripts/deploy.sh`'s own `release="$(date -u +%Y%m%dT%H%M%SZ)-$(node -e
# ...)"` line and the real anchor value
# (`20260924T141451Z-e994aaab005c`) -- deliberately *not* the original
# `[^[:space:]]+` (any non-whitespace token). A permissive pattern let
# `.agents/deployment.md`'s own "Release:`/`Commit:` marker format"
# documentation example (a line literally starting `Release: ` for
# illustration) be picked up by `tail -n1` as though it were the real last
# marker, since it appears later in the file than the real anchor --
# confirmed to reproduce against the committed file before this fix, and
# it would have made every guarded `--deploy` run abort as though an
# unrecorded deploy had happened. The doc's own example was also reworded
# (see `.agents/deployment.md`'s "marker format" section) so it can never
# start a line with a real-shaped `Release: ` token even if this regex is
# ever loosened again later -- defense in depth on both sides, per the
# review's own "fix must do both" guidance. Deliberately not attempting to
# strip Markdown code fences here (`if practical` per the task): the strict
# id-shape anchor already makes an accidental false match need an actual,
# correctly-shaped fake release id typed into prose, which the reworded
# example above no longer does, and fence-tracking would add real
# complexity for that already-closed residual case.
deploy_last_release_marker() {
  local doc=${DEPLOY_DEPLOYMENT_DOC:-.agents/deployment.md}
  [[ -f "$doc" ]] || { printf ''; return 0; }
  # `|| true`: under `set -o pipefail`, grep finding no marker at all (exit 1)
  # would otherwise propagate as this pipeline's exit status and, because
  # the whole thing is the value side of `marker=$(deploy_last_release_marker)`,
  # trip the caller's `set -e` -- a legitimate "no marker yet" case must
  # return an empty string, not abort the script here.
  grep -oE '^Release: [0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$' -- "$doc" | tail -n1 | sed -E 's/^Release: //' || true
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
