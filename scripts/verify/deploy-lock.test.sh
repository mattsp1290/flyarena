#!/usr/bin/env bash
# WP2 of `.agents/plans/findings-tour` (`02-verified-redeploy.md`): exercises
# scripts/deploy-lock.sh's lock, `Release:`-marker check, and rollback logic
# against a local scratch directory standing in for `DEPLOY_ROOT`, through
# the `DEPLOY_LOCK_ROOT_OVERRIDE` hook -- no SSH, no `.env`, no real host.
#
# Each check runs the library in a fresh child `bash` process (rather than
# sourcing it into this test runner's own process) so a `die()` call really
# `exit`s that one attempt, exactly as it does inside `scripts/deploy.sh`,
# without ending the whole test run; this script inspects each child's exit
# status, stdout/stderr, and the scratch directory's resulting filesystem
# state.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

lock_lib="$PWD/scripts/deploy-lock.sh"
trap_lib="$PWD/scripts/deploy-trap.sh"
[[ -f "$lock_lib" ]] || { printf 'deploy-lock.test: cannot find %s\n' "$lock_lib" >&2; exit 1; }
[[ -f "$trap_lib" ]] || { printf 'deploy-lock.test: cannot find %s\n' "$trap_lib" >&2; exit 1; }

failures=0
scratch_dirs=()

pass() { printf 'ok - %s\n' "$1"; }
fail() {
  printf 'not ok - %s\n' "$1"
  [[ -n "${2:-}" ]] && printf '  # %s\n' "$2"
  failures=$((failures + 1))
}

cleanup() {
  local dir
  for dir in "${scratch_dirs[@]:-}"; do
    [[ -n "$dir" ]] && rm -rf -- "$dir"
  done
}
trap cleanup EXIT

new_scratch_root() {
  local dir
  dir=$(mktemp -d)
  scratch_dirs+=("$dir")
  printf '%s' "$dir"
}

# A child script's shared prelude: the same `die()` deploy.sh defines
# (exits 1 on failure) and an `ssh_options` array deploy-lock.sh expects in
# scope -- unused whenever `DEPLOY_LOCK_ROOT_OVERRIDE` is set, since every
# "remote" operation then runs as a local filesystem call instead.
child_prelude() {
  cat <<PRELUDE
set -euo pipefail
die() { printf 'deploy: %s\n' "\$*" >&2; exit 1; }
ssh_options=()
source "$lock_lib"
PRELUDE
}

# Deploy.sh's own real ordering (die, then scripts/deploy-trap.sh -- which
# installs the actual on_exit/EXIT-trap machinery -- then, in --deploy mode,
# scripts/deploy-lock.sh, which overrides the trap library's no-op
# `deploy_lock_release` stub with the real one). Sourcing the literal
# scripts/deploy-trap.sh here (dual review, Important), rather than a
# hand-written reimplementation of its trap, is what makes the "trap
# releases the lock on failure" case below actually exercise deploy.sh's
# own trap -- a regression introduced only in scripts/deploy-trap.sh would
# be caught; used only by that one test below, so the other tests'
# "acquire and leave the lock behind" checks are not accidentally undone by
# an auto-release.
child_prelude_with_trap() {
  cat <<PRELUDE
set -euo pipefail
die() { printf 'deploy: %s\n' "\$*" >&2; exit 1; }
source "$trap_lib"
ssh_options=()
source "$lock_lib"
PRELUDE
}

# Runs \$2 (a body of shell statements) as a child bash process with
# DEPLOY_LOCK_ROOT_OVERRIDE and DEPLOY_ROOT both set to \$1, plus \$3
# prepended as extra environment assignments (may be empty). Captures
# combined stdout+stderr and the exit status into the caller's own
# `child_out` / `child_status` variables (avoids a subshell so the caller
# can inspect them directly).
run_child() {
  local root=$1 body=$2 prelude=$3
  local script
  script="$(printf '%s\n%s\n' "$prelude" "$body")"
  set +e
  child_out=$(DEPLOY_LOCK_ROOT_OVERRIDE="$root" DEPLOY_ROOT="$root" DEPLOY_SSH='unused@unused' \
    bash -c "$script" 2>&1)
  child_status=$?
  set -e
}

# ---------------------------------------------------------------------------
# 1. acquire, then a second acquire aborts (lock still fresh)
# ---------------------------------------------------------------------------
root=$(new_scratch_root)
run_child "$root" 'deploy_lock_acquire "release-one"; echo ACQUIRED' "$(child_prelude)"
if [[ $child_status -eq 0 && "$child_out" == *ACQUIRED* ]]; then
  pass 'first acquire succeeds'
else
  fail 'first acquire succeeds' "status=$child_status out=$child_out"
fi
if [[ -d "$root/.deploy.lock" && -f "$root/.deploy.lock.owner" ]]; then
  pass 'lock directory and owner file persist after the acquiring process exits'
else
  fail 'lock directory and owner file persist after the acquiring process exits'
fi

run_child "$root" 'deploy_lock_acquire "release-two"; echo ACQUIRED' "$(child_prelude)"
if [[ $child_status -ne 0 && "$child_out" == *'held'* && "$child_out" != *ACQUIRED* ]]; then
  pass 'second acquire aborts while the lock is fresh, with no changes'
else
  fail 'second acquire aborts while the lock is fresh, with no changes' "status=$child_status out=$child_out"
fi
if grep -q '^release=release-one$' "$root/.deploy.lock.owner"; then
  pass 'the first acquire remains the recorded owner (second attempt made no changes)'
else
  fail 'the first acquire remains the recorded owner (second attempt made no changes)'
fi

# ---------------------------------------------------------------------------
# 2. a stale lock aborts as stale, and is never broken automatically
# ---------------------------------------------------------------------------
root=$(new_scratch_root)
mkdir -- "$root/.deploy.lock"
stale_ts=$(( $(date -u +%s) - 3600 ))
{
  printf 'timestamp=%s\n' "$stale_ts"
  printf 'hostname=some-other-host\n'
  printf 'pid=12345\n'
  printf 'release=release-stale\n'
} > "$root/.deploy.lock.owner"

run_child "$root" 'deploy_lock_acquire "release-new"; echo ACQUIRED' "$(child_prelude)"
if [[ $child_status -ne 0 && "$child_out" == *STALE* && "$child_out" == *rmdir* ]]; then
  pass 'a stale lock (age > 30 minutes) aborts, reported as stale, with the manual-clear command'
else
  fail 'a stale lock (age > 30 minutes) aborts, reported as stale, with the manual-clear command' "status=$child_status out=$child_out"
fi
if [[ -d "$root/.deploy.lock" ]] && grep -q '^release=release-stale$' "$root/.deploy.lock.owner"; then
  pass 'the stale lock is left untouched, never broken automatically'
else
  fail 'the stale lock is left untouched, never broken automatically'
fi

# ---------------------------------------------------------------------------
# 2b. (dual review, Important) a failure that is NOT lock contention (e.g.
# permission denied on the lock host) is reported distinctly, never
# misreported as a stale lock -- both independent reviewers flagged that an
# acquire-script failure unrelated to contention (SSH drop, permissions,
# disk full) previously produced empty/unusable output that the staleness
# math defaulted into a fabricated "STALE" report. Simulated here with a
# root directory this user cannot write into, so `mkdir -- "$lock"` fails
# with EACCES rather than EEXIST.
# ---------------------------------------------------------------------------
root=$(new_scratch_root)
chmod 0500 -- "$root"
run_child "$root" 'deploy_lock_acquire "release-perm"; echo ACQUIRED' "$(child_prelude)"
chmod 0700 -- "$root" # restore write access so the outer trap can clean it up
if [[ $child_status -ne 0 && "$child_out" != *STALE* && "$child_out" != *'is held'* && "$child_out" == *'not confirmed lock contention'* ]]; then
  pass 'a lock-host failure unrelated to contention aborts with its own message, never reported as stale or held'
else
  fail 'a lock-host failure unrelated to contention aborts with its own message, never reported as stale or held' "status=$child_status out=$child_out"
fi

# ---------------------------------------------------------------------------
# 3. the trap releases the lock on EXIT, including on failure
# ---------------------------------------------------------------------------
root=$(new_scratch_root)
run_child "$root" 'deploy_lock_acquire "release-trap"; false' "$(child_prelude_with_trap)"
if [[ $child_status -ne 0 ]]; then
  pass 'the child process that acquires the lock and then fails exits nonzero'
else
  fail 'the child process that acquires the lock and then fails exits nonzero' "status=$child_status out=$child_out"
fi
if [[ ! -e "$root/.deploy.lock" && ! -e "$root/.deploy.lock.owner" ]]; then
  pass 'the EXIT trap released the lock despite the failure'
else
  fail 'the EXIT trap released the lock despite the failure' "lock still present under $root"
fi

# ---------------------------------------------------------------------------
# 3b. (dual review, Important) deploy_lock_release refuses to remove a lock
# whose owner file no longer records this run's own release -- the
# documented manual stale-clear command is an operator escape hatch; if it
# is used against a lock this run still legitimately holds, a different run
# may acquire it before this one exits, and this run's own release must not
# then delete that other run's lock out from under it.
# ---------------------------------------------------------------------------
root=$(new_scratch_root)
run_child "$root" \
  'deploy_lock_acquire "release-mine"
   printf "timestamp=%s\nhostname=x\npid=1\nrelease=release-other\n" "$(date -u +%s)" > "$DEPLOY_LOCK_ROOT_OVERRIDE/.deploy.lock.owner"
   deploy_lock_release
   echo DONE' \
  "$(child_prelude)"
if [[ $child_status -eq 0 && "$child_out" == *DONE* && "$child_out" == *'NOT releasing it'* ]]; then
  pass 'deploy_lock_release refuses to remove a lock no longer owned by this run, and warns'
else
  fail 'deploy_lock_release refuses to remove a lock no longer owned by this run, and warns' "status=$child_status out=$child_out"
fi
if [[ -d "$root/.deploy.lock" ]] && grep -q '^release=release-other$' "$root/.deploy.lock.owner"; then
  pass 'the other run'"'"'s lock and owner file are left untouched'
else
  fail 'the other run'"'"'s lock and owner file are left untouched'
fi

# ---------------------------------------------------------------------------
# 4. rollback is refused when `current` changed since this run published
# ---------------------------------------------------------------------------
root=$(new_scratch_root)
mkdir -p -- "$root/releases/release-new" "$root/releases/release-previous"
ln -s -- releases/release-new "$root/current"

run_child "$root" \
  'if deploy_rollback "release-old" "release-previous"; then echo ROLLED_BACK; else echo REFUSED; fi' \
  "$(child_prelude)"
if [[ $child_status -eq 0 && "$child_out" == *REFUSED* && "$child_out" != *ROLLED_BACK* ]]; then
  pass 'rollback is refused when current no longer equals the release this run published'
else
  fail 'rollback is refused when current no longer equals the release this run published' "status=$child_status out=$child_out"
fi
if [[ "$(readlink -- "$root/current")" == releases/release-new ]]; then
  pass 'a refused rollback leaves current untouched'
else
  fail 'a refused rollback leaves current untouched'
fi

# Positive case: rollback proceeds and matches the documented symlink swap
# (`ln -s releases/<previous> .rollback-current && mv -Tf .rollback-current current`)
# when current still equals the release this run published.
root=$(new_scratch_root)
mkdir -p -- "$root/releases/release-just-deployed" "$root/releases/release-previous"
ln -s -- releases/release-just-deployed "$root/current"

run_child "$root" \
  'if deploy_rollback "release-just-deployed" "release-previous"; then echo ROLLED_BACK; else echo REFUSED; fi' \
  "$(child_prelude)"
if [[ $child_status -eq 0 && "$child_out" == *ROLLED_BACK* && "$(readlink -- "$root/current")" == releases/release-previous ]]; then
  pass 'rollback proceeds and current points at the previous release when current still matches'
else
  fail 'rollback proceeds and current points at the previous release when current still matches' "status=$child_status out=$child_out current=$(readlink -- "$root/current" 2>&1)"
fi

# (dual review, Important) An orphaned `.rollback-current` left behind by a
# rollback interrupted between its `ln -s` and `mv -Tf` (a dropped SSH
# connection, a killed process) must not permanently block every later
# rollback attempt.
root=$(new_scratch_root)
mkdir -p -- "$root/releases/release-current" "$root/releases/release-target"
ln -s -- releases/release-current "$root/current"
ln -s -- releases/some-orphaned-release "$root/.rollback-current"

run_child "$root" \
  'if deploy_rollback "release-current" "release-target"; then echo ROLLED_BACK; else echo REFUSED; fi' \
  "$(child_prelude)"
if [[ $child_status -eq 0 && "$child_out" == *ROLLED_BACK* && "$(readlink -- "$root/current")" == releases/release-target ]]; then
  pass 'rollback succeeds despite a pre-existing orphaned .rollback-current from an earlier interrupted rollback'
else
  fail 'rollback succeeds despite a pre-existing orphaned .rollback-current from an earlier interrupted rollback' "status=$child_status out=$child_out"
fi

# ---------------------------------------------------------------------------
# 5. the Release-marker mismatch aborts
# ---------------------------------------------------------------------------
root=$(new_scratch_root)
mkdir -p -- "$root/releases/release-live"
ln -s -- releases/release-live "$root/current"
doc=$(mktemp)
scratch_dirs+=("$doc")
printf 'Release: release-stale-marker\nCommit: 0000000\n' > "$doc"

run_child "$root" 'deploy_check_release_marker; echo CHECKED' \
  "$(child_prelude)"$'\n'"export DEPLOY_DEPLOYMENT_DOC=$doc"
if [[ $child_status -ne 0 && "$child_out" == *'unrecorded deploy'* ]]; then
  pass 'a Release: marker that does not match the live current aborts, reporting an unrecorded deploy'
else
  fail 'a Release: marker that does not match the live current aborts, reporting an unrecorded deploy' "status=$child_status out=$child_out"
fi

# Positive case: a marker that matches the live current succeeds and
# echoes that release id (the same value deploy.sh records as
# `previous_release` for rollback).
printf 'Release: release-live\nCommit: 0000000\n' > "$doc"
run_child "$root" 'result=$(deploy_check_release_marker); printf "MATCHED:%s\n" "$result"' \
  "$(child_prelude)"$'\n'"export DEPLOY_DEPLOYMENT_DOC=$doc"
if [[ $child_status -eq 0 && "$child_out" == *'MATCHED:release-live'* ]]; then
  pass 'a Release: marker that matches the live current succeeds and returns that release id'
else
  fail 'a Release: marker that matches the live current succeeds and returns that release id' "status=$child_status out=$child_out"
fi

# No marker at all also aborts (nothing to anchor the guard to).
: > "$doc"
run_child "$root" 'deploy_check_release_marker; echo CHECKED' \
  "$(child_prelude)"$'\n'"export DEPLOY_DEPLOYMENT_DOC=$doc"
if [[ $child_status -ne 0 && "$child_out" == *"No 'Release:' marker found"* ]]; then
  pass 'a missing Release: marker aborts rather than deploying with no anchor'
else
  fail 'a missing Release: marker aborts rather than deploying with no anchor' "status=$child_status out=$child_out"
fi

echo
if [[ $failures -eq 0 ]]; then
  printf 'deploy-lock.test: all checks passed.\n'
  exit 0
else
  printf 'deploy-lock.test: %d check(s) failed.\n' "$failures"
  exit 1
fi
