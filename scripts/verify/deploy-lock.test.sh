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

# shellcheck disable=SC2329 # false positive: registered below via `trap cleanup EXIT`, which shellcheck's static analysis doesn't connect back to this definition.
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

# Same as run_child, but with distinctive, obviously-fake DEPLOY_SSH/
# DEPLOY_URL values in place of the generic 'unused@unused' -- used by every
# test below that asserts a message never leaks either one (thermo review,
# Critical C2/ops-safety and its regression-fix follow-up).
fake_ssh='secret-deploy-user@secret-deploy-host.example.internal'
fake_url='https://secret-deploy-host.example.internal/fly/'
run_child_secret() {
  local root=$1 body=$2 prelude=$3
  local script
  script="$(printf '%s\n%s\n' "$prelude" "$body")"
  set +e
  child_out=$(DEPLOY_LOCK_ROOT_OVERRIDE="$root" DEPLOY_ROOT="$root" \
    DEPLOY_SSH="$fake_ssh" DEPLOY_URL="$fake_url" \
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
# 2c/2d. (thermo review follow-up, regression fix) A missing owner file --
# the lock directory exists but its `.deploy.lock.owner` never got written
# (or was removed), e.g. the owning run crashed between `mkdir` and the
# owner write (a dropped SSH connection, `kill -9`) -- must never become a
# permanently orphaned lock with no recovery path. The lock directory's own
# mtime stands in for the missing owner timestamp, giving this case the
# same fresh/stale math and, once stale, the same manual-clear command as a
# normal lock. Both cases also assert the fake DEPLOY_SSH/DEPLOY_URL never
# leak, via run_child_secret.
# ---------------------------------------------------------------------------

# 2c. old lock-directory mtime (>= 30 minutes) -> reported through the
# normal STALE path, noting the owner file is missing, with the manual-clear
# command -- never left as an unrecoverable "indeterminate" forever.
root=$(new_scratch_root)
mkdir -- "$root/.deploy.lock"
old_mtime_epoch=$(( $(date -u +%s) - 3600 ))
touch -d "@$old_mtime_epoch" -- "$root/.deploy.lock"
run_child_secret "$root" 'deploy_lock_acquire "release-owner-missing-stale"; echo ACQUIRED' "$(child_prelude)"
if [[ $child_status -ne 0 && "$child_out" == *STALE* && "$child_out" == *'owner file is missing'* && "$child_out" == *rmdir* \
  && "$child_out" != *"$fake_ssh"* && "$child_out" != *"$fake_url"* ]]; then
  pass 'a missing owner file with an old lock-directory mtime is reported as stale, with the manual-clear command'
else
  fail 'a missing owner file with an old lock-directory mtime is reported as stale, with the manual-clear command' "status=$child_status out=$child_out"
fi
if [[ -d "$root/.deploy.lock" ]]; then
  pass 'the owner-missing stale lock is left untouched, never broken automatically'
else
  fail 'the owner-missing stale lock is left untouched, never broken automatically'
fi

# 2d. fresh lock-directory mtime (< 30 minutes) -> reported as held/
# indeterminate ("retry shortly"), but with the "if this persists beyond 30
# minutes" guidance and the same manual-clear command as a fallback --
# never a dead end with no recovery path at all.
root=$(new_scratch_root)
mkdir -- "$root/.deploy.lock"
run_child_secret "$root" 'deploy_lock_acquire "release-owner-missing-fresh"; echo ACQUIRED' "$(child_prelude)"
if [[ $child_status -ne 0 && "$child_out" != *ACQUIRED* && "$child_out" != *STALE* \
  && "$child_out" == *'owner file is missing'* && "$child_out" == *'retry shortly'* \
  && "$child_out" == *'If this persists beyond'* && "$child_out" == *rmdir* \
  && "$child_out" != *"$fake_ssh"* && "$child_out" != *"$fake_url"* ]]; then
  pass 'a missing owner file with a fresh lock-directory mtime is reported as held/indeterminate, with retry and orphan-recovery guidance'
else
  fail 'a missing owner file with a fresh lock-directory mtime is reported as held/indeterminate, with retry and orphan-recovery guidance' "status=$child_status out=$child_out"
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
#
# (thermo review, Critical C1) Fixture ids below are deliberately in
# deploy.sh's real release-id shape (8 digits, "T", 6 digits, "Z", "-", 12
# hex chars) now that deploy_last_release_marker's regex requires it --
# an arbitrary label like the previous "release-live" fixture would no
# longer match at all, silently turning this into the "no marker found"
# case instead of the "mismatch" case it's meant to exercise.
# ---------------------------------------------------------------------------
live_release_id='20200101T000000Z-aaaaaaaaaaaa'
other_release_id='20200202T000000Z-bbbbbbbbbbbb'
root=$(new_scratch_root)
mkdir -p -- "$root/releases/$live_release_id"
ln -s -- "releases/$live_release_id" "$root/current"
doc=$(mktemp)
scratch_dirs+=("$doc")
printf 'Release: %s\nCommit: 0000000\n' "$other_release_id" > "$doc"

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
printf 'Release: %s\nCommit: 0000000\n' "$live_release_id" > "$doc"
run_child "$root" 'result=$(deploy_check_release_marker); printf "MATCHED:%s\n" "$result"' \
  "$(child_prelude)"$'\n'"export DEPLOY_DEPLOYMENT_DOC=$doc"
if [[ $child_status -eq 0 && "$child_out" == *"MATCHED:$live_release_id"* ]]; then
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

# ---------------------------------------------------------------------------
# 5b. (thermo review, Critical C1, both reviewers) the parser against the
# REAL, checked-in .agents/deployment.md returns the real anchor -- this is
# exactly the gap that let the doc's own "marker format" example poison the
# parser ship undetected: every prior marker test above only ever used a
# synthetic, single-marker fixture, never the actual production input.
# ---------------------------------------------------------------------------
#
# The expected id is the doc's last real release entry, found here with awk
# rather than a hard-coded id, so appending a deployment record doesn't break
# this check. It must also be well formed, so an awk match on a placeholder
# can't make this pass vacuously.
expected_real_marker=$(awk '$1 == "Release:" && $2 ~ /^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$/ && NF == 2 { last = $2 } END { print last }' .agents/deployment.md)
run_child "$root" 'result=$(deploy_last_release_marker); printf "REAL_MARKER:%s\n" "$result"' \
  "$(child_prelude)"
if [[ $child_status -eq 0 && -n "$expected_real_marker" && "$child_out" == *"REAL_MARKER:$expected_real_marker"* ]]; then
  pass 'the parser run against the real, checked-in .agents/deployment.md returns the real anchor release id'
else
  fail 'the parser run against the real, checked-in .agents/deployment.md returns the real anchor release id' "status=$child_status expected=$expected_real_marker out=$child_out"
fi

# 5c. A decoy example line shaped like a real marker (mimicking the exact
# bug this suite failed to catch before) must never be picked up as the
# last marker when a real anchor line also exists earlier in the file.
decoy_doc=$(mktemp)
scratch_dirs+=("$decoy_doc")
{
  printf 'Release: %s\n' "$live_release_id"
  printf 'Commit: 0000000\n'
  printf '\n### marker format\n\n'
  printf '```text\n'
  printf 'Release: RELEASE_ID\n'
  printf 'Commit: SHORT_SHA\n'
  printf '```\n'
} > "$decoy_doc"
run_child "$root" 'result=$(deploy_last_release_marker); printf "DECOY_TEST:%s\n" "$result"' \
  "$(child_prelude)"$'\n'"export DEPLOY_DEPLOYMENT_DOC=$decoy_doc"
if [[ $child_status -eq 0 && "$child_out" == *"DECOY_TEST:$live_release_id"* ]]; then
  pass 'a non-id-shaped decoy example line after the real marker is ignored, not picked up as the last marker'
else
  fail 'a non-id-shaped decoy example line after the real marker is ignored, not picked up as the last marker' "status=$child_status out=$child_out"
fi

# ---------------------------------------------------------------------------
# 6. (thermo review, Critical C2/ops-safety) neither DEPLOY_SSH nor
# DEPLOY_URL ever appears in stdout/stderr across the stale, held (fresh),
# and error (not-contention) acquire outcomes.
# ---------------------------------------------------------------------------

# 6a. held (fresh) path.
root=$(new_scratch_root)
run_child_secret "$root" 'deploy_lock_acquire "release-secret-one"' "$(child_prelude)"
run_child_secret "$root" 'deploy_lock_acquire "release-secret-two"' "$(child_prelude)"
if [[ $child_status -ne 0 && "$child_out" != *"$fake_ssh"* && "$child_out" != *"$fake_url"* ]]; then
  pass 'the held (fresh-lock) abort message never prints DEPLOY_SSH or DEPLOY_URL'
else
  fail 'the held (fresh-lock) abort message never prints DEPLOY_SSH or DEPLOY_URL' "status=$child_status out=$child_out"
fi

# 6b. stale path.
root=$(new_scratch_root)
mkdir -- "$root/.deploy.lock"
stale_ts=$(( $(date -u +%s) - 3600 ))
{
  printf 'timestamp=%s\n' "$stale_ts"
  printf 'hostname=some-other-host\n'
  printf 'pid=12345\n'
  printf 'release=release-secret-stale\n'
} > "$root/.deploy.lock.owner"
run_child_secret "$root" 'deploy_lock_acquire "release-secret-three"' "$(child_prelude)"
if [[ $child_status -ne 0 && "$child_out" == *STALE* && "$child_out" != *"$fake_ssh"* && "$child_out" != *"$fake_url"* ]]; then
  pass 'the stale-lock abort message (with its manual-clear command) never prints DEPLOY_SSH or DEPLOY_URL'
else
  fail 'the stale-lock abort message (with its manual-clear command) never prints DEPLOY_SSH or DEPLOY_URL' "status=$child_status out=$child_out"
fi

# 6c. error (not lock contention) path.
root=$(new_scratch_root)
chmod 0500 -- "$root"
run_child_secret "$root" 'deploy_lock_acquire "release-secret-four"' "$(child_prelude)"
chmod 0700 -- "$root"
if [[ $child_status -ne 0 && "$child_out" != *"$fake_ssh"* && "$child_out" != *"$fake_url"* ]]; then
  pass 'the lock-host-failure abort message never prints DEPLOY_SSH or DEPLOY_URL'
else
  fail 'the lock-host-failure abort message never prints DEPLOY_SSH or DEPLOY_URL' "status=$child_status out=$child_out"
fi

# ---------------------------------------------------------------------------
# 7. (thermo review, Critical, maintainability C2 / ops-safety item 3)
# DEPLOY_LOCK_ROOT_OVERRIDE and DEPLOY_DEPLOYMENT_DOC (test-only hooks) are
# loudly refused, before any action, if present in a real --deploy run --
# exercises deploy_refuse_test_overrides directly (the literal function
# deploy.sh calls as the first statement of its --deploy branch), not a
# reimplementation.
# ---------------------------------------------------------------------------
override_prelude() {
  cat <<PRELUDE
set -euo pipefail
die() { printf 'deploy: %s\n' "\$*" >&2; exit 1; }
source "$trap_lib"
PRELUDE
}

# (thermo review follow-up) A bare `child_out=$(cmd)` assignment where `cmd`
# is *expected* to fail would itself trip this script's own `set -e` (the
# same command-substitution-assignment gotcha `run_child`/`run_child_secret`
# above already guard against with `set +e`/`set -e`) -- this helper applies
# the same guard for the raw `bash -c` invocations below.
run_override_check() {
  local extra_env=$1 body=$2
  set +e
  # (shellcheck SC2086) `$extra_env` must stay unquoted-safe: it is either
  # empty or exactly one NAME=VALUE token (never containing spaces/globs in
  # practice), but quoting it unconditionally would break the empty case --
  # `env ""` tries to exec a literal empty-named command instead of just
  # running bash with the inherited environment. Branching avoids both.
  if [[ -n "$extra_env" ]]; then
    child_out=$(env "$extra_env" bash -c "$(override_prelude)"$'\n'"$body" 2>&1)
  else
    child_out=$(bash -c "$(override_prelude)"$'\n'"$body" 2>&1)
  fi
  child_status=$?
  set -e
}

run_override_check 'DEPLOY_LOCK_ROOT_OVERRIDE=/tmp/should-never-be-used' 'deploy_refuse_test_overrides; echo SHOULD_NOT_REACH_HERE'
if [[ $child_status -ne 0 && "$child_out" == *'DEPLOY_LOCK_ROOT_OVERRIDE is set'* && "$child_out" != *SHOULD_NOT_REACH_HERE* ]]; then
  pass 'deploy_refuse_test_overrides refuses loudly when DEPLOY_LOCK_ROOT_OVERRIDE is set'
else
  fail 'deploy_refuse_test_overrides refuses loudly when DEPLOY_LOCK_ROOT_OVERRIDE is set' "status=$child_status out=$child_out"
fi

run_override_check 'DEPLOY_DEPLOYMENT_DOC=/tmp/should-never-be-used' 'deploy_refuse_test_overrides; echo SHOULD_NOT_REACH_HERE'
if [[ $child_status -ne 0 && "$child_out" == *'DEPLOY_DEPLOYMENT_DOC is set'* && "$child_out" != *SHOULD_NOT_REACH_HERE* ]]; then
  pass 'deploy_refuse_test_overrides refuses loudly when DEPLOY_DEPLOYMENT_DOC is set'
else
  fail 'deploy_refuse_test_overrides refuses loudly when DEPLOY_DEPLOYMENT_DOC is set' "status=$child_status out=$child_out"
fi

run_override_check '' 'deploy_refuse_test_overrides; echo NEITHER_SET_OK'
if [[ $child_status -eq 0 && "$child_out" == *NEITHER_SET_OK* ]]; then
  pass 'deploy_refuse_test_overrides is a no-op when neither test-only hook is set'
else
  fail 'deploy_refuse_test_overrides is a no-op when neither test-only hook is set' "status=$child_status out=$child_out"
fi

# ---------------------------------------------------------------------------
# 8. (thermo review, Suggestion S3/ops-safety) scripts/deploy-trap.sh's
# on_exit runs every registered $cleanup_paths entry, not only the lock
# release half already covered by test group 3.
# ---------------------------------------------------------------------------
scratch_file=$(mktemp)
set +e
child_out=$(CLEANUP_TEST_PATH="$scratch_file" bash -c "
set -euo pipefail
die() { printf 'deploy: %s\n' \"\$*\" >&2; exit 1; }
source \"$trap_lib\"
cleanup_paths+=(\"\$CLEANUP_TEST_PATH\")
echo BEFORE_EXIT
" 2>&1)
child_status=$?
set -e
if [[ $child_status -eq 0 && "$child_out" == *BEFORE_EXIT* && ! -e "$scratch_file" ]]; then
  pass 'on_exit removes every registered cleanup_paths entry, not only the lock'
else
  fail 'on_exit removes every registered cleanup_paths entry, not only the lock' "status=$child_status out=$child_out exists=$([[ -e "$scratch_file" ]] && echo yes || echo no)"
fi
rm -f -- "$scratch_file" # in case the check above failed and left it behind

echo
if [[ $failures -eq 0 ]]; then
  printf 'deploy-lock.test: all checks passed.\n'
  exit 0
else
  printf 'deploy-lock.test: %d check(s) failed.\n' "$failures"
  exit 1
fi
