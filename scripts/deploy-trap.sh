# Sourced library (not executed directly): the generic "run every
# registered cleanup path, then release the deploy lock, on ANY exit --
# success, `die`, or an unexpected failure" trap `scripts/deploy.sh`
# installs immediately after defining `die()`.
#
# Extracted into its own file (rather than left inline in deploy.sh, as it
# originally was) so `scripts/verify/deploy-lock.test.sh`'s "the trap
# releases the lock on failure" case can source this exact code -- the same
# statements deploy.sh itself runs -- instead of hand-maintaining a second
# copy that could silently drift from deploy.sh's real trap (dual review,
# Important: a prior version of that test exercised its own reimplementation
# of this trap, which would not have caught a regression introduced only in
# deploy.sh's own copy).
#
# Sourced very early (right after `die()`), before `deploy.sh` even parses
# its mode argument, so it is in effect for every exit path, including a
# `die` issued long before scripts/deploy-lock.sh (which defines the real
# `deploy_lock_release`) is sourced in `--deploy` mode.
cleanup_paths=()
# Overridden by scripts/deploy-lock.sh in --deploy mode; a no-op otherwise
# (--build-only/--package-backend never acquire a lock) and safe even if an
# early `die` fires before deploy-lock.sh is sourced.
deploy_lock_release() { return 0; }
on_exit() {
  local status=$?
  local path
  for path in "${cleanup_paths[@]:-}"; do
    [[ -n "$path" ]] && rm -rf -- "$path"
  done
  deploy_lock_release
  exit "$status"
}
trap on_exit EXIT
