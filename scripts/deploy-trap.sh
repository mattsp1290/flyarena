# shellcheck shell=bash
# (thermo review, Suggestion S2/ops-safety) this file has no shebang -- it is
# always sourced, never executed -- so shellcheck can't infer the target
# shell on its own (SC2148) without this directive.
#
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

# (thermo review, Critical/maintainability C2 + Suggestion S1/ops-safety)
# `DEPLOY_LOCK_ROOT_OVERRIDE` and `DEPLOY_DEPLOYMENT_DOC` are test-only hooks
# (scripts/deploy-lock.sh's own header comment): the former redirects every
# "remote" lock operation to a local directory instead of the real host over
# SSH, and the latter redirects which local file the Release: marker is read
# from. Neither is checked anywhere else, so a leftover exported value (a
# stale shell session, or accidentally landing in `.env`, which deploy.sh
# sources wholesale) would silently defeat the concurrency guard against the
# real host during an actual `--deploy` run -- the one thing this whole
# mechanism exists to prevent. deploy.sh calls this as the very first
# statement on entering `--deploy` mode, before anything else. Defined here
# (not in scripts/deploy-lock.sh, which isn't sourced until later in that
# same mode block) so it can run first and so
# scripts/verify/deploy-lock.test.sh can call this exact function directly.
deploy_refuse_test_overrides() {
  [[ -z "${DEPLOY_LOCK_ROOT_OVERRIDE:-}" ]] || die 'DEPLOY_LOCK_ROOT_OVERRIDE is set; this is a test-only hook for scripts/verify/deploy-lock.test.sh and must never be set for a real --deploy run. Unset it and retry.'
  [[ -z "${DEPLOY_DEPLOYMENT_DOC:-}" ]] || die 'DEPLOY_DEPLOYMENT_DOC is set; this is a test-only hook for scripts/verify/deploy-lock.test.sh and must never be set for a real --deploy run. Unset it and retry.'
}
