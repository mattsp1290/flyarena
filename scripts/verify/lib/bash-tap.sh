# shellcheck shell=bash
# (thermo review, Important I1/maintainability) Shared scaffolding for this
# repo's hand-rolled TAP-like `scripts/verify/*.test.sh` files
# (`scripts/verify/deploy-lock.test.sh`, `scripts/verify/graph-lab.test.sh`):
# `pass`/`fail` bookkeeping, an EXIT trap that removes every scratch
# directory a test created, `new_scratch()`, and `tap_summary()` for the
# final report line and exit code. This file was extracted after the same
# ~25 lines were found copy-pasted line-for-line between the two test files
# above (identical `pass`/`fail`/`cleanup`/`new_scratch` bodies, down to the
# same `# shellcheck disable=SC2329` comment) -- a third `*.test.sh` file
# (WP5's planned `graph-lab-live.test.sh`) would otherwise paste it a third
# time.
#
# Deliberately NOT included here: `run_child`/`child_prelude`. Those
# genuinely differ per file (deploy-lock's prelude wires `ssh_options`/
# `die`/optionally `scripts/deploy-trap.sh`; graph-lab's wires a stub `PATH`
# and sources `scripts/graph-lab.sh` directly) -- unifying them would trade
# real, harmless duplication for a forced, over-parameterized abstraction.
# Each `*.test.sh` file keeps its own `run_child`/`child_prelude`.
#
# Sourced, not executed (no shebang -- shellcheck can't infer the shell on
# its own, hence the `shell=bash` directive above). Callers `source` this
# file near the top (after their own `set -euo pipefail`/`cd`) and call
# `tap_summary '<script-name>'` as their last statement.

failures=0
scratch_dirs=()

# `new_scratch` is meant to be called as `dir=$(new_scratch)` -- command
# substitution, which always runs in a subshell. A plain `scratch_dirs+=(...)`
# inside `new_scratch` would therefore mutate only that subshell's copy of
# the array, invisible to the parent shell's own `scratch_dirs` once the
# subshell exits -- every scratch directory created this way would silently
# never make it into the array cleanup() reads, and would never actually be
# removed. Tracked instead via a plain registry FILE (`_bash_tap_registry`,
# appended with `>>`, which does write through a subshell boundary to the
# same inode), which cleanup() reads back in the parent shell. Callers that
# append to `scratch_dirs` directly, in the parent shell's own top-level
# code rather than through `new_scratch` (not run in a subshell), still work
# too -- cleanup() removes both.
_bash_tap_registry=$(mktemp)

pass() { printf 'ok - %s\n' "$1"; }
fail() {
  printf 'not ok - %s\n' "$1"
  [[ -n "${2:-}" ]] && printf '  # %s\n' "$2"
  failures=$((failures + 1))
}

# (thermo review follow-up, regression fix) Ends with an explicit,
# unconditional `return 0`: without it, when `scratch_dirs` and the registry
# file both happen to be empty, this function's own last evaluated command
# is `[[ -n "$dir" ]] && rm -rf -- "$dir"` with `dir=""` (bash's
# `"${arr[@]:-}"` on an empty array substitutes a single empty-string
# element) -- a *false* `[[ ]]` test, i.e. this function would then return
# nonzero. Bash uses an EXIT trap's own final exit status as the process's
# real exit code unless the trap itself calls `exit` -- so `tap_summary`'s
# explicit `exit 0` was silently overridden to exit 1 by *this trap running
# after it*, even while printing "all checks passed" (confirmed by
# reproducing it with an isolated repro script before this fix, and by the
# fact that `scripts/verify/graph-lab.test.sh`'s `scratch_dirs` array was
# *always* empty in practice -- every call site used `dir=$(new_scratch)`,
# so the array-mutation-across-subshell issue above meant it could never
# have been anything else, independent of this fix).
# shellcheck disable=SC2329 # false positive: registered below via `trap cleanup EXIT`, which shellcheck's static analysis doesn't connect back to this definition.
cleanup() {
  local dir
  for dir in "${scratch_dirs[@]:-}"; do
    [[ -n "$dir" ]] && rm -rf -- "$dir"
  done
  if [[ -f "$_bash_tap_registry" ]]; then
    while IFS= read -r dir; do
      [[ -n "$dir" ]] && rm -rf -- "$dir"
    done < "$_bash_tap_registry"
    rm -f -- "$_bash_tap_registry"
  fi
  return 0
}
trap cleanup EXIT

new_scratch() {
  local dir
  dir=$(mktemp -d)
  printf '%s\n' "$dir" >> "$_bash_tap_registry"
  printf '%s' "$dir"
}

# Prints the final ok/not-ok tally under $name and exits 0 (all passed) or 1
# (one or more failed) -- the exact block every `*.test.sh` file used to
# duplicate at its own end.
tap_summary() {
  local name=$1
  echo
  if [[ $failures -eq 0 ]]; then
    printf '%s: all checks passed.\n' "$name"
    exit 0
  else
    printf '%s: %d check(s) failed.\n' "$name" "$failures"
    exit 1
  fi
}
