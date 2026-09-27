#!/usr/bin/env bash
# WP4 of `.agents/plans/graph-lab` (`04-launch-and-runbook.md`): exercises
# scripts/graph-lab.sh's bind/origin/token validation, its DOCKER-USER
# egress-drop rule check-and-insert logic, --status/--uninstall reporting,
# and its test-override refusal -- entirely against PATH-stubbed
# `tailscale`/`iptables`/`sudo`/`docker`/`npm`. No sudo, no real container, no
# real tailnet, and no `.env` are used, in the style of
# scripts/verify/deploy-lock.test.sh.
#
# Two invocation styles, matching deploy-lock.test.sh's own split:
#  - Most checks `source` scripts/graph-lab.sh in a fresh child bash process
#    and call its functions (cmd_start, cmd_status, cmd_uninstall,
#    graph_lab_check_bind, ...) directly -- scripts/graph-lab.sh only
#    dispatches its CLI when executed directly (see its own header comment),
#    so sourcing it never runs anything on its own.
#  - A few checks run the file as a real subprocess (`bash
#    scripts/graph-lab.sh --status`) to exercise `graph_lab_main`'s own
#    dispatch and its GRAPH_LAB_ENV_FILE test-override refusal end to end.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

lib="$PWD/scripts/graph-lab.sh"
[[ -f "$lib" ]] || { printf 'graph-lab.test: cannot find %s\n' "$lib" >&2; exit 1; }

# Shared pass/fail/cleanup/new_scratch/tap_summary scaffolding (thermo
# review, Important I1/maintainability) -- see that file's own header
# comment for why this is a separate file, and what is deliberately kept
# per-file instead (run_child/child_prelude below).
# shellcheck source=scripts/verify/lib/bash-tap.sh
source "$PWD/scripts/verify/lib/bash-tap.sh"

# ---------------------------------------------------------------------------
# Command stubs. Every stub reads a single env var, GRAPH_LAB_STUB_WORLD, and
# keeps its fake "system state" as plain files under it -- see each stub's
# own body for the file names it uses. `stub_bin` (with sudo) and
# `stub_bin_no_sudo` (identical, minus the sudo stub -- simulating a host
# with no sudo on PATH) are built once and reused, with PATH prepended per
# child process; `GRAPH_LAB_STUB_WORLD` is fresh per test/scratch dir.
# ---------------------------------------------------------------------------
write_stub() {
  local path=$1 body=$2
  printf '%s\n' "$body" > "$path"
  chmod +x "$path"
}

tailscale_stub_body='#!/usr/bin/env bash
set -euo pipefail
world="${GRAPH_LAB_STUB_WORLD:?}"
if [[ "${1:-}" == ip && "${2:-}" == -4 ]]; then
  cat -- "$world/tailscale-ips" 2>/dev/null || true
  exit 0
fi
exit 1'

iptables_stub_body='#!/usr/bin/env bash
set -euo pipefail
world="${GRAPH_LAB_STUB_WORLD:?}"
printf "%s\n" "$*" >> "$world/iptables-calls.log"
# Validates the FULL invocation shape (not just $1) -- a regression that
# dropped `-m conntrack --ctstate NEW` or changed `-j DROP` to `-j ACCEPT`
# in scripts/graph-lab.sh must fail loudly here (exit 2), not silently pass
# by matching only on the leading flag.
case "$*" in
  "-C DOCKER-USER -s "*" -m conntrack --ctstate NEW -j DROP")
    [[ -f "$world/iptables-rule" ]] && exit 0 || exit 1 ;;
  "-I DOCKER-USER -s "*" -m conntrack --ctstate NEW -j DROP")
    if [[ -f "$world/iptables-insert-fails" ]]; then exit 1; fi
    : > "$world/iptables-rule"
    exit 0 ;;
  "-D DOCKER-USER -s "*" -m conntrack --ctstate NEW -j DROP")
    if [[ -f "$world/iptables-remove-fails" ]]; then exit 1; fi
    rm -f -- "$world/iptables-rule"
    exit 0 ;;
  *) exit 2 ;;
esac'

sudo_stub_body='#!/usr/bin/env bash
set -euo pipefail
world="${GRAPH_LAB_STUB_WORLD:?}"
printf "%s\n" "$*" >> "$world/sudo-calls.log"
exec "$@"'

docker_stub_body='#!/usr/bin/env bash
set -euo pipefail
world="${GRAPH_LAB_STUB_WORLD:?}"
mkdir -p -- "$world/docker"
printf "%s\n" "$*" >> "$world/docker/calls.log"
case "${1:-}" in
  network)
    case "${2:-}" in
      inspect)
        # `network inspect -f FORMAT NAME` returns the stored subnet (the
        # network-<name> file'"'"'s contents, written by `create` below);
        # plain `network inspect NAME` is an existence check only, matching
        # what scripts/graph-lab.sh actually calls for each.
        if [[ "${3:-}" == -f ]]; then
          name="${5:-}"
          [[ -f "$world/docker/network-$name" ]] || exit 1
          cat -- "$world/docker/network-$name"
          exit 0
        fi
        [[ -f "$world/docker/network-${3:-}" ]] && exit 0 || exit 1 ;;
      create)
        printf "%s" "${4:-}" > "$world/docker/network-${5:-}"
        exit 0 ;;
      rm)
        if [[ -f "$world/docker-network-rm-fails" ]]; then exit 1; fi
        rm -f -- "$world/docker/network-${3:-}"
        exit 0 ;;
      *) exit 1 ;;
    esac ;;
  run)
    name="" prev=""
    for arg in "$@"; do
      [[ "$prev" == "--name" ]] && name="$arg"
      prev="$arg"
    done
    [[ -n "$name" ]] || exit 1
    if [[ -f "$world/docker-run-fails" ]]; then exit 1; fi
    : > "$world/docker/container-$name-running"
    exit 0 ;;
  stop)
    [[ -f "$world/docker/container-${2:-}-running" ]] || exit 1
    rm -f -- "$world/docker/container-${2:-}-running"
    exit 0 ;;
  rm) exit 0 ;;
  inspect)
    if [[ "${2:-}" == -f && -f "$world/docker/container-${4:-}-running" ]]; then
      printf "true\n"
      exit 0
    fi
    exit 1 ;;
  build)
    : > "$world/docker/image-built"
    exit 0 ;;
  *) exit 1 ;;
esac'

npm_stub_body='#!/usr/bin/env bash
set -euo pipefail
world="${GRAPH_LAB_STUB_WORLD:?}"
mkdir -p -- "$world"
printf "%s\n" "$*" >> "$world/npm-calls.log"
if [[ "${1:-}" == run && "${2:-}" == graph-lab:bundle ]]; then
  exit "${GRAPH_LAB_STUB_NPM_BUNDLE_EXIT:-0}"
fi
exit 1'

# Simulates `uv run --locked python -I -m pytest -q -m spark -k <expr>`
# (the exact invocation scripts/graph-lab.sh uses). Validates that prefix
# (exit 2, distinct from a real pass/fail, on any drift) so a regression in
# the real invocation shape is caught here rather than passing vacuously.
# The hard-gate call (`-k "not AtlasReproductionTests"`) exits
# GRAPH_LAB_STUB_UV_HARDGATE_EXIT (default 0/pass). The atlas call (`-k
# AtlasReproductionTests`) fails on its first GRAPH_LAB_STUB_UV_ATLAS_FAIL_UNTIL
# invocations (tracked via a call-count file under $world) and succeeds
# after, so tests can exercise "passes first try", "fails all 3 retries",
# and "fails then recovers on retry".
uv_stub_body='#!/usr/bin/env bash
set -euo pipefail
world="${GRAPH_LAB_STUB_WORLD:?}"
mkdir -p -- "$world"
printf "%s\n" "$*" >> "$world/uv-calls.log"
case "$*" in
  "run --locked python -I -m pytest -q -m spark -k "*"AtlasReproductionTests"*)
    if [[ "$*" == *"not AtlasReproductionTests"* ]]; then
      exit "${GRAPH_LAB_STUB_UV_HARDGATE_EXIT:-0}"
    fi
    counter_file="$world/uv-atlas-call-count"
    count=0
    [[ -f "$counter_file" ]] && count=$(cat -- "$counter_file")
    count=$((count + 1))
    printf "%s" "$count" > "$counter_file"
    fail_until="${GRAPH_LAB_STUB_UV_ATLAS_FAIL_UNTIL:-0}"
    if (( count <= fail_until )); then
      exit 1
    fi
    exit 0 ;;
  *) exit 2 ;;
esac'

# Simulates `nvidia-smi --query-gpu=memory.free ...` and
# `--query-compute-apps=pid ...` (the exact invocations
# graph_lab_gpu_busy uses). Defaults to "abundant free memory, no compute
# apps" (not busy) unless a test writes $world/nvidia-smi-free-mib and/or
# $world/nvidia-smi-compute-apps to simulate contention.
nvidia_smi_stub_body='#!/usr/bin/env bash
set -euo pipefail
world="${GRAPH_LAB_STUB_WORLD:?}"
mkdir -p -- "$world"
printf "%s\n" "$*" >> "$world/nvidia-smi-calls.log"
case "$*" in
  "--query-gpu=memory.free --format=csv,noheader,nounits")
    cat -- "$world/nvidia-smi-free-mib" 2>/dev/null || printf "999999\n"
    ;;
  "--query-compute-apps=pid --format=csv,noheader")
    cat -- "$world/nvidia-smi-compute-apps" 2>/dev/null || true
    ;;
  *) exit 1 ;;
esac'

stub_bin=$(new_scratch)
write_stub "$stub_bin/tailscale" "$tailscale_stub_body"
write_stub "$stub_bin/iptables" "$iptables_stub_body"
write_stub "$stub_bin/sudo" "$sudo_stub_body"
write_stub "$stub_bin/docker" "$docker_stub_body"
write_stub "$stub_bin/npm" "$npm_stub_body"
write_stub "$stub_bin/uv" "$uv_stub_body"
write_stub "$stub_bin/nvidia-smi" "$nvidia_smi_stub_body"

stub_bin_no_sudo=$(new_scratch)
write_stub "$stub_bin_no_sudo/tailscale" "$tailscale_stub_body"
write_stub "$stub_bin_no_sudo/iptables" "$iptables_stub_body"
write_stub "$stub_bin_no_sudo/docker" "$docker_stub_body"
write_stub "$stub_bin_no_sudo/npm" "$npm_stub_body"
write_stub "$stub_bin_no_sudo/uv" "$uv_stub_body"
write_stub "$stub_bin_no_sudo/nvidia-smi" "$nvidia_smi_stub_body"

# No `uv` at all -- exercises graph_lab_run_reproduction_checks' own hard
# failure when uv is missing (not a silent skip).
stub_bin_no_uv=$(new_scratch)
write_stub "$stub_bin_no_uv/tailscale" "$tailscale_stub_body"
write_stub "$stub_bin_no_uv/iptables" "$iptables_stub_body"
write_stub "$stub_bin_no_uv/sudo" "$sudo_stub_body"
write_stub "$stub_bin_no_uv/docker" "$docker_stub_body"
write_stub "$stub_bin_no_uv/npm" "$npm_stub_body"
write_stub "$stub_bin_no_uv/nvidia-smi" "$nvidia_smi_stub_body"

# No `nvidia-smi` at all -- exercises graph_lab_gpu_busy's "no way to check,
# so not treated as busy" fallback.
stub_bin_no_nvidia=$(new_scratch)
write_stub "$stub_bin_no_nvidia/tailscale" "$tailscale_stub_body"
write_stub "$stub_bin_no_nvidia/iptables" "$iptables_stub_body"
write_stub "$stub_bin_no_nvidia/sudo" "$sudo_stub_body"
write_stub "$stub_bin_no_nvidia/docker" "$docker_stub_body"
write_stub "$stub_bin_no_nvidia/npm" "$npm_stub_body"
write_stub "$stub_bin_no_nvidia/uv" "$uv_stub_body"

# A minimal, real-coreutils-only PATH component (symlinks, no `nvidia-smi`
# or `uv`) for the two isolated-PATH tests below: this machine has a REAL
# `nvidia-smi` and a real `uv` (this repo's own dev environment has a real
# GPU), so simply prepending a stub dir onto the inherited PATH is not
# enough to simulate "missing" for either -- the real binary would still be
# found further down the same PATH. These tests instead use a fully
# isolated PATH (a no-nvidia-smi/no-uv stub dir plus only this directory),
# never inheriting the real PATH at all.
core_bin=$(new_scratch)
for _coreutil in bash dirname mktemp cat mkdir rm grep wc head tail sort; do
  _coreutil_path=$(command -v -- "$_coreutil") || { printf 'graph-lab.test: missing required coreutil: %s\n' "$_coreutil" >&2; exit 1; }
  ln -s -- "$_coreutil_path" "$core_bin/$_coreutil"
done

# A child script's shared prelude: sources scripts/graph-lab.sh (defining
# every function, running nothing -- see the guard at its own end) with a
# fresh PATH pointing only at the given stub dir plus the real system PATH
# (for bash builtins/coreutils the stubs themselves need: cat, mkdir, rm,
# printf).
child_prelude() {
  local stub_dir=$1
  cat <<PRELUDE
set -euo pipefail
export PATH="$stub_dir:\$PATH"
source "$lib"
PRELUDE
}

run_child() {
  local body=$1 prelude=$2
  local script
  script="$(printf '%s\n%s\n' "$prelude" "$body")"
  set +e
  child_out=$(bash -c "$script" 2>&1)
  child_status=$?
  set -e
}

# A fixture .env with a distinctive fake token/hostname/bind, used both for
# ordinary positive-path checks and for the leak test below.
fake_token='XLEAKCANARYTOKENxxxxxxxxxxxxxxxxxxxx'
fake_bind='100.66.77.88'
fake_host='leak-canary-host.example.internal'
fake_origin="https://$fake_host"
fake_deploy_url="$fake_origin/fly/"

write_env_fixture() {
  local path=$1 token=$2 origins=$3 bind=$4 deploy_url=$5
  {
    printf 'GRAPH_LAB_TOKEN=%s\n' "$token"
    printf 'GRAPH_LAB_ORIGINS=%s\n' "$origins"
    printf 'GRAPH_LAB_BIND=%s\n' "$bind"
    printf 'DEPLOY_URL=%s\n' "$deploy_url"
  } > "$path"
}

# ---------------------------------------------------------------------------
# 1. The bind check: tailnet IP accepted; 0.0.0.0, a LAN IP, and empty are
#    rejected.
# ---------------------------------------------------------------------------
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"

run_child 'graph_lab_check_bind "'"$fake_bind"'" && echo ACCEPTED || echo REJECTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *ACCEPTED* ]] \
  && pass 'a real tailnet address (from `tailscale ip -4`) is accepted as GRAPH_LAB_BIND' \
  || fail 'a real tailnet address (from `tailscale ip -4`) is accepted as GRAPH_LAB_BIND' "status=$child_status out=$child_out"

for bad in 0.0.0.0 192.168.1.5 ''; do
  run_child 'graph_lab_check_bind "'"$bad"'" && echo ACCEPTED || echo REJECTED' \
    "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
  [[ $child_status -eq 0 && "$child_out" == *REJECTED* ]] \
    && pass "GRAPH_LAB_BIND='$bad' is rejected (not printed by \`tailscale ip -4\`)" \
    || fail "GRAPH_LAB_BIND='$bad' is rejected (not printed by \`tailscale ip -4\`)" "status=$child_status out=$child_out"
done

# ---------------------------------------------------------------------------
# 2. Origin derivation and the GRAPH_LAB_ORIGINS membership check.
# ---------------------------------------------------------------------------
run_child 'origin=$(graph_lab_derive_origin "'"$fake_deploy_url"'") && printf "ORIGIN:%s\n" "$origin"' \
  "$(child_prelude "$stub_bin")"
[[ $child_status -eq 0 && "$child_out" == "ORIGIN:$fake_origin" ]] \
  && pass 'graph_lab_derive_origin extracts scheme+host, dropping the path' \
  || fail 'graph_lab_derive_origin extracts scheme+host, dropping the path' "status=$child_status out=$child_out"

run_child 'graph_lab_derive_origin "not-a-url" && echo DERIVED || echo REFUSED' \
  "$(child_prelude "$stub_bin")"
[[ $child_status -eq 0 && "$child_out" == *REFUSED* ]] \
  && pass 'graph_lab_derive_origin refuses a non-http(s) value' \
  || fail 'graph_lab_derive_origin refuses a non-http(s) value' "status=$child_status out=$child_out"

run_child 'graph_lab_origin_in_list "'"$fake_origin"'" "http://other.example, '"$fake_origin"' ,http://third.example" && echo FOUND || echo MISSING' \
  "$(child_prelude "$stub_bin")"
[[ $child_status -eq 0 && "$child_out" == *FOUND* ]] \
  && pass 'graph_lab_origin_in_list finds the origin among comma-separated entries, trimming whitespace' \
  || fail 'graph_lab_origin_in_list finds the origin among comma-separated entries, trimming whitespace' "status=$child_status out=$child_out"

run_child 'graph_lab_origin_in_list "'"$fake_origin"'" "http://other.example,http://third.example" && echo FOUND || echo MISSING' \
  "$(child_prelude "$stub_bin")"
[[ $child_status -eq 0 && "$child_out" == *MISSING* ]] \
  && pass 'graph_lab_origin_in_list reports missing when the origin is absent from the list' \
  || fail 'graph_lab_origin_in_list reports missing when the origin is absent from the list' "status=$child_status out=$child_out"

# ---------------------------------------------------------------------------
# 3. A short GRAPH_LAB_TOKEN is rejected by cmd_start, before anything else
#    (docker/network/rule) is touched.
# ---------------------------------------------------------------------------
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
: > "$world/iptables-rule"
env_file="$world/.env"
write_env_fixture "$env_file" 'too-short' "$fake_origin" "$fake_bind" "$fake_deploy_url"

run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'at least 16 characters'* && "$child_out" != *STARTED* ]] \
  && pass 'cmd_start refuses a GRAPH_LAB_TOKEN shorter than 16 characters' \
  || fail 'cmd_start refuses a GRAPH_LAB_TOKEN shorter than 16 characters' "status=$child_status out=$child_out"
[[ ! -e "$world/docker/calls.log" ]] \
  && pass 'a short-token refusal never invokes docker' \
  || fail 'a short-token refusal never invokes docker' "log: $(cat -- "$world/docker/calls.log" 2>&1)"

# ---------------------------------------------------------------------------
# 4. GRAPH_LAB_ORIGINS not containing DEPLOY_URL's origin is rejected.
# ---------------------------------------------------------------------------
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" 'http://unrelated.example' "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'must include the origin'* ]] \
  && pass 'cmd_start refuses GRAPH_LAB_ORIGINS that does not include the DEPLOY_URL origin' \
  || fail 'cmd_start refuses GRAPH_LAB_ORIGINS that does not include the DEPLOY_URL origin' "status=$child_status out=$child_out"

# ---------------------------------------------------------------------------
# 5. Rule check-and-insert logic.
# ---------------------------------------------------------------------------
subnet='172.31.250.0/24'

# 5a. graph_lab_rule_present reflects the -C check only (never mutates).
world=$(new_scratch)
run_child 'graph_lab_rule_present "'"$subnet"'" && echo PRESENT || echo ABSENT' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *ABSENT* && ! -e "$world/iptables-rule" ]] \
  && pass 'graph_lab_rule_present reports absent and makes no changes when the rule is missing' \
  || fail 'graph_lab_rule_present reports absent and makes no changes when the rule is missing' "status=$child_status out=$child_out"

# 5b. graph_lab_ensure_rule inserts with -I only when -C fails.
world=$(new_scratch)
run_child 'graph_lab_ensure_rule "'"$subnet"'" && echo ENSURED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *ENSURED* && -f "$world/iptables-rule" ]] \
  && pass 'graph_lab_ensure_rule inserts the missing rule via sudo iptables -I' \
  || fail 'graph_lab_ensure_rule inserts the missing rule via sudo iptables -I' "status=$child_status out=$child_out"
grep -qxF -- "-I DOCKER-USER -s $subnet -m conntrack --ctstate NEW -j DROP" "$world/iptables-calls.log" \
  && pass 'the insert used the exact -I DOCKER-USER ... -j DROP invocation, not a substring match' \
  || fail 'the insert used the exact -I DOCKER-USER ... -j DROP invocation, not a substring match' "log: $(cat -- "$world/iptables-calls.log")"

# 5c. Already present -> -I is never called.
world=$(new_scratch)
: > "$world/iptables-rule"
run_child 'graph_lab_ensure_rule "'"$subnet"'" && echo ENSURED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *ENSURED* ]] && ! grep -q -- '^-I ' "$world/iptables-calls.log" \
  && pass 'graph_lab_ensure_rule never calls -I when -C already succeeds' \
  || fail 'graph_lab_ensure_rule never calls -I when -C already succeeds' "status=$child_status out=$child_out calls=$(cat -- "$world/iptables-calls.log" 2>&1)"

# 5d. --start refuses when the rule is missing and cannot be inserted
# (sudo unavailable).
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin_no_sudo")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'Refusing to start the container'* && "$child_out" != *STARTED* ]] \
  && pass '--start refuses to run the container when the rule is missing and sudo is unavailable' \
  || fail '--start refuses to run the container when the rule is missing and sudo is unavailable' "status=$child_status out=$child_out"
[[ ! -e "$world/docker/container-flyarena-graph-lab-running" ]] \
  && pass 'no container was started when the rule could not be inserted' \
  || fail 'no container was started when the rule could not be inserted'

# 5e. --start refuses when the rule is missing and the insert itself fails
# (sudo present, iptables -I fails).
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
: > "$world/iptables-insert-fails"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'Refusing to start the container'* ]] \
  && pass '--start refuses when sudo iptables -I itself fails' \
  || fail '--start refuses when sudo iptables -I itself fails' "status=$child_status out=$child_out"

# 5f. Positive: --start succeeds end to end (rule missing but insertable,
# network absent, origin/token/bind all valid) and issues the exact docker
# run flags from the plan.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin,http://127.0.0.1:5173" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *STARTED* ]] \
  && pass '--start succeeds when token/origins/bind are valid and the rule can be inserted' \
  || fail '--start succeeds when token/origins/bind are valid and the rule can be inserted' "status=$child_status out=$child_out"
[[ -f "$world/iptables-rule" ]] \
  && pass 'the egress-drop rule is present after a successful --start' \
  || fail 'the egress-drop rule is present after a successful --start'
[[ -f "$world/docker/network-flyarena-graph-lab-net" ]] \
  && pass 'the dedicated bridge network was created' \
  || fail 'the dedicated bridge network was created'
[[ -f "$world/docker/container-flyarena-graph-lab-running" ]] \
  && pass 'the container was started' \
  || fail 'the container was started'
run_log="$world/docker/calls.log"
run_line=$(grep '^run ' "$run_log")
for flag in '--name flyarena-graph-lab' '--network flyarena-graph-lab-net' '--init' '--gpus all' \
  '--memory 8g' '--cpus 4' '--cap-drop ALL' '--security-opt no-new-privileges' '--read-only' \
  '--tmpfs /tmp:rw,noexec,nosuid,size=1g' "-p $fake_bind:8766:8000" '-e GRAPH_LAB_TOKEN' \
  '-e GRAPH_LAB_ORIGINS' 'flyarena-graph-lab:local'; do
  [[ "$run_line" == *"$flag"* ]] \
    && pass "docker run includes the plan's exact flag: $flag" \
    || fail "docker run includes the plan's exact flag: $flag" "run_line=$run_line"
done

# ---------------------------------------------------------------------------
# 5g. --start refuses when the existing network's ACTUAL subnet (read via
# `docker network inspect`) does not match the configured GRAPH_LAB_SUBNET
# -- never trusts config alone (ops-security thermo review, Important 2).
# ---------------------------------------------------------------------------
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
mkdir -p "$world/docker"
printf '%s' '172.31.250.0/24' > "$world/docker/network-flyarena-graph-lab-net"
env_file="$world/.env"
{
  write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
  printf 'GRAPH_LAB_SUBNET=%s\n' '10.99.0.0/24' >> "$env_file"
}
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'does not match the configured GRAPH_LAB_SUBNET'* \
  && "$child_out" == *'--uninstall'* && "$child_out" != *STARTED* ]] \
  && pass '--start refuses when the existing network subnet does not match the configured GRAPH_LAB_SUBNET' \
  || fail '--start refuses when the existing network subnet does not match the configured GRAPH_LAB_SUBNET' "status=$child_status out=$child_out"
[[ ! -e "$world/docker/container-flyarena-graph-lab-running" ]] \
  && pass 'no container was started on a subnet mismatch' \
  || fail 'no container was started on a subnet mismatch'

# Matching subnets: no refusal.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
mkdir -p "$world/docker"
printf '%s' '10.99.0.0/24' > "$world/docker/network-flyarena-graph-lab-net"
env_file="$world/.env"
{
  write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
  printf 'GRAPH_LAB_SUBNET=%s\n' '10.99.0.0/24' >> "$env_file"
}
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *STARTED* ]] \
  && pass '--start proceeds when the existing network subnet matches the configured GRAPH_LAB_SUBNET' \
  || fail '--start proceeds when the existing network subnet matches the configured GRAPH_LAB_SUBNET' "status=$child_status out=$child_out"
# The rule must be inserted for the (matching) actual/configured subnet.
grep -qxF -- '-I DOCKER-USER -s 10.99.0.0/24 -m conntrack --ctstate NEW -j DROP' "$world/iptables-calls.log" \
  && pass 'the egress-drop rule is inserted for the actual (matching) subnet' \
  || fail 'the egress-drop rule is inserted for the actual (matching) subnet' "log=$(cat -- "$world/iptables-calls.log" 2>&1)"

# --status also reports the mismatch (without refusing -- it is read-only).
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
mkdir -p "$world/docker"
printf '%s' '172.31.250.0/24' > "$world/docker/network-flyarena-graph-lab-net"
env_file="$world/.env"
printf 'GRAPH_LAB_SUBNET=%s\n' '10.99.0.0/24' > "$env_file"
run_child 'cmd_status' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *'network subnet: MISMATCH'* && "$child_out" == *'172.31.250.0/24'* && "$child_out" == *'10.99.0.0/24'* ]] \
  && pass '--status reports a network-subnet mismatch (both CIDR values, no token/hostname/IP)' \
  || fail '--status reports a network-subnet mismatch' "out=$child_out"

# --uninstall removes the rule keyed on the network's ACTUAL subnet, not the
# (mismatched) configured one.
world=$(new_scratch)
mkdir -p "$world/docker"
printf '%s' '172.31.250.0/24' > "$world/docker/network-flyarena-graph-lab-net"
: > "$world/iptables-rule"
env_file="$world/.env"
printf 'GRAPH_LAB_SUBNET=%s\n' '10.99.0.0/24' > "$env_file"
run_child 'cmd_uninstall' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
grep -qxF -- '-D DOCKER-USER -s 172.31.250.0/24 -m conntrack --ctstate NEW -j DROP' "$world/iptables-calls.log" \
  && pass "--uninstall removes the rule keyed on the network's actual subnet, not the mismatched configured one" \
  || fail "--uninstall removes the rule keyed on the actual subnet" "log=$(cat -- "$world/iptables-calls.log" 2>&1)"

# ---------------------------------------------------------------------------
# 5h. GRAPH_LAB_SUBNET is validated as a plausible IPv4 CIDR before it
# reaches docker/iptables (ops-security thermo review, Suggestion S1).
# ---------------------------------------------------------------------------
for bad_subnet in 'not-a-subnet' '172.31.250.0' '172.31.250.0/33' '0.0.0.0/0' '172.31.250.0/8'; do
  world=$(new_scratch)
  printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
  env_file="$world/.env"
  {
    write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
    printf 'GRAPH_LAB_SUBNET=%s\n' "$bad_subnet" >> "$env_file"
  }
  run_child 'cmd_start; echo STARTED' \
    "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
  [[ $child_status -ne 0 && "$child_out" != *STARTED* ]] \
    && pass "--start refuses an invalid GRAPH_LAB_SUBNET ('$bad_subnet')" \
    || fail "--start refuses an invalid GRAPH_LAB_SUBNET ('$bad_subnet')" "status=$child_status out=$child_out"
done
# A valid, in-range subnet is accepted.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
env_file="$world/.env"
{
  write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
  printf 'GRAPH_LAB_SUBNET=%s\n' '10.5.0.0/28' >> "$env_file"
}
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *STARTED* ]] \
  && pass "--start accepts a valid, in-range GRAPH_LAB_SUBNET ('10.5.0.0/28')" \
  || fail "--start accepts a valid, in-range GRAPH_LAB_SUBNET" "status=$child_status out=$child_out"

# ---------------------------------------------------------------------------
# 5i. The bind-check failure message distinguishes "tailscale printed no
# addresses at all" from "addresses were printed, but none matched"
# (ops-security thermo review, Suggestion S3), without ever printing an
# address either way.
# ---------------------------------------------------------------------------
world=$(new_scratch)
: > "$world/tailscale-ips" # tailscale prints nothing at all
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'printed no addresses at all'* && "$child_out" != *"$fake_bind"* ]] \
  && pass 'the bind-check failure distinguishes "no addresses at all" (tailscaled likely down)' \
  || fail 'the bind-check failure distinguishes "no addresses at all"' "out=$child_out"

world=$(new_scratch)
printf '%s\n' '100.1.2.3' > "$world/tailscale-ips" # prints an address, just not $fake_bind
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'none matched'* && "$child_out" != *'printed no addresses at all'* \
  && "$child_out" != *"$fake_bind"* && "$child_out" != *'100.1.2.3'* ]] \
  && pass 'the bind-check failure distinguishes "addresses printed but none matched" (stale/mistyped GRAPH_LAB_BIND)' \
  || fail 'the bind-check failure distinguishes "addresses printed but none matched"' "out=$child_out"

# ---------------------------------------------------------------------------
# 6. --status reporting.
# ---------------------------------------------------------------------------

# 6a. Nothing exists yet.
world=$(new_scratch)
run_child 'cmd_status' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$world/.env"
[[ $child_status -eq 0 && "$child_out" == *'network: MISSING'* && "$child_out" == *'egress-drop rule: MISSING'* && "$child_out" == *'container: not present'* ]] \
  && pass '--status reports MISSING/MISSING/not present with nothing installed' \
  || fail '--status reports MISSING/MISSING/not present with nothing installed' "out=$child_out"

# 6b. Everything present and bind matches -> reports tailscale0, not the address.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
: > "$world/iptables-rule"
mkdir -p "$world/docker"
: > "$world/docker/network-flyarena-graph-lab-net"
: > "$world/docker/container-flyarena-graph-lab-running"
env_file="$world/.env"
printf 'GRAPH_LAB_BIND=%s\n' "$fake_bind" > "$env_file"
run_child 'cmd_status' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *'network: present'* && "$child_out" == *'egress-drop rule: present'* \
  && "$child_out" == *'container: running'* && "$child_out" == *'bind: OK (tailscale0)'* && "$child_out" != *"$fake_bind"* ]] \
  && pass '--status reports present/present/running/bind OK, and never prints the bind address' \
  || fail '--status reports present/present/running/bind OK, and never prints the bind address' "out=$child_out"

# 6c. Bind mismatch (tailscale no longer reports the configured address,
# e.g. after a tailscaled restart) is detected and reported without the
# address.
world=$(new_scratch)
printf '%s\n' '100.1.2.3' > "$world/tailscale-ips"
mkdir -p "$world/docker"
: > "$world/docker/container-flyarena-graph-lab-running"
env_file="$world/.env"
printf 'GRAPH_LAB_BIND=%s\n' "$fake_bind" > "$env_file"
run_child 'cmd_status' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *'bind: MISMATCH'* && "$child_out" == *'--stop && ./scripts/graph-lab.sh --start'* && "$child_out" != *"$fake_bind"* ]] \
  && pass '--status detects a bind mismatch and reports the recovery command, without printing either address' \
  || fail '--status detects a bind mismatch and reports the recovery command, without printing either address' "out=$child_out"

# ---------------------------------------------------------------------------
# 7. --uninstall removes the container, rule, and network; idempotent when
#    nothing exists.
# ---------------------------------------------------------------------------
world=$(new_scratch)
: > "$world/iptables-rule"
mkdir -p "$world/docker"
: > "$world/docker/network-flyarena-graph-lab-net"
: > "$world/docker/container-flyarena-graph-lab-running"
env_file="$world/.env"
: > "$env_file"
run_child 'cmd_uninstall' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *'uninstalled'* ]] \
  && pass '--uninstall reports success' \
  || fail '--uninstall reports success' "status=$child_status out=$child_out"
[[ ! -f "$world/iptables-rule" ]] && pass '--uninstall removes the egress-drop rule' || fail '--uninstall removes the egress-drop rule'
[[ ! -f "$world/docker/network-flyarena-graph-lab-net" ]] && pass '--uninstall removes the network' || fail '--uninstall removes the network'
[[ ! -f "$world/docker/container-flyarena-graph-lab-running" ]] && pass '--uninstall removes the container' || fail '--uninstall removes the container'

# Idempotent: running it again with nothing left does not fail.
run_child 'cmd_uninstall' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 ]] \
  && pass '--uninstall is idempotent when nothing is left to remove' \
  || fail '--uninstall is idempotent when nothing is left to remove' "status=$child_status out=$child_out"

# 7b. Honesty check: if `sudo iptables -D` itself fails, --uninstall must
# report the rule as still present (not falsely claim removal) and warn.
world=$(new_scratch)
: > "$world/iptables-rule"
: > "$world/iptables-remove-fails"
env_file="$world/.env"
: > "$env_file"
run_child 'cmd_uninstall' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *'still present'* && "$child_out" == *'WARNING'* ]] \
  && pass '--uninstall reports the rule as still present (not falsely removed) when -D fails' \
  || fail '--uninstall reports the rule as still present (not falsely removed) when -D fails' "status=$child_status out=$child_out"
[[ -f "$world/iptables-rule" ]] \
  && pass 'the rule genuinely remains on disk when -D fails' \
  || fail 'the rule genuinely remains on disk when -D fails'

# 7c. Same honesty check for the network.
world=$(new_scratch)
mkdir -p "$world/docker"
: > "$world/docker/network-flyarena-graph-lab-net"
: > "$world/docker-network-rm-fails"
env_file="$world/.env"
: > "$env_file"
run_child 'cmd_uninstall' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -eq 0 && "$child_out" == *'network flyarena-graph-lab-net is still present'* ]] \
  && pass '--uninstall reports the network as still present (not falsely removed) when network rm fails' \
  || fail '--uninstall reports the network as still present (not falsely removed) when network rm fails' "status=$child_status out=$child_out"

# ---------------------------------------------------------------------------
# 8. --build: bundles, then the reproduction gate (lesion/swap-set hard gate,
#    atlas retried/non-blocking), then builds and tags the image. The `uv`
#    stub validates the exact `uv run --locked python -I -m pytest -q -m
#    spark -k <expr>` invocation shape and simulates pass/fail/retry-then-
#    recover outcomes -- see uv_stub_body's own comment.
# ---------------------------------------------------------------------------

# 8a. GRAPH_LAB_SKIP_REPRO=1 bypasses the gate entirely (no uv call at all).
world=$(new_scratch)
run_child 'cmd_build; echo BUILT' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_SKIP_REPRO=1"
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" == *'GRAPH_LAB_SKIP_REPRO=1 set'* ]] \
  && pass '--build (GRAPH_LAB_SKIP_REPRO=1) bundles and builds without running the reproduction gate' \
  || fail '--build (GRAPH_LAB_SKIP_REPRO=1) bundles and builds without running the reproduction gate' "status=$child_status out=$child_out"
grep -q '^run graph-lab:bundle$' "$world/npm-calls.log" \
  && pass '--build runs `npm run graph-lab:bundle`' \
  || fail '--build runs `npm run graph-lab:bundle`' "log=$(cat -- "$world/npm-calls.log" 2>&1)"
build_line=$(grep '^build ' "$world/docker/calls.log")
[[ "$build_line" == *'-t flyarena-graph-lab:local'* && "$build_line" == *'-f backend/graph_lab/Dockerfile'* ]] \
  && pass '--build tags flyarena-graph-lab:local from backend/graph_lab/Dockerfile' \
  || fail '--build tags flyarena-graph-lab:local from backend/graph_lab/Dockerfile' "build_line=$build_line"
[[ ! -e "$world/uv-calls.log" ]] \
  && pass 'GRAPH_LAB_SKIP_REPRO=1 means uv is never invoked' \
  || fail 'GRAPH_LAB_SKIP_REPRO=1 means uv is never invoked' "log=$(cat -- "$world/uv-calls.log" 2>&1)"

# 8b. uv missing on PATH is a hard failure, not a silent skip (the fix for
# the "an accidental PATH gap reaches the same outcome as an explicit
# GRAPH_LAB_SKIP_REPRO=1 opt-out" finding). Uses a fully isolated PATH
# (stub dir + bare coreutils only, via $core_bin), not a prepend onto the
# real inherited PATH -- prepending alone would still find this machine's
# real `uv` further down the real PATH and defeat the point of this test.
world=$(new_scratch)
script="$(printf 'set -euo pipefail\nexport PATH="%s:%s"\nsource "%s"\n%s\n' "$stub_bin_no_uv" "$core_bin" "$lib" 'graph_lab_run_reproduction_checks; echo RAN')"
set +e
child_out=$(GRAPH_LAB_STUB_WORLD="$world" bash -c "$script" 2>&1)
child_status=$?
set -e
[[ $child_status -ne 0 && "$child_out" == *'uv not found on PATH'* && "$child_out" != *RAN* ]] \
  && pass 'graph_lab_run_reproduction_checks dies when uv is missing, rather than silently skipping' \
  || fail 'graph_lab_run_reproduction_checks dies when uv is missing, rather than silently skipping' "status=$child_status out=$child_out"

# 8c. Lesion/swap-set (hard gate) failing refuses to build, before docker
# build is ever invoked.
world=$(new_scratch)
run_child 'cmd_build; echo BUILT' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_STUB_UV_HARDGATE_EXIT=1"
[[ $child_status -ne 0 && "$child_out" == *'lesion/swap-set reproduction checks failed'* && "$child_out" != *BUILT* ]] \
  && pass '--build refuses when the lesion/swap-set (hard gate) checks fail' \
  || fail '--build refuses when the lesion/swap-set (hard gate) checks fail' "status=$child_status out=$child_out"
[[ ! -e "$world/docker/image-built" ]] \
  && pass 'docker build is never invoked when the hard gate fails' \
  || fail 'docker build is never invoked when the hard gate fails'

# 8d. Hard gate passes, atlas passes on the first attempt: no retry, no
# warning, build proceeds.
world=$(new_scratch)
run_child 'cmd_build; echo BUILT' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" != *'WARNING'* ]] \
  && pass '--build succeeds with no warning when both the hard gate and the atlas check pass on the first attempt' \
  || fail '--build succeeds with no warning when both the hard gate and the atlas check pass on the first attempt' "status=$child_status out=$child_out"
[[ -f "$world/docker/image-built" ]] \
  && pass 'docker build runs once both checks pass' \
  || fail 'docker build runs once both checks pass'
uv_call_count=$(wc -l < "$world/uv-calls.log")
[[ "$uv_call_count" -eq 2 ]] \
  && pass 'uv is called exactly twice (hard gate once, atlas once) when atlas passes immediately' \
  || fail 'uv is called exactly twice (hard gate once, atlas once) when atlas passes immediately' "count=$uv_call_count"

# 8e. Hard gate passes, atlas fails all 3 attempts: never a hard gate --
# --build still tags the image, but prints the documented warning.
world=$(new_scratch)
run_child 'cmd_build; echo BUILT' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_STUB_UV_ATLAS_FAIL_UNTIL=3"
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" == *'WARNING: the atlas reproduction check did not pass in 3 attempts'* ]] \
  && pass '--build tags the image with a warning when the atlas check fails all 3 attempts (never a hard gate)' \
  || fail '--build tags the image with a warning when the atlas check fails all 3 attempts (never a hard gate)' "status=$child_status out=$child_out"
[[ -f "$world/docker/image-built" ]] \
  && pass 'docker build still runs despite the persistent atlas-check failure' \
  || fail 'docker build still runs despite the persistent atlas-check failure'
uv_call_count=$(wc -l < "$world/uv-calls.log")
[[ "$uv_call_count" -eq 4 ]] \
  && pass 'uv is called exactly 4 times (hard gate once, atlas 3 retries) when atlas never passes' \
  || fail 'uv is called exactly 4 times (hard gate once, atlas 3 retries) when atlas never passes' "count=$uv_call_count"

# 8f. Atlas fails twice, then recovers on the 3rd attempt: no warning, build
# proceeds -- this is the "signal to re-run" behavior actually working.
world=$(new_scratch)
run_child 'cmd_build; echo BUILT' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_STUB_UV_ATLAS_FAIL_UNTIL=2"
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" != *'WARNING'* ]] \
  && pass '--build succeeds with no warning when the atlas check recovers on retry' \
  || fail '--build succeeds with no warning when the atlas check recovers on retry' "status=$child_status out=$child_out"

# ---------------------------------------------------------------------------
# 8f2-8f5. GPU-busy pre-flight (ops-security thermo review, Important 3):
# --build skips only the atlas check (never the CPU lesion/swap-set hard
# gate) when the GPU pre-flight finds it busy, and records the skip loudly
# in the build output.
# ---------------------------------------------------------------------------

# 8f2. Free GPU memory below the 2 GiB threshold -> atlas skipped, hard gate
# still runs, image still built, skip recorded in the output.
world=$(new_scratch)
printf '1024\n' > "$world/nvidia-smi-free-mib" # under 2048 MiB
run_child 'cmd_build; echo BUILT' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" == *'GPU pre-flight found it busy'* \
  && "$child_out" == *'SKIPPED this run'* ]] \
  && pass '--build skips the atlas check with a loud warning when free GPU memory is below 2 GiB' \
  || fail '--build skips the atlas check when free GPU memory is below 2 GiB' "status=$child_status out=$child_out"
grep -qxF -- "run --locked python -I -m pytest -q -m spark -k not AtlasReproductionTests" "$world/uv-calls.log" \
  && pass 'the CPU lesion/swap-set hard gate still runs when the atlas check is skipped for a busy GPU' \
  || fail 'the CPU lesion/swap-set hard gate still runs when the atlas check is skipped for a busy GPU' "log=$(cat -- "$world/uv-calls.log" 2>&1)"
! grep -qxF -- 'run --locked python -I -m pytest -q -m spark -k AtlasReproductionTests' "$world/uv-calls.log" \
  && pass 'the atlas check itself is never invoked when skipped for a busy GPU' \
  || fail 'the atlas check itself is never invoked when skipped for a busy GPU' "log=$(cat -- "$world/uv-calls.log" 2>&1)"
[[ -f "$world/docker/image-built" ]] \
  && pass 'the image is still built when only the atlas check is skipped' \
  || fail 'the image is still built when only the atlas check is skipped'

# 8f3. Another compute process running (free memory otherwise fine) also
# triggers the skip.
world=$(new_scratch)
printf '999999\n' > "$world/nvidia-smi-free-mib"
printf '12345\n' > "$world/nvidia-smi-compute-apps"
run_child 'cmd_build; echo BUILT' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" == *'GPU pre-flight found it busy'* ]] \
  && pass '--build skips the atlas check when another compute process is running' \
  || fail '--build skips the atlas check when another compute process is running' "status=$child_status out=$child_out"

# 8f4. Plenty of free memory and no compute apps (the default stub state):
# GPU pre-flight finds it free, atlas check runs as normal, no skip message.
world=$(new_scratch)
run_child 'cmd_build; echo BUILT' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world"
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" != *'GPU pre-flight found it busy'* && "$child_out" != *'SKIPPED this run'* ]] \
  && pass '--build runs the atlas check normally when the GPU pre-flight finds it free' \
  || fail '--build runs the atlas check normally when the GPU pre-flight finds it free' "status=$child_status out=$child_out"
grep -qxF -- 'run --locked python -I -m pytest -q -m spark -k AtlasReproductionTests' "$world/uv-calls.log" \
  && pass 'the atlas check is actually invoked when the GPU is free' \
  || fail 'the atlas check is actually invoked when the GPU is free' "log=$(cat -- "$world/uv-calls.log" 2>&1)"

# 8f5. nvidia-smi missing entirely: no evidence of contention, so not
# treated as busy -- the atlas check still runs (this environment's own
# fallback, not a false "safe" skip).
world=$(new_scratch)
script="$(printf 'set -euo pipefail\nexport PATH="%s:%s"\nsource "%s"\n%s\n' "$stub_bin_no_nvidia" "$core_bin" "$lib" 'cmd_build; echo BUILT')"
set +e
child_out=$(GRAPH_LAB_STUB_WORLD="$world" bash -c "$script" 2>&1)
child_status=$?
set -e
[[ $child_status -eq 0 && "$child_out" == *BUILT* && "$child_out" != *'GPU pre-flight found it busy'* ]] \
  && pass '--build runs the atlas check normally when nvidia-smi is unavailable (no evidence of contention)' \
  || fail '--build runs the atlas check normally when nvidia-smi is unavailable' "status=$child_status out=$child_out"

# 8g. A failing `npm run graph-lab:bundle` aborts before any reproduction
# check or docker build runs (ordinary `set -e` propagation).
world=$(new_scratch)
run_child 'cmd_build; echo BUILT' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_STUB_NPM_BUNDLE_EXIT=1 GRAPH_LAB_SKIP_REPRO=1"
[[ $child_status -ne 0 && "$child_out" != *BUILT* ]] \
  && pass '--build aborts when `npm run graph-lab:bundle` fails' \
  || fail '--build aborts when `npm run graph-lab:bundle` fails' "status=$child_status out=$child_out"
[[ ! -e "$world/docker/calls.log" ]] \
  && pass 'docker is never invoked when the bundle step fails' \
  || fail 'docker is never invoked when the bundle step fails'

# 8h. A failing `docker run` (--start) never leaks the bind address, and
# reports a generic, secret-free failure message.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
: > "$world/iptables-rule"
: > "$world/docker-run-fails"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start; echo STARTED' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
[[ $child_status -ne 0 && "$child_out" == *'docker run failed to start the container'* && "$child_out" != *STARTED* \
  && "$child_out" != *"$fake_bind"* && "$child_out" != *"$fake_token"* && "$child_out" != *"$fake_host"* ]] \
  && pass 'a failing docker run reports a generic, secret-free failure message' \
  || fail 'a failing docker run reports a generic, secret-free failure message' "status=$child_status out=$child_out"

# ---------------------------------------------------------------------------
# 9. GRAPH_LAB_ENV_FILE is a test-only hook: graph_lab_refuse_test_overrides
# refuses loudly when it is set, and is a no-op otherwise -- exercised both
# as a direct function call and through a real subprocess invocation of the
# CLI dispatcher (`graph_lab_main`/`bash scripts/graph-lab.sh ...`).
# ---------------------------------------------------------------------------
run_child 'graph_lab_refuse_test_overrides; echo SHOULD_NOT_REACH_HERE' \
  "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_ENV_FILE=/tmp/should-never-be-used"
[[ $child_status -ne 0 && "$child_out" == *'GRAPH_LAB_ENV_FILE is set'* && "$child_out" != *SHOULD_NOT_REACH_HERE* ]] \
  && pass 'graph_lab_refuse_test_overrides refuses loudly when GRAPH_LAB_ENV_FILE is set' \
  || fail 'graph_lab_refuse_test_overrides refuses loudly when GRAPH_LAB_ENV_FILE is set' "status=$child_status out=$child_out"

run_child 'graph_lab_refuse_test_overrides; echo NEITHER_SET_OK' "$(child_prelude "$stub_bin")"
[[ $child_status -eq 0 && "$child_out" == *NEITHER_SET_OK* ]] \
  && pass 'graph_lab_refuse_test_overrides is a no-op when the hook is unset' \
  || fail 'graph_lab_refuse_test_overrides is a no-op when the hook is unset' "status=$child_status out=$child_out"

# Real subprocess: `bash scripts/graph-lab.sh --status` with the hook set
# must refuse before touching docker/tailscale/iptables at all.
world=$(new_scratch)
set +e
child_out=$(GRAPH_LAB_ENV_FILE=/tmp/should-never-be-used GRAPH_LAB_STUB_WORLD="$world" \
  PATH="$stub_bin:$PATH" bash "$lib" --status 2>&1)
child_status=$?
set -e
[[ $child_status -ne 0 && "$child_out" == *'GRAPH_LAB_ENV_FILE is set'* ]] \
  && pass 'running scripts/graph-lab.sh directly (not sourced) refuses when GRAPH_LAB_ENV_FILE is set, before dispatch' \
  || fail 'running scripts/graph-lab.sh directly (not sourced) refuses when GRAPH_LAB_ENV_FILE is set, before dispatch' "status=$child_status out=$child_out"
[[ ! -e "$world/docker" && ! -e "$world/tailscale-ips" ]] \
  && pass 'the refused real-mode run never invoked docker or tailscale' \
  || fail 'the refused real-mode run never invoked docker or tailscale'

# Sanity: the same real-subprocess path works normally (usage/--help) when
# the hook is unset.
set +e
child_out=$(bash "$lib" --help 2>&1)
child_status=$?
set -e
[[ $child_status -eq 0 && "$child_out" == *'Usage: scripts/graph-lab.sh'* ]] \
  && pass 'scripts/graph-lab.sh --help runs through the real CLI dispatcher when no test hook is set' \
  || fail 'scripts/graph-lab.sh --help runs through the real CLI dispatcher when no test hook is set' "status=$child_status out=$child_out"

# ---------------------------------------------------------------------------
# 10. Leak test: a fake, unique token/hostname/IP must never appear in
# stdout or stderr, on any path -- success, refusal, or failure -- across
# every subcommand.
# ---------------------------------------------------------------------------
leak_check() {
  local label=$1 out=$2
  local leaked=0
  [[ "$out" == *"$fake_token"* ]] && { leaked=1; }
  [[ "$out" == *"$fake_host"* ]] && { leaked=1; }
  [[ "$out" == *"$fake_bind"* ]] && { leaked=1; }
  if [[ $leaked -eq 0 ]]; then
    pass "$label never leaks the fake token, hostname, or bind address"
  else
    fail "$label never leaks the fake token, hostname, or bind address" "out=$out"
  fi
}

# 10a. A fully successful --start.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check 'a successful --start' "$child_out"

# 10b. --status while running.
run_child 'cmd_status' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check '--status while the container is running' "$child_out"

# 10c. --stop and --uninstall.
run_child 'cmd_stop' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check '--stop' "$child_out"
run_child 'cmd_uninstall' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check '--uninstall' "$child_out"

# 10d. Failure paths: bad bind, missing rule/no sudo, and the bind-mismatch
# status line all still must not leak the fake values that were involved.
world=$(new_scratch)
printf '%s\n' '100.1.2.3' > "$world/tailscale-ips" # does not include $fake_bind
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check 'a bind-rejected --start failure' "$child_out"

world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start' "$(child_prelude "$stub_bin_no_sudo")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check 'a rule-missing/no-sudo --start refusal' "$child_out"

# 10e. GRAPH_LAB_ORIGINS-mismatch refusal.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" 'http://unrelated.example' "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check 'a GRAPH_LAB_ORIGINS-mismatch --start refusal' "$child_out"

# 10f. A failing docker run (see 8h) never leaks the bind address either.
world=$(new_scratch)
printf '%s\n' "$fake_bind" > "$world/tailscale-ips"
: > "$world/iptables-rule"
: > "$world/docker-run-fails"
env_file="$world/.env"
write_env_fixture "$env_file" "$fake_token" "$fake_origin" "$fake_bind" "$fake_deploy_url"
run_child 'cmd_start' "$(child_prelude "$stub_bin")"$'\n'"export GRAPH_LAB_STUB_WORLD=$world GRAPH_LAB_ENV_FILE=$env_file"
leak_check 'a failing docker run' "$child_out"

tap_summary 'graph-lab.test'
