#!/usr/bin/env bash
# WP4 of `.agents/plans/graph-lab` (`04-launch-and-runbook.md`): build, launch,
# and tear down the private real-graph lab container on the Spark, bound to
# the tailnet only, with a `DOCKER-USER` egress-DROP rule on its dedicated
# bridge subnet. See docs/graph-lab.md and the "Real-graph lab" section of
# .agents/deployment.md for the operator runbook. This script never prints
# GRAPH_LAB_TOKEN, a hostname, or an IP address -- see
# scripts/verify/graph-lab.test.sh's leak test.
#
# Structured like scripts/deploy.sh/deploy-lock.sh: every helper below is a
# plain function, and `graph_lab_main` (the CLI dispatcher) only runs when
# this file is executed directly, not when it is `source`d -- see the guard
# at the bottom. scripts/verify/graph-lab.test.sh sources this file to call
# individual functions (bind/origin/token checks, rule check-and-insert,
# cmd_status, cmd_uninstall, ...) directly against PATH-stubbed `tailscale`/
# `iptables`/`sudo`/`docker`, with no sudo, no real container, and no real
# tailnet -- and separately runs the file as a real subprocess to exercise
# `graph_lab_main`'s own dispatch and its test-override refusal.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

die() { printf 'graph-lab: %s\n' "$*" >&2; exit 1; }

GRAPH_LAB_CONTAINER_NAME=flyarena-graph-lab
GRAPH_LAB_NETWORK_NAME=flyarena-graph-lab-net
GRAPH_LAB_SUBNET_DEFAULT=172.31.250.0/24
GRAPH_LAB_IMAGE=flyarena-graph-lab:local

# `GRAPH_LAB_ENV_FILE` is a test-only hook (parallel to scripts/deploy-lock.sh's
# `DEPLOY_LOCK_ROOT_OVERRIDE`/`DEPLOY_DEPLOYMENT_DOC`): it lets
# scripts/verify/graph-lab.test.sh point cmd_start/cmd_status/cmd_uninstall at
# a fixture file instead of the real, gitignored `.env`. Called as the very
# first thing `graph_lab_main` does (mirrors scripts/deploy.sh's
# `deploy_refuse_test_overrides`, "the very first thing this branch does"),
# so a leftover exported value can never silently make a real invocation of
# this script read the wrong file. The test file never goes through
# `graph_lab_main` while this variable is set for its function-level checks
# -- it sources this script and calls `cmd_start`/`cmd_status`/`cmd_uninstall`
# directly, exactly as scripts/verify/deploy-lock.test.sh calls
# `deploy_lock_acquire` etc. directly rather than through `deploy.sh`'s CLI.
graph_lab_refuse_test_overrides() {
  [[ -z "${GRAPH_LAB_ENV_FILE:-}" ]] || die 'GRAPH_LAB_ENV_FILE is set; this is a test-only hook for scripts/verify/graph-lab.test.sh and must never be set for a real run. Unset it and retry.'
}

# The subnet the network/iptables rule use: GRAPH_LAB_SUBNET from .env, or
# the plan's documented default.
graph_lab_subnet() {
  printf '%s' "${GRAPH_LAB_SUBNET:-$GRAPH_LAB_SUBNET_DEFAULT}"
}

# Extracts the scheme+authority ("origin") from a URL, e.g.
# `https://host.example/fly/` -> `https://host.example`. No hostname is
# hard-coded here; the value always comes from the caller's own DEPLOY_URL.
graph_lab_derive_origin() {
  local url=$1
  [[ "$url" =~ ^(https?://[^/[:space:]]+) ]] || return 1
  printf '%s' "${BASH_REMATCH[1]}"
}

# True iff $1 (an origin) appears, trimmed, as one of the comma-separated
# entries in $2 (GRAPH_LAB_ORIGINS).
graph_lab_origin_in_list() {
  local origin=$1 list=$2 item
  set -f
  local IFS=','
  for item in $list; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ "$item" == "$origin" ]]; then
      set +f
      return 0
    fi
  done
  set +f
  return 1
}

# True (exit 0) iff $1 is non-empty and exactly matches one address printed
# by `tailscale ip -4`. This is what rejects `0.0.0.0`, LAN addresses, and an
# empty/unset bind: none of those can ever appear in that command's output.
# On failure, distinguishes two different operator problems via exit status
# (never by printing an address): exit 2 means `tailscale ip -4` printed no
# addresses at all (tailscaled likely down or not connected); exit 1 means
# it printed at least one address, but none matched (a stale/mistyped
# GRAPH_LAB_BIND is more likely) -- ops-security thermo review, Suggestion
# S3.
graph_lab_check_bind() {
  local bind=$1 addr found_any=0
  [[ -n "$bind" ]] || return 1
  while IFS= read -r addr; do
    [[ -n "$addr" ]] || continue
    found_any=1
    [[ "$addr" == "$bind" ]] && return 0
  done < <(tailscale ip -4 2>/dev/null || true)
  [[ "$found_any" == 1 ]] && return 1
  return 2
}

# True iff the DOCKER-USER egress-DROP rule for $1 (the subnet) is present.
# `-C` only checks; it never modifies the ruleset.
graph_lab_rule_present() {
  local subnet=$1
  iptables -C DOCKER-USER -s "$subnet" -m conntrack --ctstate NEW -j DROP >/dev/null 2>&1
}

# Inserts the rule with `-I` only if `-C` says it is missing (never inserts a
# duplicate). Requires sudo. Returns nonzero (without dying) if sudo is
# unavailable or the insert itself fails, so callers can report their own,
# more specific refusal message.
graph_lab_ensure_rule() {
  local subnet=$1
  graph_lab_rule_present "$subnet" && return 0
  if ! command -v sudo >/dev/null 2>&1; then
    printf 'graph-lab: sudo not found; cannot insert the DOCKER-USER egress-drop rule.\n' >&2
    return 1
  fi
  sudo iptables -I DOCKER-USER -s "$subnet" -m conntrack --ctstate NEW -j DROP
}

# The dedicated network's ACTUAL configured subnet, straight from docker --
# never trusted from GRAPH_LAB_SUBNET/config alone. Empty (and nonzero) if
# the network doesn't exist or has no IPAM config.
graph_lab_network_subnet() {
  docker network inspect -f '{{(index .IPAM.Config 0).Subnet}}' "$GRAPH_LAB_NETWORK_NAME" 2>/dev/null
}

# Creates the dedicated bridge network with a fixed subnet, idempotently. If
# the network already exists, its ACTUAL subnet (read via `docker network
# inspect`, never assumed) must equal the configured one -- otherwise a
# stale network from a previous, differently-configured GRAPH_LAB_SUBNET
# could silently keep running while the egress-drop rule gets inserted for
# the *new* subnet, covering a network the container isn't actually on
# (ops-security thermo review, Important 2). Refuses loudly instead.
graph_lab_ensure_network() {
  local subnet=$1
  if docker network inspect "$GRAPH_LAB_NETWORK_NAME" >/dev/null 2>&1; then
    local actual
    actual=$(graph_lab_network_subnet) || actual=""
    if [[ -n "$actual" && "$actual" != "$subnet" ]]; then
      die "The existing docker network $GRAPH_LAB_NETWORK_NAME has subnet $actual, which does not match the configured GRAPH_LAB_SUBNET ($subnet). Run './scripts/graph-lab.sh --uninstall' first (this removes the egress-drop rule and the network), then --start again, rather than leaving a stale network and a rule mismatch."
    fi
    return 0
  fi
  docker network create --subnet "$subnet" "$GRAPH_LAB_NETWORK_NAME" >/dev/null \
    || die "Failed to create the docker network $GRAPH_LAB_NETWORK_NAME (subnet $subnet)."
}

# The subnet to actually check/manage the egress-drop rule against: the
# network's real, current subnet if the network exists (authoritative --
# this is what the container is actually attached to), falling back to the
# configured value only when there is no network yet to read from docker.
# Never trusts GRAPH_LAB_SUBNET/config alone when the network already
# exists (ops-security thermo review, Important 2).
graph_lab_effective_subnet() {
  local configured=$1 actual
  actual=$(graph_lab_network_subnet) || actual=""
  if [[ -n "$actual" ]]; then
    printf '%s' "$actual"
  else
    printf '%s' "$configured"
  fi
}

# Validates GRAPH_LAB_SUBNET is a plausible IPv4 CIDR (dotted-quad + prefix
# length) before it ever reaches `docker network create`/`iptables` -- a
# malformed or absurdly wide value (e.g. a typo'd mask, or `0.0.0.0/0`)
# would otherwise silently produce a network/rule pair that are internally
# consistent with each other but not with the operator's intent
# (ops-security thermo review, Suggestion S1). Not a security boundary (the
# value is always double-quoted before reaching docker/iptables, so this is
# a footgun guard, not an injection fix).
graph_lab_validate_subnet() {
  local subnet=$1
  [[ "$subnet" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}/([0-9]|[12][0-9]|3[0-2])$ ]] \
    || die "GRAPH_LAB_SUBNET '$subnet' is not a valid IPv4 CIDR (e.g. 172.31.250.0/24)."
  local prefix=${subnet##*/}
  (( prefix >= 16 && prefix <= 30 )) \
    || die "GRAPH_LAB_SUBNET '$subnet' must have a prefix length between /16 and /30 (got /$prefix) -- this is meant to be a small, dedicated bridge subnet, not a wide range."
}

# Gate 2 of `.agents/plans/graph-lab/00-overview.md`, re-run on every build
# per `02-job-engines.md`: `backend/graph_lab/tests/test_reproduction.py`'s
# `@pytest.mark.spark` checks. Split into two pytest invocations rather than
# one, because they are not equally strict:
#
# - Lesion and swap-set are exact-match, deterministic reproductions of the
#   published offline pipelines -- a real hard gate. A failure here refuses
#   to build/tag the image at all.
# - Atlas is deliberately tolerance-based (`ATLAS_MIN_MATCHING_CELLS`,
#   >= 24/30 cells) because `torch.sparse.mm`'s CUDA kernel has no
#   deterministic implementation (see that constant's own comment in
#   test_reproduction.py). test_reproduction.py's own module docstring says
#   as much: "Whoever implements `--build` should treat this specific
#   check's failure as a signal to re-run, not as a hard release gate."
#
# (Deviation from `02-job-engines.md`'s literal "refuses to tag ... on
# failure", which does not distinguish the two: that line predates
# test_reproduction.py's own tolerance-based design for the atlas check, and
# test_reproduction.py's docstring explicitly asks the implementer of
# `--build` to follow its guidance instead. See docs/graph-lab.md.)
#
# A second, smaller deviation from the same line: these run against the
# freshly bundled JS on the host, *before* `docker build` (see cmd_build),
# not "against the fresh image" as `02-job-engines.md` literally says --
# test_reproduction.py's own checks invoke the engines directly (not the
# built container), so there is nothing image-specific to test post-build,
# and failing fast here skips an otherwise-wasted image build. See
# docs/graph-lab.md's "Reproduction checks" section.
#
# Set GRAPH_LAB_SKIP_REPRO=1 to skip this step entirely -- e.g. while another
# GPU job is already running on the Spark (a multi-day training job has no
# spare GPU memory for a concurrent atlas search) -- and re-run it manually
# once the GPU is free: `cd backend/graph_lab && uv run pytest -m spark -v`.
# Missing `uv` is treated as a hard failure here, not a silent skip: only the
# explicit GRAPH_LAB_SKIP_REPRO=1 opt-out (checked by cmd_build, before this
# function is ever called) may bypass the hard gate -- an accidental PATH
# gap (a fresh shell, a systemd unit, a misconfigured cron) must never reach
# the same "checks never ran" outcome as a deliberate operator choice.
#
# $1, if "1", skips only the atlas check (the lesion/swap-set hard gate
# still runs) -- used by cmd_build's own GPU pre-flight below, since
# `test_reproduction.py`'s `AtlasReproductionTests` injects a fake,
# always-available GPU (`gpu_free_bytes=lambda: 999 * 1024**3`) for test
# determinism, so nothing in the reproduction test itself would ever refuse
# to run a real ~1-2 minute GPU job just because the GPU is actually busy.
graph_lab_run_reproduction_checks() {
  local skip_atlas=${1:-0}
  command -v uv >/dev/null 2>&1 || die 'uv not found on PATH; cannot run the reproduction checks. Install uv, or explicitly skip with GRAPH_LAB_SKIP_REPRO=1 (only while another GPU job needs the GPU) and re-run them once uv is available: cd backend/graph_lab && uv run pytest -m spark -v'
  printf 'graph-lab: running the lesion/swap-set reproduction checks (exact-match; hard gate)...\n'
  if ! (cd backend/graph_lab && uv run --locked python -I -m pytest -q -m spark -k 'not AtlasReproductionTests'); then
    die 'the lesion/swap-set reproduction checks failed (exact-match, deterministic). Refusing to build or tag flyarena-graph-lab:local. See backend/graph_lab/tests/test_reproduction.py.'
  fi
  if [[ "$skip_atlas" == 1 ]]; then
    printf 'graph-lab: atlas reproduction check SKIPPED this run (GPU pre-flight found the GPU busy) -- see the WARNING above. Re-run it manually once the GPU is free: cd backend/graph_lab && uv run pytest -m spark -k AtlasReproductionTests -v\n'
    return 0
  fi
  printf 'graph-lab: running the atlas reproduction check (tolerance-based; retried on failure, never a hard gate)...\n'
  local attempt ok=0
  for attempt in 1 2 3; do
    if (cd backend/graph_lab && uv run --locked python -I -m pytest -q -m spark -k AtlasReproductionTests); then
      ok=1
      break
    fi
    printf 'graph-lab: atlas reproduction check attempt %d/3 did not pass; retrying (known CUDA nondeterminism -- see ATLAS_MIN_MATCHING_CELLS in test_reproduction.py).\n' "$attempt" >&2
  done
  if [[ "$ok" != 1 ]]; then
    printf 'graph-lab: WARNING: the atlas reproduction check did not pass in 3 attempts. This is a known nondeterministic-CUDA-kernel effect (see test_reproduction.py) and is documented as a signal to re-run later, not a release gate -- continuing to build and tag the image. Investigate manually: cd backend/graph_lab && uv run pytest -m spark -k AtlasReproductionTests -v\n'
  fi
}

# Mirrors the production atlas engine's own "GPU busy" guard
# (`02-job-engines.md`: refuses to start if free GPU memory is below 2 GiB,
# read via `torch.cuda.mem_get_info`) so `--build`'s reproduction gate does
# not blindly launch a real GPU job while another one (e.g. a multi-day
# training run) is already using the GPU -- `test_reproduction.py`'s own
# `AtlasReproductionTests` injects an always-free fake GPU for test
# determinism, so nothing in the test itself would ever catch this
# (ops-security thermo review, Important 3). True (busy) iff free memory is
# under 2 GiB, or another compute process is already running. `nvidia-smi`
# missing entirely is not treated as "busy" -- there is no real evidence of
# contention, just no way to check; the atlas check then runs as before.
graph_lab_gpu_busy() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib < 2048 )); then
    return 0
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    return 0
  fi
  return 1
}

# `--build`: bundle the TS entries/workers, run the reproduction gate
# (unless skipped), then build and tag the image. Exact docker invocation
# per `.agents/plans/graph-lab/04-launch-and-runbook.md`'s change-surface
# table.
cmd_build() {
  command -v npm >/dev/null 2>&1 || die 'Missing command: npm'
  command -v docker >/dev/null 2>&1 || die 'Missing command: docker'
  npm run graph-lab:bundle
  if [[ "${GRAPH_LAB_SKIP_REPRO:-0}" == 1 ]]; then
    printf 'graph-lab: GRAPH_LAB_SKIP_REPRO=1 set; skipping the spark-marked reproduction checks entirely (including the CPU lesion/swap-set checks). Re-run them once the GPU is free: cd backend/graph_lab && uv run pytest -m spark -v\n' >&2
  elif graph_lab_gpu_busy; then
    printf 'graph-lab: WARNING: the GPU pre-flight found it busy (free memory below 2 GiB, or another compute process already running) -- skipping only the atlas reproduction check this run (it needs the GPU); the CPU lesion/swap-set checks still run as a hard gate. Consider GRAPH_LAB_SKIP_REPRO=1 to skip the whole gate outright, or re-run the atlas check manually once the GPU is free: cd backend/graph_lab && uv run pytest -m spark -k AtlasReproductionTests -v\n' >&2
    graph_lab_run_reproduction_checks 1
  else
    graph_lab_run_reproduction_checks 0
  fi
  docker build -t "$GRAPH_LAB_IMAGE" -f backend/graph_lab/Dockerfile .
  printf 'graph-lab: built and tagged %s\n' "$GRAPH_LAB_IMAGE"
}

# `--start`: validate configuration, ensure the network and egress-drop rule
# exist, then run the container. Refuses outright if the rule cannot be
# confirmed present afterward -- this is the one thing that makes the
# tailnet exposure safe.
cmd_start() {
  local env_file=${GRAPH_LAB_ENV_FILE:-.env}
  [[ -f "$env_file" ]] || die "Copy .env.example to $env_file and set GRAPH_LAB_TOKEN, GRAPH_LAB_ORIGINS, GRAPH_LAB_BIND, and DEPLOY_URL first."
  # Trusted local configuration file, not input from the server -- same
  # trust model as scripts/deploy.sh's `source .env`.
  # shellcheck disable=SC1090
  source "$env_file"

  local token=${GRAPH_LAB_TOKEN:-}
  [[ ${#token} -ge 16 ]] || die 'Set GRAPH_LAB_TOKEN to at least 16 characters.'

  local origins=${GRAPH_LAB_ORIGINS:-}
  [[ -n "$origins" ]] || die 'Set GRAPH_LAB_ORIGINS (comma-separated origins; no wildcards).'

  local bind=${GRAPH_LAB_BIND:-}
  [[ -n "$bind" ]] || die 'Set GRAPH_LAB_BIND (an address from `tailscale ip -4`).'

  local deploy_url=${DEPLOY_URL:-}
  [[ -n "$deploy_url" ]] || die 'Set DEPLOY_URL.'

  local origin
  origin=$(graph_lab_derive_origin "$deploy_url") || die 'DEPLOY_URL is not a valid http(s) URL; cannot derive its origin.'
  graph_lab_origin_in_list "$origin" "$origins" || die 'GRAPH_LAB_ORIGINS must include the origin derived from DEPLOY_URL.'

  local bind_status=0
  graph_lab_check_bind "$bind" || bind_status=$?
  case "$bind_status" in
    0) : ;;
    2) die 'GRAPH_LAB_BIND must exactly equal an address printed by `tailscale ip -4`, but that command printed no addresses at all -- tailscaled is likely not running or not connected. This check also rejects 0.0.0.0, LAN addresses, and an unset/empty value.' ;;
    *) die 'GRAPH_LAB_BIND must exactly equal an address printed by `tailscale ip -4`; that command printed at least one address, but none matched -- check for a stale or mistyped GRAPH_LAB_BIND. This check also rejects 0.0.0.0, LAN addresses, and an unset/empty value.' ;;
  esac

  command -v docker >/dev/null 2>&1 || die 'Missing command: docker'

  local subnet
  subnet=$(graph_lab_subnet)
  graph_lab_validate_subnet "$subnet"
  graph_lab_ensure_network "$subnet"

  # Never trust $subnet (config) alone for the rule once a network may
  # already exist: graph_lab_ensure_network above already refused on a real
  # mismatch, but re-deriving it here from docker directly (rather than
  # reusing the config value) keeps this check correct even if that guard
  # is ever weakened later (ops-security thermo review, Important 2).
  local rule_subnet
  rule_subnet=$(graph_lab_effective_subnet "$subnet")

  if ! graph_lab_rule_present "$rule_subnet"; then
    graph_lab_ensure_rule "$rule_subnet" || true
  fi
  graph_lab_rule_present "$rule_subnet" || die 'The DOCKER-USER egress-drop rule for the graph-lab subnet is not present and could not be inserted (sudo may be required, or iptables is unavailable). Refusing to start the container without it.'

  local port=${GRAPH_LAB_PORT:-8766}
  export GRAPH_LAB_TOKEN="$token" GRAPH_LAB_ORIGINS="$origins"
  # `docker run`'s own stderr on a real failure (e.g. "address already in
  # use") routinely echoes back its failing arguments, which here include
  # the tailnet bind address in the `-p` flag -- exactly what this script
  # must never print (see this file's header comment). Captured to a local
  # file instead of left connected to our own stderr; on failure, only a
  # generic, secret-free message is printed, pointing at that file for local
  # inspection rather than echoing its contents here. That file is
  # deliberately left on disk on failure (mode 0600, owner-only, per
  # `mktemp`'s default) rather than auto-deleted -- delete it yourself once
  # you're done inspecting it, since its name is random per attempt and this
  # script has no reliable way to find and clean up a prior run's copy.
  local run_err
  run_err=$(mktemp)
  if ! docker run -d --name "$GRAPH_LAB_CONTAINER_NAME" --network "$GRAPH_LAB_NETWORK_NAME" --init --gpus all \
    --memory 8g --cpus 4 --cap-drop ALL --security-opt no-new-privileges --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=1g \
    -p "${bind}:${port}:8000" \
    -e GRAPH_LAB_TOKEN -e GRAPH_LAB_ORIGINS \
    "$GRAPH_LAB_IMAGE" >/dev/null 2>"$run_err"; then
    die "docker run failed to start the container. Its stderr was saved to $run_err for local inspection, then delete it (it may include the configured bind address -- do not paste it into a shared channel). Common causes: the port is already bound, or the GPU is unavailable."
  fi
  rm -f -- "$run_err"

  printf 'graph-lab: started, bound to the tailnet interface (tailscale0). The DOCKER-USER egress-drop rule is present.\n'
}

# `--status`: reports network/rule/container state, and whether the
# container's configured bind still matches a live tailnet address, without
# ever printing the address itself -- only "tailscale0" or "MISMATCH".
cmd_status() {
  command -v docker >/dev/null 2>&1 || die 'Missing command: docker'
  local env_file=${GRAPH_LAB_ENV_FILE:-.env}
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi
  local subnet
  subnet=$(graph_lab_subnet)

  if docker network inspect "$GRAPH_LAB_NETWORK_NAME" >/dev/null 2>&1; then
    printf 'network: present (%s)\n' "$GRAPH_LAB_NETWORK_NAME"
    local actual_subnet
    actual_subnet=$(graph_lab_network_subnet) || actual_subnet=""
    if [[ -n "$actual_subnet" && "$actual_subnet" != "$subnet" ]]; then
      printf 'network subnet: MISMATCH -- the network is actually on %s, but the configured GRAPH_LAB_SUBNET is %s. --start will refuse until you run --uninstall then --start again.\n' "$actual_subnet" "$subnet"
    fi
  else
    printf 'network: MISSING (%s)\n' "$GRAPH_LAB_NETWORK_NAME"
  fi

  # Never trust $subnet (config) alone: check the rule against the
  # network's ACTUAL subnet when it exists (ops-security thermo review,
  # Important 2), matching cmd_start's own logic.
  local rule_subnet
  rule_subnet=$(graph_lab_effective_subnet "$subnet")
  if graph_lab_rule_present "$rule_subnet"; then
    printf 'egress-drop rule: present\n'
  else
    printf 'egress-drop rule: MISSING -- run --start to insert it (requires sudo)\n'
  fi

  local state
  state=$(docker inspect -f '{{.State.Running}}' "$GRAPH_LAB_CONTAINER_NAME" 2>/dev/null || printf 'absent')
  case "$state" in
    true)
      printf 'container: running\n'
      local bind=${GRAPH_LAB_BIND:-}
      if [[ -z "$bind" ]]; then
        printf 'bind: unknown (GRAPH_LAB_BIND not set in %s)\n' "$env_file"
      elif graph_lab_check_bind "$bind"; then
        printf 'bind: OK (tailscale0)\n'
      else
        printf 'bind: MISMATCH -- the configured address is no longer reported by `tailscale ip -4` (tailscaled may have restarted, or the tailnet address changed). Recovery: ./scripts/graph-lab.sh --stop && ./scripts/graph-lab.sh --start\n'
      fi
      ;;
    false) printf 'container: stopped\n' ;;
    *) printf 'container: not present\n' ;;
  esac
}

# `--stop`: stops and removes the container only. The egress-drop rule and
# network are left in place -- they only ever affect the dedicated subnet,
# so leaving them does not expose anything.
cmd_stop() {
  command -v docker >/dev/null 2>&1 || die 'Missing command: docker'
  docker stop "$GRAPH_LAB_CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$GRAPH_LAB_CONTAINER_NAME" >/dev/null 2>&1 || true
  printf 'graph-lab: stopped. The egress-drop rule and network are left in place.\n'
}

# `--uninstall`: removes the container, the egress-drop rule, and the
# network. Idempotent -- safe to run even if some or all of these are
# already gone.
cmd_uninstall() {
  command -v docker >/dev/null 2>&1 || die 'Missing command: docker'
  local env_file=${GRAPH_LAB_ENV_FILE:-.env}
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi
  local subnet
  subnet=$(graph_lab_subnet)
  graph_lab_validate_subnet "$subnet"
  # Remove the rule keyed on the network's ACTUAL subnet (read before the
  # network itself is torn down below), not just the configured value --
  # if GRAPH_LAB_SUBNET drifted from the network's real subnet (the exact
  # state graph_lab_ensure_network's mismatch check now prevents going
  # forward), the rule that actually needs removing is the one covering
  # the real subnet, not the currently-configured one.
  local rule_subnet
  rule_subnet=$(graph_lab_effective_subnet "$subnet")

  docker stop "$GRAPH_LAB_CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$GRAPH_LAB_CONTAINER_NAME" >/dev/null 2>&1 || true

  if graph_lab_rule_present "$rule_subnet"; then
    command -v sudo >/dev/null 2>&1 || die 'sudo is required to remove the DOCKER-USER egress-drop rule.'
    sudo iptables -D DOCKER-USER -s "$rule_subnet" -m conntrack --ctstate NEW -j DROP || true
  fi
  # Re-verify rather than trust the `-D` exit code (swallowed by `|| true`
  # above): a failed removal is not an exposure (a leftover DROP rule is
  # more restrictive, not less), but reporting "removed" when it is not
  # would be dishonest and could mislead a later --uninstall/--start.
  local rule_status
  if graph_lab_rule_present "$rule_subnet"; then
    rule_status='still present'
    printf 'graph-lab: WARNING: the egress-drop rule is still present after attempting removal. Remove it manually: sudo iptables -D DOCKER-USER -s %s -m conntrack --ctstate NEW -j DROP\n' "$rule_subnet" >&2
  else
    rule_status='removed'
  fi

  docker network rm "$GRAPH_LAB_NETWORK_NAME" >/dev/null 2>&1 || true
  local network_status
  if docker network inspect "$GRAPH_LAB_NETWORK_NAME" >/dev/null 2>&1; then
    network_status='still present'
    printf 'graph-lab: WARNING: the network %s is still present after attempting removal.\n' "$GRAPH_LAB_NETWORK_NAME" >&2
  else
    network_status='removed'
  fi

  printf 'graph-lab: uninstalled (container removed; egress-drop rule %s; network %s).\n' "$rule_status" "$network_status"
}

graph_lab_usage() {
  printf 'Usage: scripts/graph-lab.sh --build|--start|--status|--stop|--uninstall\n'
  printf '  --build      Bundle the TS entries, run the reproduction gate, build and tag %s.\n' "$GRAPH_LAB_IMAGE"
  printf '  --start      Ensure the network and egress-drop rule, then run the container bound to GRAPH_LAB_BIND.\n'
  printf '  --status     Report network/rule/container state without printing any address or token.\n'
  printf '  --stop       Stop and remove the container. Leaves the network and egress-drop rule in place.\n'
  printf '  --uninstall  Remove the container, the egress-drop rule, and the network.\n'
}

graph_lab_main() {
  graph_lab_refuse_test_overrides
  [[ $# -eq 1 ]] || { graph_lab_usage >&2; exit 1; }
  case "$1" in
    --build) cmd_build ;;
    --start) cmd_start ;;
    --status) cmd_status ;;
    --stop) cmd_stop ;;
    --uninstall) cmd_uninstall ;;
    --help) graph_lab_usage ;;
    *) die 'Unknown option; use --help.' ;;
  esac
}

# Only dispatch the CLI when this file is executed directly. When it is
# `source`d (scripts/verify/graph-lab.test.sh), every function above is
# defined and callable, but nothing runs automatically -- see this file's
# header comment.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  graph_lab_main "$@"
fi
