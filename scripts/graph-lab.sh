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

# True iff $1 is non-empty and exactly matches one address printed by
# `tailscale ip -4`. This is what rejects `0.0.0.0`, LAN addresses, and an
# empty/unset bind: none of those can ever appear in that command's output.
graph_lab_check_bind() {
  local bind=$1 addr
  [[ -n "$bind" ]] || return 1
  while IFS= read -r addr; do
    [[ -n "$addr" ]] || continue
    [[ "$addr" == "$bind" ]] && return 0
  done < <(tailscale ip -4 2>/dev/null || true)
  return 1
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

# Creates the dedicated bridge network with a fixed subnet, idempotently.
graph_lab_ensure_network() {
  local subnet=$1
  docker network inspect "$GRAPH_LAB_NETWORK_NAME" >/dev/null 2>&1 && return 0
  docker network create --subnet "$subnet" "$GRAPH_LAB_NETWORK_NAME" >/dev/null \
    || die "Failed to create the docker network $GRAPH_LAB_NETWORK_NAME (subnet $subnet)."
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
graph_lab_run_reproduction_checks() {
  command -v uv >/dev/null 2>&1 || die 'uv not found on PATH; cannot run the reproduction checks. Install uv, or explicitly skip with GRAPH_LAB_SKIP_REPRO=1 (only while another GPU job needs the GPU) and re-run them once uv is available: cd backend/graph_lab && uv run pytest -m spark -v'
  printf 'graph-lab: running the lesion/swap-set reproduction checks (exact-match; hard gate)...\n'
  if ! (cd backend/graph_lab && uv run --locked python -I -m pytest -q -m spark -k 'not AtlasReproductionTests'); then
    die 'the lesion/swap-set reproduction checks failed (exact-match, deterministic). Refusing to build or tag flyarena-graph-lab:local. See backend/graph_lab/tests/test_reproduction.py.'
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

# `--build`: bundle the TS entries/workers, run the reproduction gate
# (unless skipped), then build and tag the image. Exact docker invocation
# per `.agents/plans/graph-lab/04-launch-and-runbook.md`'s change-surface
# table.
cmd_build() {
  command -v npm >/dev/null 2>&1 || die 'Missing command: npm'
  command -v docker >/dev/null 2>&1 || die 'Missing command: docker'
  npm run graph-lab:bundle
  if [[ "${GRAPH_LAB_SKIP_REPRO:-0}" == 1 ]]; then
    printf 'graph-lab: GRAPH_LAB_SKIP_REPRO=1 set; skipping the spark-marked reproduction checks. Re-run them once the GPU is free: cd backend/graph_lab && uv run pytest -m spark -v\n' >&2
  else
    graph_lab_run_reproduction_checks
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

  graph_lab_check_bind "$bind" || die 'GRAPH_LAB_BIND must exactly equal an address printed by `tailscale ip -4`. This rejects 0.0.0.0, LAN addresses, and an unset/empty value.'

  command -v docker >/dev/null 2>&1 || die 'Missing command: docker'

  local subnet
  subnet=$(graph_lab_subnet)
  graph_lab_ensure_network "$subnet"

  if ! graph_lab_rule_present "$subnet"; then
    graph_lab_ensure_rule "$subnet" || true
  fi
  graph_lab_rule_present "$subnet" || die 'The DOCKER-USER egress-drop rule for the graph-lab subnet is not present and could not be inserted (sudo may be required, or iptables is unavailable). Refusing to start the container without it.'

  local port=${GRAPH_LAB_PORT:-8766}
  export GRAPH_LAB_TOKEN="$token" GRAPH_LAB_ORIGINS="$origins"
  # `docker run`'s own stderr on a real failure (e.g. "address already in
  # use") routinely echoes back its failing arguments, which here include
  # the tailnet bind address in the `-p` flag -- exactly what this script
  # must never print (see this file's header comment). Captured to a local
  # file instead of left connected to our own stderr; on failure, only a
  # generic, secret-free message is printed, pointing at that file for local
  # inspection rather than echoing its contents here.
  local run_err
  run_err=$(mktemp)
  if ! docker run -d --name "$GRAPH_LAB_CONTAINER_NAME" --network "$GRAPH_LAB_NETWORK_NAME" --init --gpus all \
    --memory 8g --cpus 4 --cap-drop ALL --security-opt no-new-privileges --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=1g \
    -p "${bind}:${port}:8000" \
    -e GRAPH_LAB_TOKEN -e GRAPH_LAB_ORIGINS \
    "$GRAPH_LAB_IMAGE" >/dev/null 2>"$run_err"; then
    die "docker run failed to start the container. Its stderr was saved to $run_err for local inspection (it may include the configured bind address -- do not paste it into a shared channel). Common causes: the port is already bound, or the GPU is unavailable."
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
  else
    printf 'network: MISSING (%s)\n' "$GRAPH_LAB_NETWORK_NAME"
  fi

  if graph_lab_rule_present "$subnet"; then
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

  docker stop "$GRAPH_LAB_CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$GRAPH_LAB_CONTAINER_NAME" >/dev/null 2>&1 || true

  if graph_lab_rule_present "$subnet"; then
    command -v sudo >/dev/null 2>&1 || die 'sudo is required to remove the DOCKER-USER egress-drop rule.'
    sudo iptables -D DOCKER-USER -s "$subnet" -m conntrack --ctstate NEW -j DROP || true
  fi
  # Re-verify rather than trust the `-D` exit code (swallowed by `|| true`
  # above): a failed removal is not an exposure (a leftover DROP rule is
  # more restrictive, not less), but reporting "removed" when it is not
  # would be dishonest and could mislead a later --uninstall/--start.
  local rule_status
  if graph_lab_rule_present "$subnet"; then
    rule_status='still present'
    printf 'graph-lab: WARNING: the egress-drop rule is still present after attempting removal. Remove it manually: sudo iptables -D DOCKER-USER -s %s -m conntrack --ctstate NEW -j DROP\n' "$subnet" >&2
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
