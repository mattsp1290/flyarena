#!/usr/bin/env bash
#
# .agents/plans/rewiring-null/03-trained-sample.md's train-sample.sh: CEM-trains
# one readout per rewired graph (rewiring seeds 0..19 by default) with the
# EXACT production CEM config flyarena-bigq's biological replicas used --
# population 128, elites 32, generations 150, alpha 0.7, stdFloor 0.02,
# initStd 0.5, E (training seeds/generation) 16, H (hidden size) 16, ticks
# 1800 -- so the null's 20 rewired scores and bigq's 3 biological replicas
# differ only in graph topology / replica seed, never in hyperparameters.
#
# population/elites/generations/train-seeds-per-generation/alpha/std-floor/
# init-std/hidden-size are READ DIRECTLY from
# public/data/trained-readout-v1.manifest.json's "training" block (+ its
# top-level "H") at every invocation, below -- never hard-coded literals a
# future manifest regeneration (or a copy-paste of this script for a similar
# future study) could silently drift from. This runs before any GPU time is
# spent (before the seed loop, before the first export-arms/flyarena-train
# call) and fails fast with a clear error if the manifest is missing or
# malformed (a thermo-architecture review finding: an earlier version
# hard-coded these as literals with a comment claiming they were "verified
# byte-for-byte against that manifest block before this script was written"
# -- a one-time, manual, unenforced claim with no repeatable check). An
# explicit `--population`/`--elites`/etc. flag still overrides its
# manifest-derived default, exactly as before (see the `--dry-run-fixture`
# fast-smoke-test example below, which overrides the population/generations/
# etc. flags that affect run time). `ticks` is NOT part of the manifest's
# "training" block (it is recorded per-run-directory, not per-study) and
# stays this script's own literal default, unaffected by the above.
#
# For each rewiring seed s:
#   1. `npm run training:export-arms` exports the rewired graph's CSR arrays
#      as an export-arms.ts bundle (never re-derived in Python -- the
#      trained-readout plan's "Arm CSR arrays exported from TypeScript" key
#      decision). The bundle lands at
#      <arms-out>/seed<s>/<sha256 of --graph>/rewired.json -- this script
#      computes that sha256 itself (sha256sum, matching export-arms.ts's own
#      `sha256Hex(raw file bytes)`) and asserts the bundle exists before
#      training, rather than letting a missing/misplaced bundle surface only
#      as a much-later, much-less-specific flyarena-train error.
#   2. `flyarena-train` (via training/scripts/run.sh, which disables this
#      host's Datadog APM auto-injection -- see training/README.md) trains a
#      readout against that bundle with the exact CEM config above and
#      `--replica-seed 101` (isolating topology from trainer-seed variance --
#      see this plan's "Key decisions"), writing
#      <trained-out>/seed<s>/{theta_final.npy,config.json,env.json,generations.csv}.
#
# Resumable: a seed whose <trained-out>/seed<s>/config.json already exists
# is skipped entirely (export-arms is not re-run for it either).
# Fail-fast (`set -euo pipefail`): the first failed step aborts the whole
# script immediately, leaving every earlier seed's completed run directory
# untouched. `flyarena-train` itself deletes any pre-existing config.json in
# its --out directory BEFORE training (training/src/flyarena_training/cli.py's
# run_training) and writes theta_final.npy/theta_best.npy FIRST, config.json
# LAST -- so config.json's existence, not theta_final.npy's, is the true
# "this run finished" signal. (A dual-review finding: an earlier version of
# this script checked theta_final.npy instead, which np.save writes
# non-atomically and BEFORE config.json -- a run killed between those two
# writes would have been treated as complete and skipped forever, even
# though config.json describes the OLD run, not this one, or is simply
# missing.) Recovery from a truncated/corrupt theta_final.npy (caught
# downstream by null-trained-evaluate.ts, which fails loudly on a
# short/malformed .npy, or by this same resumability check if config.json
# itself never got written): delete that seed's <trained-out>/seed<s>/
# directory and rerun this script; it regenerates only that seed.
#
# Usage:
#   scripts/null/train-sample.sh
#       # the real 20-rewiring run (~14h projected on this host at G=150 --
#       # see the WP3 report's calibration numbers; NOT run by default here)
#   scripts/null/train-sample.sh --dry-run-fixture --seed-count 2 \
#       --generations 2 --population 8 --elites 4 --train-seeds-per-generation 2
#       # fast smoke test against the trace-graph fixture, no GPU/real assets
#       # required
#
# --dry-run-fixture switches both steps to
# tests/fixtures/trace-graph.ts (via export-arms.ts's --fixture-rewire),
# never usable with --graph/--rewired-index (a real graph's rewired arm must
# come from the offline-compiled --rewired artifact -- see export-arms.ts's
# own doc comment) -- so a fixture run can never be mistaken for, or
# accidentally overwrite, the real study's output: its --arms-out/--trained-out
# default to a distinct "-dry-run-fixture" suffixed directory.
#
# TRAIN_SAMPLE_MANIFEST_PATH (env var, not a flag): overrides which manifest
# the CEM-config preflight below reads. Only meant for this script's own
# tests (pointing at a fixture manifest to prove the preflight fails fast on
# a missing/malformed one) -- a real invocation should never set it, so it
# always reads the real, shipped public/data/trained-readout-v1.manifest.json.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
cd "$repo_root"

dry_run_fixture=0
seed_start=0
seed_count=20
replica_seed=101
# task-generality WP1 (.agents/plans/task-generality/01-task-plumbing.md):
# src/lib/arena/tasks.ts's ARENA_TASKS id, passed to flyarena-train's own
# --arena-task at both invocation sites below. "default" (ARENA_CONFIG,
# unchanged) unless overridden.
arena_task="default"
# population/elites/generations/train_seeds_per_generation/alpha/std_floor/
# init_std/hidden_size are intentionally left UNSET here -- read_manifest_cem_config
# (below) fills each one that is still unset after CLI parsing from
# public/data/trained-readout-v1.manifest.json, so there is no hard-coded
# literal here to silently drift from it. An explicit --population/--elites/
# etc. flag (parsed below) still overrides its manifest-derived default.
population=""
elites=""
generations=""
train_seeds_per_generation=""
alpha=""
std_floor=""
init_std=""
hidden_size=""
ticks=1800
graph_path="public/data/malecns-arena-v1.bin.gz"
graphs_dir="training/runs/null/graphs"
arms_out="training/runs/null/arms"
trained_out="training/runs/null/trained"
arms_out_explicit=0
trained_out_explicit=0
# --graph-list/--ids (`.agents/plans/pathway-interventions/03-evaluation.md`'s
# WP3): trains one readout per id in a `scripts/analysis/interventions.py`
# `index.json` (P/Q/C000.../M1000...) instead of a `rewire_batch.py`
# seed-keyed `--graphs-dir`. `graph_list_mode` (set at the flags themselves,
# below) is what actually switches the seed loop over to id-keyed behavior.
graph_list=""
ids=""
graph_list_mode=0
# training/runs/interventions/{arms,trained} -- WP1's own gitignored output
# root, kept separate from train-sample.sh's existing training/runs/null/*
# defaults so a --graph-list run can never collide with (or be mistaken
# for) the rewired-seed study's own arms/trained trees. Only applied when
# --graph-list/--ids is used AND the operator did not pass an explicit
# --arms-out/--trained-out of their own (see right after CLI parsing) --
# an explicit flag always wins, exactly like --dry-run-fixture's own
# directory suffixing.
DEFAULT_GRAPH_LIST_ARMS_OUT="training/runs/interventions/arms"
DEFAULT_GRAPH_LIST_TRAINED_OUT="training/runs/interventions/trained"

# A RELATIVE --graph/--graphs-dir/--arms-out/--trained-out resolves against
# repo_root (this script `cd`s there below), NOT against the directory this
# script happens to be invoked from -- unlike null-trained-evaluate.ts's own
# CLI flags, which resolve relative paths against `process.cwd()` (the
# normal Node-CLI convention). This is a deliberate difference, not an
# oversight (a round-2 dual-review finding asked that it be documented
# rather than "fixed" either way, since both conventions are individually
# correct and reconciling them risks a regression under time pressure): this
# script's defaults are themselves repo-root-relative paths, and it is most
# often run via cron/automation from an arbitrary cwd while always meaning
# "this repo's training/runs/ tree" -- so repo-root-relative is the more
# useful default here. Pass an ABSOLUTE path (`to_abs_path` below leaves it
# unchanged) for a custom location, when invoking this script and
# null-trained-evaluate.ts together and comparing --trained-out to
# --rewired-trained-dir, etc.

usage() {
  echo "Usage: $0 [--dry-run-fixture] [--seed-start N] [--seed-count N] [--replica-seed N] [--arena-task ID]" >&2
  echo "          [--population N] [--elites N] [--generations N] [--train-seeds-per-generation N]" >&2
  echo "          [--alpha F] [--std-floor F] [--init-std F] [--hidden-size N] [--ticks N]" >&2
  echo "          [--graph PATH] [--graphs-dir DIR] [--arms-out DIR] [--trained-out DIR]" >&2
  echo "          [--graph-list index.json --ids id1,id2,...]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run-fixture) dry_run_fixture=1; shift ;;
    --seed-start) seed_start="${2:?}"; shift 2 ;;
    --seed-count) seed_count="${2:?}"; shift 2 ;;
    --replica-seed) replica_seed="${2:?}"; shift 2 ;;
    --arena-task) arena_task="${2:?}"; shift 2 ;;
    --population) population="${2:?}"; shift 2 ;;
    --elites) elites="${2:?}"; shift 2 ;;
    --generations) generations="${2:?}"; shift 2 ;;
    --train-seeds-per-generation) train_seeds_per_generation="${2:?}"; shift 2 ;;
    --alpha) alpha="${2:?}"; shift 2 ;;
    --std-floor) std_floor="${2:?}"; shift 2 ;;
    --init-std) init_std="${2:?}"; shift 2 ;;
    --hidden-size) hidden_size="${2:?}"; shift 2 ;;
    --ticks) ticks="${2:?}"; shift 2 ;;
    --graph) graph_path="${2:?}"; shift 2 ;;
    --graphs-dir) graphs_dir="${2:?}"; shift 2 ;;
    --arms-out) arms_out="${2:?}"; arms_out_explicit=1; shift 2 ;;
    --trained-out) trained_out="${2:?}"; trained_out_explicit=1; shift 2 ;;
    --graph-list) graph_list="${2:?}"; graph_list_mode=1; shift 2 ;;
    --ids) ids="${2:?}"; graph_list_mode=1; shift 2 ;;
    -h|--help) usage ;;
    *) echo "train-sample.sh: unknown argument: $1" >&2; usage ;;
  esac
done

# --graph-list/--ids form one mode together: an id list with no index.json
# to resolve it against (or vice versa) cannot proceed -- EXCEPT under
# --dry-run-fixture, which (like the existing seed-based dry run) never
# reads a real index.json at all; there, --ids alone selects id-keyed
# fixture-rewire exports (one per id, in --ids order) instead of the
# existing numeric --seed-start/--seed-count range. --graph-list itself is
# refused under --dry-run-fixture (mirroring export-arms.ts's own
# "--fixture-rewire cannot be combined with --graph" rule): a fixture run
# must never be able to point at, verify against, or be mistaken for a real
# WP1 index.json.
if [[ "$graph_list_mode" -eq 1 && "$dry_run_fixture" -eq 0 ]]; then
  if [[ -z "$graph_list" || -z "$ids" ]]; then
    echo "train-sample.sh: --graph-list and --ids must be given together" >&2
    usage
  fi
fi
if [[ "$dry_run_fixture" -eq 1 && -n "$graph_list" ]]; then
  echo "train-sample.sh: --graph-list cannot be combined with --dry-run-fixture (pass --ids alone for an id-keyed fixture dry run)" >&2
  usage
fi
if [[ "$dry_run_fixture" -eq 1 && "$graph_list_mode" -eq 1 && -z "$ids" ]]; then
  echo "train-sample.sh: --dry-run-fixture's id-keyed mode requires --ids" >&2
  usage
fi

if [[ "$graph_list_mode" -eq 1 ]]; then
  # See DEFAULT_GRAPH_LIST_ARMS_OUT/DEFAULT_GRAPH_LIST_TRAINED_OUT's own doc
  # comment above: an explicit --arms-out/--trained-out always wins.
  if [[ "$arms_out_explicit" -eq 0 ]]; then
    arms_out="$DEFAULT_GRAPH_LIST_ARMS_OUT"
  fi
  if [[ "$trained_out_explicit" -eq 0 ]]; then
    trained_out="$DEFAULT_GRAPH_LIST_TRAINED_OUT"
  fi
fi

# --seed-start/--seed-count feed the `(( seed = seed_start; ...))` C-style
# `for` loop below -- a non-integer value there fails with bash's own
# unhelpful "syntax error in expression" rather than this script's own
# argument-specific error message, so it is validated explicitly here.
if [[ ! "$seed_start" =~ ^[0-9]+$ ]]; then
  echo "train-sample.sh: --seed-start must be a non-negative integer, got \"$seed_start\"" >&2
  exit 1
fi
if [[ ! "$seed_count" =~ ^[0-9]+$ ]] || [[ "$seed_count" -eq 0 ]]; then
  echo "train-sample.sh: --seed-count must be a positive integer, got \"$seed_count\"" >&2
  exit 1
fi
# --graph-list/--ids mode's id_out/config_path embed `$replica_seed` verbatim
# as "<id>-seed<replica_seed>" (a review finding): an out-of-canonical-form
# value ("0101", "+101") would still train correctly (flyarena-train's own
# argparse `type=int` canonicalizes it for `config.json`'s `trainerSeed`),
# but would write to a directory name (e.g. "P-seed0101") that no later
# invocation with the canonical "101" -- including null-trained-evaluate.ts's
# own "<id>-seed<trainerSeed>" lookup -- would ever find, silently wasting a
# full retrain. Required to already be in canonical (no leading zero, no
# sign) form, not merely integer-valued.
if [[ ! "$replica_seed" =~ ^[1-9][0-9]*$ ]]; then
  echo "train-sample.sh: --replica-seed must be a positive integer with no leading zeros, got \"$replica_seed\"" >&2
  exit 1
fi

# Reads public/data/trained-readout-v1.manifest.json's "training" block
# (population/elites/generations/trainingSeedsPerGeneration/alpha/stdFloor/
# initStd) plus its top-level "H", validating every field's presence/type,
# and prints each value on its own stdout line, in that fixed order. Exits 1
# with a clear stderr message (never a silent `KeyError`/`None`) on a
# missing/unreadable/malformed manifest or a missing/wrong-typed field --
# this is the CEM-config pre-flight itself: it runs (below) before any
# export-arms/flyarena-train call, so a bad manifest fails in milliseconds,
# not after however much of a real run has already completed. `ticks` is
# deliberately not read here -- see this script's module doc comment.
read_manifest_cem_config() {
  local manifest_path="$1"
  python3 - "$manifest_path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path) as f:
        text = f.read()
except OSError as exc:
    sys.stderr.write(f"train-sample.sh: cannot read manifest {path}: {exc}\n")
    sys.exit(1)

try:
    manifest = json.loads(text)
except json.JSONDecodeError as exc:
    sys.stderr.write(f"train-sample.sh: {path} is not valid JSON: {exc}\n")
    sys.exit(1)

if not isinstance(manifest, dict):
    sys.stderr.write(f"train-sample.sh: {path} does not contain a JSON object\n")
    sys.exit(1)

training = manifest.get("training")
if not isinstance(training, dict):
    sys.stderr.write(f"train-sample.sh: {path} has no \"training\" object\n")
    sys.exit(1)


def require_number(container, key, label, integer=False):
    value = container.get(key)
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or (integer and float(value) != int(value)):
        kind = "an integer" if integer else "a number"
        sys.stderr.write(f"train-sample.sh: {path}'s {label} is missing or not {kind} (got {value!r})\n")
        sys.exit(1)
    # Cast to `int` (not just validated as mathematically integral) before
    # this is ever printed below: a manifest that happens to serialize an
    # integral value as a JSON float (e.g. "150.0") would otherwise flow
    # through as the literal string "150.0" into `--generations`/etc., which
    # `flyarena_training.cli`'s `type=int` argparse flags reject -- exactly
    # the kind of manifest drift this pre-flight exists to catch cleanly,
    # not turn into an unrelated argparse traceback (a review finding).
    return int(value) if integer else value


population = require_number(training, "population", "training.population", integer=True)
elites = require_number(training, "elites", "training.elites", integer=True)
generations = require_number(training, "generations", "training.generations", integer=True)
train_seeds_per_generation = require_number(
    training, "trainingSeedsPerGeneration", "training.trainingSeedsPerGeneration", integer=True
)
alpha = require_number(training, "alpha", "training.alpha")
std_floor = require_number(training, "stdFloor", "training.stdFloor")
init_std = require_number(training, "initStd", "training.initStd")
hidden_size = require_number(manifest, "H", "H", integer=True)

for value in (population, elites, generations, train_seeds_per_generation, alpha, std_floor, init_std, hidden_size):
    print(value)
PY
}

manifest_path="${TRAIN_SAMPLE_MANIFEST_PATH:-${repo_root}/public/data/trained-readout-v1.manifest.json}"
manifest_output="$(read_manifest_cem_config "$manifest_path")"
mapfile -t manifest_cem_values <<< "$manifest_output"
if [[ "${#manifest_cem_values[@]}" -ne 8 ]]; then
  echo "train-sample.sh: expected 8 values from read_manifest_cem_config, got ${#manifest_cem_values[@]}" >&2
  exit 1
fi

# Only fills a field CLI parsing above left unset -- an explicit --population/
# --elites/etc. flag always wins (see this script's module doc comment and
# the --dry-run-fixture fast-smoke-test usage example, which overrides
# several of these for speed).
population="${population:-${manifest_cem_values[0]}}"
elites="${elites:-${manifest_cem_values[1]}}"
generations="${generations:-${manifest_cem_values[2]}}"
train_seeds_per_generation="${train_seeds_per_generation:-${manifest_cem_values[3]}}"
alpha="${alpha:-${manifest_cem_values[4]}}"
std_floor="${std_floor:-${manifest_cem_values[5]}}"
init_std="${init_std:-${manifest_cem_values[6]}}"
hidden_size="${hidden_size:-${manifest_cem_values[7]}}"

if [[ "$dry_run_fixture" -eq 1 ]]; then
  # Never share a directory with the real study's output -- see this
  # script's module doc comment.
  arms_out="${arms_out}-dry-run-fixture"
  trained_out="${trained_out}-dry-run-fixture"
fi

# K = NEURAL_SUBSTEPS_PER_TICK, read from the TS source at run time (never
# hard-coded in this script) -- .agents/plans/rewiring-null/00-overview.md's
# "Assumption: NEURAL_SUBSTEPS_PER_TICK ... is read at run time, not
# hard-coded."
substeps_file="src/lib/connectome/constants.ts"
# `|| true`: under `set -e`, a failed `grep -oP` (no match) would otherwise
# abort the script right here with no message, skipping the `-z` check below
# entirely -- a dual-review finding. `|| true` lets a no-match fall through
# to that check instead, which reports a clear, specific error.
substeps="$(grep -oP 'export const NEURAL_SUBSTEPS_PER_TICK = \K[0-9]+' "$substeps_file" || true)"
if [[ -z "$substeps" ]]; then
  echo "train-sample.sh: could not read NEURAL_SUBSTEPS_PER_TICK from $substeps_file" >&2
  exit 1
fi

# `run.sh` (training/scripts/run.sh) `cd`s into training/ before `exec`ing
# `uv run`, so any relative path this script hands to `flyarena-train` must
# first be resolved against `repo_root`, not left relative (it would then
# resolve against training/ instead). An ALREADY-absolute path (an operator
# passing an absolute --arms-out/--trained-out/--graph) must be returned
# unchanged -- naively prefixing `repo_root` onto an absolute path produces
# a broken `<repo_root>/<absolute path>` (a dual-review finding: an earlier
# version always prefixed `${repo_root}/`, which is correct only for the
# relative-path default case).
to_abs_path() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "${repo_root}/${p}"
  fi
}

export PATH="${HOME}/.nvm/versions/node/v22.22.3/bin:${PATH}"

# Looks up rewire_batch.py's recorded artifact filename for one rewiring seed
# in <graphs-dir>/index.json (the same file null-evaluate.ts's RewireIndex
# reads) -- data-driven rather than assuming a fixed
# "malecns-arena-v1-rewired-seed<N>.bin.gz" naming convention, since that
# name is derived from --graph's own graphId, not fixed by this script.
#
# Shells out to scripts/null/lookup-rewired-artifact.ts, which reuses
# null-evaluate.ts's already-validated `readRewireIndex`/`RewireIndex`
# schema, instead of a second, disconnected `index.json` parser hand-written
# in Python here (a thermo-maintainability review finding: the previous
# inline python3 heredoc only did unvalidated `entry["seed"]`/
# `entry["artifact"]` dict lookups, so a renamed/malformed field would raise
# a generic `KeyError` here instead of a specific, actionable error).
rewired_artifact_for_seed() {
  local index_path="$1" seed="$2"
  node --import tsx scripts/null/lookup-rewired-artifact.ts "$index_path" "$seed"
}

# `--graph-list`/`--ids`'s own lookup, mirroring `rewired_artifact_for_seed`
# above but for `scripts/analysis/interventions.py`'s id-keyed index.json --
# see `scripts/null/lookup-intervention-graph.ts`'s doc comment for why this
# is a separate script/module rather than an addition to
# `lookup-rewired-artifact.ts`/`null-evaluate.ts`. Verifies the resolved
# graph's gzip sha256 against the index BEFORE printing its path (so a
# corrupted/stale/mismatched graph fails here, before any GPU time is
# spent), unlike `rewired_artifact_for_seed`, which only ever returns a
# filename for its caller to separately check `-f` on.
verified_intervention_graph_for_id() {
  local index_path="$1" id="$2"
  node --import tsx scripts/null/lookup-intervention-graph.ts "$index_path" "$id"
}

# Reused by two checks below (a review finding, both independently):
# (1) before trusting the resumability skip, that an existing config.json
#     actually describes the run being requested right now, not a stale or
#     differently-configured run (a calibration run left in the default
#     --trained-out, an interrupted smoke test, ...); (2) after exporting an
#     arm bundle (or finding one already on disk), that its
#     `provenance.artifactSha256` still matches the id's `--graph-list`
#     entry, closing the gap between `verified_intervention_graph_for_id`'s
#     own read of the graph file and `export-arms`'s separate, later read of
#     the same file.
verify_resumed_config_matches_request() {
  local config_path="$1"
  python3 - "$config_path" "$replica_seed" "$population" "$elites" "$generations" \
    "$train_seeds_per_generation" "$alpha" "$std_floor" "$init_std" "$hidden_size" "$ticks" "$arena_task" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path) as f:
    c = json.load(f)

want = {
    "arm": "rewired",
    "trainerSeed": int(sys.argv[2]),
    "population": int(sys.argv[3]),
    "elites": int(sys.argv[4]),
    "generations": int(sys.argv[5]),
    "trainingSeedsPerGeneration": int(sys.argv[6]),
    "alpha": float(sys.argv[7]),
    "stdFloor": float(sys.argv[8]),
    "initStd": float(sys.argv[9]),
    "H": int(sys.argv[10]),
    "ticks": int(sys.argv[11]),
    "arenaTask": sys.argv[12],
}
mismatches = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if mismatches:
    sys.stderr.write(f"{config_path} does not match the requested run: {mismatches}\n")
    sys.exit(1)
PY
}

verify_arm_bundle_matches_graph() {
  local bundle_path="$1" expected_gzip_sha256="$2"
  python3 - "$bundle_path" "$expected_gzip_sha256" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    bundle = json.load(f)
provenance = bundle.get("provenance") or {}
if provenance.get("kind") != "rewired-artifact" or provenance.get("artifactSha256") != sys.argv[2]:
    sys.exit(1)
PY
}

mkdir -p "$trained_out"

if [[ "$graph_list_mode" -eq 1 ]]; then
  if [[ "$dry_run_fixture" -eq 0 && ! -f "$graph_list" ]]; then
    echo "train-sample.sh: --graph-list ${graph_list} not found" >&2
    exit 1
  fi

  # A review finding: --ids must be exactly one comma-separated line with no
  # whitespace. `IFS=',' read -r -a` silently reads only the FIRST LINE of
  # its input, so an --ids value containing an embedded newline (e.g.
  # `--ids "$(cat ids.txt)"` with one id per line) would otherwise silently
  # train only the first id and report success. Whitespace around an id
  # (e.g. "C000, C001") would otherwise become a literal " C001" id that
  # fails lookup only once the loop reaches it.
  if [[ "$ids" == *$'\n'* ]]; then
    echo "train-sample.sh: --ids must not contain a newline (a single comma-separated line)" >&2
    exit 1
  fi
  if [[ "$ids" =~ [[:space:]] ]]; then
    echo "train-sample.sh: --ids must not contain whitespace (a comma-separated list with no spaces)" >&2
    exit 1
  fi

  # Split --ids on commas into an array, preserving the caller's own order
  # (never re-sorted): this script's own stdout/log order should match what
  # the operator asked for, and downstream rescoring
  # (null-trained-evaluate.ts's --graph-list mode) establishes its own
  # canonical output order independently -- this loop's order is a
  # scheduling convenience only, not a correctness requirement.
  IFS=',' read -r -a id_list <<< "$ids"
  if [[ "${#id_list[@]}" -eq 0 ]]; then
    echo "train-sample.sh: --ids must list at least one id" >&2
    exit 1
  fi

  declare -A seen_ids=()
  for id in "${id_list[@]}"; do
    if [[ -z "$id" ]]; then
      echo "train-sample.sh: --ids has an empty entry (check for a stray comma)" >&2
      exit 1
    fi
    # A review finding (defense in depth): every id feeds directly into
    # filesystem paths ("${trained_out}/${id}-seed${replica_seed}",
    # "${arms_out}/${id}"), and the resumability check above tests
    # `-f "$config_path"` before any id is ever looked up against
    # --graph-list (and --dry-run-fixture never looks ids up at all) -- so
    # restricting ids to a safe character set here, unconditionally, closes
    # that gap rather than relying on the lookup's sha check alone.
    if [[ ! "$id" =~ ^[A-Za-z0-9_-]+$ ]]; then
      echo "train-sample.sh: --ids has an invalid id \"${id}\" (letters, digits, underscore, hyphen only)" >&2
      exit 1
    fi
    if [[ -n "${seen_ids[$id]:-}" ]]; then
      echo "train-sample.sh: --ids lists \"${id}\" more than once" >&2
      exit 1
    fi
    seen_ids[$id]=1
  done

  # Preflight (a review finding): resolve and sha256-verify EVERY requested
  # id against --graph-list before training ANY of them. Without this, a
  # typo, an id missing from the index, or a corrupted graph at position k
  # of a long (e.g. this study's 10-id, multi-hour) batch would only surface
  # once the training loop reached it -- after k-1 ids' worth of real GPU
  # time already ran, with the remaining queue then sitting idle until an
  # operator notices. This costs only a few seconds (hashing small gzip
  # files) and never touches flyarena-train. Skipped entirely under
  # --dry-run-fixture, which never reads a real --graph-list at all.
  if [[ "$dry_run_fixture" -eq 0 ]]; then
    for id in "${id_list[@]}"; do
      verified_intervention_graph_for_id "$graph_list" "$id" >/dev/null
    done
  fi

  id_index=0
  for id in "${id_list[@]}"; do
    # id-and-trainer-seed-keyed, NOT id-only: this study trains "P" at three
    # different --replica-seed values (101/202/303) across three separate
    # invocations of this script, and each one is its own run directory --
    # keying on id alone would make the second/third invocation's
    # resumability check find the FIRST invocation's (different-trainer-seed)
    # config.json and wrongly skip training entirely.
    id_out="${trained_out}/${id}-seed${replica_seed}"
    config_path="${id_out}/config.json"

    if [[ -f "$config_path" ]]; then
      # A review finding: an existing config.json is only trusted as "this
      # run is done" once it is confirmed to actually describe THIS request
      # (same arm/trainerSeed/CEM hyperparameters) -- never merely because a
      # file happens to exist at this path (e.g. a calibration run, or a
      # smoke test, that reused the default --trained-out).
      if ! verify_resumed_config_matches_request "$config_path"; then
        echo "train-sample.sh: id ${id}: ${config_path} exists but does NOT match this run's requested config (see stderr above) -- move it aside (a stale/calibration run?) and rerun" >&2
        exit 1
      fi
      echo "train-sample.sh: id ${id} (seed ${replica_seed}): ${config_path} already exists and matches -- skipping (resumable, config verified)"
      id_index=$((id_index + 1))
      continue
    fi

    # Exported once per id (not once per id+trainer-seed): the rewired arm
    # bundle itself doesn't depend on the trainer seed, only training does --
    # matches this script's existing seed-based mode, which likewise exports
    # arms once per rewiring seed regardless of --replica-seed. A review
    # finding: re-exporting unconditionally on every invocation risked
    # silently orphaning an already-trained run's bundle (the bundle's
    # self-hash includes `provenance.artifactPath`, an ABSOLUTE path built by
    # `lookup-intervention-graph.ts` -- re-exporting from a different
    # worktree/checkout, or after a moved --graph-list, would rewrite the
    # SAME id's bundle with a DIFFERENT sha256, and an already-trained run's
    # config.json still points at the old one). An existing, still-verified
    # bundle is now reused instead of being unconditionally overwritten.
    arms_id_out="${arms_out}/${id}"

    if [[ "$dry_run_fixture" -eq 1 ]]; then
      echo "train-sample.sh: id ${id}: exporting fixture-rewired arms (fixture-rewire-seed=${id_index})"
      npm run training:export-arms -- --fixture-rewire --fixture-rewire-seed "$id_index" --out "$arms_id_out"
      bundle_dir="$(find "$arms_id_out" -mindepth 1 -maxdepth 1 -type d)"
      bundle_path="${bundle_dir}/rewired.json"
    else
      rewired_path="$(verified_intervention_graph_for_id "$graph_list" "$id")"
      expected_gzip_sha256="$(sha256sum "$rewired_path" | cut -d' ' -f1)"
      graph_sha256="$(sha256sum "$graph_path" | cut -d' ' -f1)"
      bundle_dir="${arms_id_out}/${graph_sha256}"
      bundle_path="${bundle_dir}/rewired.json"

      if [[ -f "$bundle_path" ]] && verify_arm_bundle_matches_graph "$bundle_path" "$expected_gzip_sha256"; then
        echo "train-sample.sh: id ${id}: reusing existing verified arm bundle ${bundle_path}"
      else
        echo "train-sample.sh: id ${id}: exporting arms from ${rewired_path}"
        npm run training:export-arms -- --graph "$graph_path" --rewired "$rewired_path" --out "$arms_id_out"
        # Closes a TOCTOU gap (a review finding): lookup-intervention-graph.ts
        # and export-arms each independently read $rewired_path from disk --
        # re-verify what export-arms actually read, right here, before any
        # GPU time is spent on it, rather than trusting it only implicitly.
        if [[ ! -f "$bundle_path" ]] || ! verify_arm_bundle_matches_graph "$bundle_path" "$expected_gzip_sha256"; then
          echo "train-sample.sh: id ${id}: exported ${bundle_path} does not match --graph-list's gzipSha256 for id ${id} (did the graph change between verification and export?)" >&2
          exit 1
        fi
      fi
    fi

    if [[ ! -f "$bundle_path" ]]; then
      echo "train-sample.sh: id ${id}: expected export-arms bundle not found at ${bundle_path}" >&2
      exit 1
    fi

    echo "train-sample.sh: id ${id}: training (replica-seed=${replica_seed} population=${population} elites=${elites} generations=${generations})"
    "${repo_root}/training/scripts/run.sh" flyarena-train \
      --arm rewired \
      --graph "$(to_abs_path "$bundle_path")" \
      --replica-seed "$replica_seed" \
      --substeps "$substeps" \
      --hidden-size "$hidden_size" \
      --ticks "$ticks" \
      --population "$population" \
      --elites "$elites" \
      --generations "$generations" \
      --train-seeds-per-generation "$train_seeds_per_generation" \
      --alpha "$alpha" \
      --std-floor "$std_floor" \
      --init-std "$init_std" \
      --arena-task "$arena_task" \
      --out "$(to_abs_path "$id_out")"

    echo "train-sample.sh: id ${id}: done"
    id_index=$((id_index + 1))
  done

  echo "train-sample.sh: complete (ids ${ids}, replica-seed ${replica_seed})"
  exit 0
fi

if [[ "$dry_run_fixture" -eq 0 && ! -f "${graphs_dir}/index.json" ]]; then
  echo "train-sample.sh: ${graphs_dir}/index.json not found (pass --dry-run-fixture for a fixture smoke test, or --graphs-dir)" >&2
  exit 1
fi

for (( seed = seed_start; seed < seed_start + seed_count; seed += 1 )); do
  seed_out="${trained_out}/seed${seed}"
  config_path="${seed_out}/config.json"

  if [[ -f "$config_path" ]]; then
    echo "train-sample.sh: seed ${seed}: ${config_path} already exists -- skipping (resumable)"
    continue
  fi

  arms_seed_out="${arms_out}/seed${seed}"

  if [[ "$dry_run_fixture" -eq 1 ]]; then
    echo "train-sample.sh: seed ${seed}: exporting fixture-rewired arms"
    npm run training:export-arms -- --fixture-rewire --fixture-rewire-seed "$seed" --out "$arms_seed_out"
    # trace-graph-fixture's graphArtifactSha256 is a content hash of its own
    # canonical arrays (computeGraphIdentity in export-arms.ts), not a file
    # hash -- there is no --graph file to sha256sum in fixture mode, so the
    # single subdirectory export-arms.ts just wrote is located directly
    # instead of recomputing that hash here.
    bundle_dir="$(find "$arms_seed_out" -mindepth 1 -maxdepth 1 -type d)"
  else
    rewired_artifact="$(rewired_artifact_for_seed "${graphs_dir}/index.json" "$seed")"
    rewired_path="${graphs_dir}/${rewired_artifact}"
    if [[ ! -f "$rewired_path" ]]; then
      echo "train-sample.sh: seed ${seed}: rewired graph not found at ${rewired_path}" >&2
      exit 1
    fi
    echo "train-sample.sh: seed ${seed}: exporting arms from ${rewired_path}"
    npm run training:export-arms -- --graph "$graph_path" --rewired "$rewired_path" --out "$arms_seed_out"
    graph_sha256="$(sha256sum "$graph_path" | cut -d' ' -f1)"
    bundle_dir="${arms_seed_out}/${graph_sha256}"
  fi

  bundle_path="${bundle_dir}/rewired.json"
  if [[ ! -f "$bundle_path" ]]; then
    echo "train-sample.sh: seed ${seed}: expected export-arms bundle not found at ${bundle_path}" >&2
    exit 1
  fi

  echo "train-sample.sh: seed ${seed}: training (population=${population} elites=${elites} generations=${generations})"
  "${repo_root}/training/scripts/run.sh" flyarena-train \
    --arm rewired \
    --graph "$(to_abs_path "$bundle_path")" \
    --replica-seed "$replica_seed" \
    --substeps "$substeps" \
    --hidden-size "$hidden_size" \
    --ticks "$ticks" \
    --population "$population" \
    --elites "$elites" \
    --generations "$generations" \
    --train-seeds-per-generation "$train_seeds_per_generation" \
    --alpha "$alpha" \
    --std-floor "$std_floor" \
    --init-std "$init_std" \
    --arena-task "$arena_task" \
    --out "$(to_abs_path "$seed_out")"

  echo "train-sample.sh: seed ${seed}: done"
done

echo "train-sample.sh: complete (seeds ${seed_start}..$((seed_start + seed_count - 1)))"
