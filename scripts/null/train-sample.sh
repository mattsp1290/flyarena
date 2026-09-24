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
  echo "Usage: $0 [--dry-run-fixture] [--seed-start N] [--seed-count N] [--replica-seed N]" >&2
  echo "          [--population N] [--elites N] [--generations N] [--train-seeds-per-generation N]" >&2
  echo "          [--alpha F] [--std-floor F] [--init-std F] [--hidden-size N] [--ticks N]" >&2
  echo "          [--graph PATH] [--graphs-dir DIR] [--arms-out DIR] [--trained-out DIR]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run-fixture) dry_run_fixture=1; shift ;;
    --seed-start) seed_start="${2:?}"; shift 2 ;;
    --seed-count) seed_count="${2:?}"; shift 2 ;;
    --replica-seed) replica_seed="${2:?}"; shift 2 ;;
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
    --arms-out) arms_out="${2:?}"; shift 2 ;;
    --trained-out) trained_out="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "train-sample.sh: unknown argument: $1" >&2; usage ;;
  esac
done

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

mkdir -p "$trained_out"

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
    --out "$(to_abs_path "$seed_out")"

  echo "train-sample.sh: seed ${seed}: done"
done

echo "train-sample.sh: complete (seeds ${seed_start}..$((seed_start + seed_count - 1)))"
