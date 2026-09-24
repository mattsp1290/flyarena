#!/usr/bin/env bash
#
# .agents/plans/rewiring-null/03-trained-sample.md's train-sample.sh: CEM-trains
# one readout per rewired graph (rewiring seeds 0..19 by default) with the
# EXACT production CEM config flyarena-bigq's biological replicas used --
# population 128, elites 32, generations 150, alpha 0.7, stdFloor 0.02,
# initStd 0.5, E (training seeds/generation) 16, H (hidden size) 16, ticks
# 1800 -- so the null's 20 rewired scores and bigq's 3 biological replicas
# differ only in graph topology / replica seed, never in hyperparameters.
# The merged source of truth for this config is
# public/data/trained-readout-v1.manifest.json's "training" block
# (re-grounded against training/runs/production/biological-{101,202,303}/config.json
# in this worktree, which was verified byte-for-byte against that manifest
# block before this script was written).
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

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
cd "$repo_root"

dry_run_fixture=0
seed_start=0
seed_count=20
replica_seed=101
population=128
elites=32
generations=150
train_seeds_per_generation=16
alpha=0.7
std_floor=0.02
init_std=0.5
hidden_size=16
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
rewired_artifact_for_seed() {
  local index_path="$1" seed="$2"
  python3 - "$index_path" "$seed" <<'PY'
import json
import sys

index = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
for entry in index["seeds"]:
    if entry["seed"] == seed:
        print(entry["artifact"])
        sys.exit(0)
sys.stderr.write(f"train-sample.sh: seed {seed} not found in {sys.argv[1]}\n")
sys.exit(1)
PY
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
