#!/usr/bin/env bash
# `.agents/plans/selection-robustness/02-per-selection-chain.md` WP2: runs
# steps 1-5 (rewirings, authored null, mirrored-decoder null, explanation,
# coverage, interventions) for one predeclared selection. Resumable: every
# step's own output file is checked first, and the step is skipped when it
# already exists -- a killed/interrupted run can simply be re-invoked with
# the same selection id. Between steps, this script also cross-checks the
# `sourceGraphSha256`/`sourceSha256` each producer stamps against the
# selection's own compiled graph -- an early, cheap failure here beats
# discovering a mismatch deep into a multi-hour downstream step (the
# producers themselves also enforce this -- `null-report.ts`'s
# `verifySourceGraphMatchesManifest`, `interventions.py`'s regenerated-seed
# sha cross-check, `intervention-report-validation.ts`'s
# `assertConsistentInputs` -- this is a belt-and-suspenders early exit, not
# the only check).
#
# Hard safety requirement (this bean's coordinator instructions): none of
# this chain's steps may write under public/ or docs/. Every producer this
# script calls (`null-report.ts`, `explain.py`, `intervention-report.ts`)
# carries its own `sourceGraphSha256`-keyed public/docs refusal guard, and
# this script never points any `--out`/`--report-md`/`--report-out`/
# `--manifest` flag anywhere but under `$B` (this selection's own scratch
# tree) or, for `--manifest`, the selection's own compiled manifest (never
# `public/data/malecns-arena-v1.manifest.json`). A final `git status`
# check (below) verifies public/docs are byte-unchanged after the chain
# completes.
#
# Usage: scripts/selections/run-chain.sh <selection-id> [--shards N] [--workers N] [--dry-run-fixture]
#   <selection-id>    one of scripts/data/selections.py's SELECTIONS keys
#                     (default/larger/smaller/random-bridge/alt-sensory-mapping)
#   --shards N        null-evaluate.ts/regime-check.ts shard count (default 12,
#                      this bean's machine-sharing cap)
#   --workers N       transfer.py/features.py process-pool size (default 12)
#   --dry-run-fixture structural self-check only: validates the selection id,
#                      that every CLI this chain invokes (uv/npm/jq) and every
#                      script path it references exist, and that every npm
#                      script name it calls is declared in package.json --
#                      then prints every step's command without running any
#                      of the real (hours-long) compute. It does NOT run any
#                      producer against real or fixture data (a full
#                      fixture-scale end-to-end run through all eight
#                      downstream tools was out of scope for WP2 -- see
#                      .agents/plans/selection-robustness/02-per-selection-chain.md's
#                      own change-surface note), so it cannot catch a flag
#                      whose *meaning* changed without also changing its name.

set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <selection-id> [--shards N] [--workers N] [--dry-run-fixture]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
SELECTION="$1"
shift

# Validated unconditionally (not only in --dry-run-fixture): matches
# scripts/data/selections.py's SELECTIONS keys exactly -- a dual-review
# finding: an earlier version accepted any string here and only discovered
# a typo'd/unknown id hours later, when the first real step failed (or, in
# dry-run mode, never discovered it at all).
case "$SELECTION" in
  default | larger | smaller | random-bridge | alt-sensory-mapping) ;;
  *)
    echo "run-chain.sh: unknown selection id '${SELECTION}' (expected one of: default, larger, smaller," >&2
    echo "  random-bridge, alt-sensory-mapping -- scripts/data/selections.py's SELECTIONS keys)" >&2
    exit 1
    ;;
esac

SHARDS=12
WORKERS=12
DRY_RUN=0
HELD_OUT_START=30001
HELD_OUT_COUNT=100
TICKS=1800

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shards)
      SHARDS="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --dry-run-fixture)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "run-chain.sh: unknown argument $1" >&2
      usage
      ;;
  esac
done

if (( SHARDS > 12 )); then
  echo "run-chain.sh: --shards ${SHARDS} exceeds this bean's 12-shard machine-sharing cap" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The analysis steps (transfer.py, features.py, explain.py) refuse to run
# unless BLAS is single-threaded (scripts/analysis/env_guard.py). Set that
# here rather than relying on the caller's shell: the first real run died at
# transfer.py because the launching shell lacked MKL_NUM_THREADS.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DD_IAST_ENABLED=false
unset PYTHONPATH

command -v jq >/dev/null 2>&1 || { echo "run-chain.sh: jq is required (sha cross-checks between steps)" >&2; exit 1; }

B="training/runs/selections/${SELECTION}"
GRAPH="$B/malecns-arena-${SELECTION}.bin.gz"
MANIFEST="$B/malecns-arena-${SELECTION}.manifest.json"

log() { echo "[run-chain:${SELECTION}] $(date -u +%FT%TZ) $*"; }

run() {
  log "+ $*"
  if [[ "$DRY_RUN" == 1 ]]; then
    return 0
  fi
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "run-chain.sh: expected file not found: $1" >&2
    exit 1
  fi
}

check_sha() {
  # check_sha <label> <file> <jq-filter> <expected-sha>
  local label="$1" file="$2" filter="$3" expected="$4" actual
  if [[ "$DRY_RUN" == 1 ]]; then
    log "  (dry-run: would check ${label} sha in ${file})"
    return 0
  fi
  actual="$(jq -r "$filter" "$file")"
  if [[ "$actual" != "$expected" ]]; then
    echo "run-chain.sh: ${label} sha mismatch in ${file}: got ${actual}, expected ${expected}" >&2
    exit 1
  fi
}

# Step 3d's skip check (below) must not treat "both output files exist" as
# "step 3d fully completed" -- explain.py's write order is args.out (982) ->
# args.report_out (985) -> update_manifest(args.manifest, ...) (987), i.e.
# the manifest's `nullExplanation` key is written LAST. A crash between the
# report write and the manifest write leaves both `null-explanation.json`
# and `null-explanation-report.md` on disk while `$MANIFEST`'s own
# `nullExplanation` entry is missing or stale -- a re-invocation must not
# silently treat that as done (a thermo-methodology review finding: this
# is the exact "partial output treated as done" case resumability must
# never allow, and `intervention-artifact.ts` later hard-requires
# `manifest.nullExplanation.sha256` to be present). `explain.py`'s
# `update_manifest` records `{"artifact": args.out.name, "sha256":
# sha256_hex(canonical_json_text(explanation))}` -- the artifact's sha256
# is over the exact bytes `atomic_write_text` puts on disk, so comparing
# it against `sha256sum "$B/null-explanation.json"` (already used the same
# way by scripts/null/train-sample.sh) proves the manifest entry actually
# describes the file currently on disk, not a stale one from a prior run.
null_explanation_manifest_matches() {
  local recorded_artifact recorded_sha256 actual_sha256
  recorded_artifact="$(jq -r '.nullExplanation.artifact // empty' "$MANIFEST")"
  recorded_sha256="$(jq -r '.nullExplanation.sha256 // empty' "$MANIFEST")"
  [[ -n "$recorded_artifact" && -n "$recorded_sha256" ]] || return 1
  [[ "$recorded_artifact" == "$(basename "$B/null-explanation.json")" ]] || return 1
  actual_sha256="$(sha256sum "$B/null-explanation.json" | cut -d' ' -f1)"
  [[ "$recorded_sha256" == "$actual_sha256" ]]
}

if [[ "$DRY_RUN" == 1 ]]; then
  log "dry-run-fixture: structural self-check only, no real compute will run"
  GRAPH_SHA="<dry-run: not read>"

  for cmd in uv npm jq; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "run-chain.sh: dry-run-fixture: '$cmd' not on PATH" >&2; exit 1; }
  done
  for script in scripts/data/rewire_batch.py scripts/analysis/transfer.py scripts/analysis/features.py \
    scripts/analysis/explain.py scripts/analysis/interventions.py scripts/selections/coverage.py \
    scripts/null/null-evaluate.ts scripts/null/null-report.ts scripts/null/regime-check.ts \
    scripts/null/intervention-report.ts; do
    require_file "$script"
  done
  for npm_script in null:evaluate null:report null:regime-check intervention:report; do
    jq -e --arg s "$npm_script" '.scripts[$s] != null' package.json >/dev/null \
      || { echo "run-chain.sh: dry-run-fixture: package.json has no \"$npm_script\" script" >&2; exit 1; }
  done
else
  require_file "$GRAPH"
  require_file "$MANIFEST"
  GRAPH_SHA="$(jq -r '.binarySha256' "$MANIFEST")"
fi

# --- Step 1: rewirings ---
if [[ "$DRY_RUN" != 1 && -f "$B/graphs/index.json" ]]; then
  log "step 1 (rewirings): $B/graphs/index.json exists, skipping"
else
  run uv run python scripts/data/rewire_batch.py --in-path "$GRAPH" --seeds 0:500 --out-dir "$B/graphs"
fi
check_sha "graphs/index.json sourceSha256" "$B/graphs/index.json" '.sourceSha256' "$GRAPH_SHA"

# --- Step 2: authored null ---
# Skip only when EVERY output null-report.ts writes exists -- not just the
# first (rewiring-null.json). null-report.ts's write order is out -> manifest
# rewiringNull key -> report md, so a crash between writes would otherwise
# leave rewiring-null-report.md silently missing (a dual-review finding).
if [[ "$DRY_RUN" != 1 && -f "$B/rewiring-null.json" && -f "$B/rewiring-null-report.md" ]]; then
  log "step 2 (authored null): $B/rewiring-null.json and .../rewiring-null-report.md exist, skipping"
else
  if [[ "$DRY_RUN" == 1 || ! -f "$B/null-raw.json" ]]; then
    run npm run null:evaluate -- --biological --graph "$GRAPH" --rewired-index "$B/graphs/index.json" \
      --graphs-dir "$B/graphs" --held-out-start "$HELD_OUT_START" --held-out-count "$HELD_OUT_COUNT" \
      --ticks "$TICKS" --shards "$SHARDS" --out "$B/null-raw.json"
  fi
  # --trained points at a path that is never created for a selection run: a
  # dual-review finding -- null-report.ts's ordinary path merges a `trained`
  # section whenever ITS DEFAULT --trained path (training/runs/null/trained.json,
  # the shipped rewiring-null study's own trained-readout output) happens to
  # exist on this machine, with no graph-sha check (NullTrainedEvaluationRaw
  # carries none). Left at its default, a selection's rewiring-null.json
  # would silently carry the *shipped* graph's trained-readout percentile.
  run npm run null:report -- --authored "$B/null-raw.json" --out "$B/rewiring-null.json" \
    --report-md "$B/rewiring-null-report.md" --manifest "$MANIFEST" --trained "$B/no-trained-arm.json"
fi
check_sha "rewiring-null.json sourceGraphSha256" "$B/rewiring-null.json" '.sourceGraphSha256' "$GRAPH_SHA"

# --- Step 2a: mirrored-decoder null (required by explain.py --selection-mode) ---
if [[ "$DRY_RUN" != 1 && -f "$B/variant-flip-both.json" ]]; then
  log "step 2a (mirrored null): $B/variant-flip-both.json exists, skipping"
else
  if [[ "$DRY_RUN" == 1 || ! -f "$B/null-flip-both-raw.json" ]]; then
    run npm run null:evaluate -- --biological --graph "$GRAPH" --rewired-index "$B/graphs/index.json" \
      --graphs-dir "$B/graphs" --held-out-start "$HELD_OUT_START" --held-out-count "$HELD_OUT_COUNT" \
      --ticks "$TICKS" --shards "$SHARDS" --decoder authored-flip-both --out "$B/null-flip-both-raw.json"
  fi
  run npm run null:report -- --authored "$B/null-flip-both-raw.json" --variant-out "$B/variant-flip-both.json"
fi
check_sha "variant-flip-both.json sourceGraphSha256" "$B/variant-flip-both.json" '.sourceGraphSha256' "$GRAPH_SHA"

# --- Step 2b: single-axis decoder-variant nulls (conditional follow-up) ---
# The predeclared rule (.agents/plans/null-explanation/00-overview.md:35):
# only if the mirrored run (step 2a) moves biological to at least the 25th
# percentile do the single-axis variants (authored-flip-thrust,
# authored-flip-yaw) run too, to find which axis is responsible. explain.py
# enforces this same rule at read time (scripts/analysis/explain.py:757,
# comparing variant_flip_both["bioPercentile"] against its own
# DECODER_PERCENTILE_THRESHOLD constant, defined at explain.py:109) and
# refuses to run --selection-mode if the single-axis files are supplied
# without having been triggered, or triggered without both being supplied.
# Rather than hardcoding 0.25 here too and risking the two drifting apart,
# this script parses the threshold straight out of explain.py's own source.
if [[ "$DRY_RUN" == 1 ]]; then
  log "step 2b (single-axis nulls): dry-run -- printing both authored-flip-thrust/authored-flip-yaw commands structurally (the real chain gates them on variant-flip-both.json's bioPercentile vs. explain.py's DECODER_PERCENTILE_THRESHOLD)"
  SINGLE_AXIS_TRIGGERED=1
else
  DECODER_PERCENTILE_THRESHOLD="$(grep -m1 '^DECODER_PERCENTILE_THRESHOLD = ' scripts/analysis/explain.py | sed -E 's/^DECODER_PERCENTILE_THRESHOLD = ([0-9.]+).*/\1/')"
  if [[ -z "$DECODER_PERCENTILE_THRESHOLD" ]]; then
    echo "run-chain.sh: could not parse DECODER_PERCENTILE_THRESHOLD out of scripts/analysis/explain.py" >&2
    exit 1
  fi
  BIO_PERCENTILE="$(jq -r '.bioPercentile // empty' "$B/variant-flip-both.json")"
  # A missing/non-numeric field must fail loudly, the same way the
  # threshold parse just above does -- not silently fall through to
  # "not triggered" (jq -r on a missing key prints the string "null",
  # which --argjson happily accepts as JSON null, and `null >= 0.25` is
  # just `false`, no error). explain.py itself fails loudly on this same
  # field via an unchecked dict index (explain.py:754), so this guard
  # keeps run-chain.sh's own read no less strict.
  if ! [[ "$BIO_PERCENTILE" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
    echo "run-chain.sh: variant-flip-both.json has no numeric bioPercentile field: $B/variant-flip-both.json" >&2
    exit 1
  fi
  SINGLE_AXIS_TRIGGERED=0
  if jq -n --argjson bio "$BIO_PERCENTILE" --argjson threshold "$DECODER_PERCENTILE_THRESHOLD" -e \
    '$bio >= $threshold' >/dev/null; then
    SINGLE_AXIS_TRIGGERED=1
  fi
fi

if [[ "$SINGLE_AXIS_TRIGGERED" == 1 ]]; then
  if [[ "$DRY_RUN" != 1 ]]; then
    log "step 2b (single-axis nulls): mirrored bioPercentile ${BIO_PERCENTILE} meets the predeclared >= ${DECODER_PERCENTILE_THRESHOLD} threshold -- running authored-flip-thrust and authored-flip-yaw"
  fi

  if [[ "$DRY_RUN" != 1 && -f "$B/variant-flip-thrust.json" ]]; then
    log "step 2b (flip-thrust null): $B/variant-flip-thrust.json exists, skipping"
  else
    if [[ "$DRY_RUN" == 1 || ! -f "$B/null-flip-thrust-raw.json" ]]; then
      run npm run null:evaluate -- --biological --graph "$GRAPH" --rewired-index "$B/graphs/index.json" \
        --graphs-dir "$B/graphs" --held-out-start "$HELD_OUT_START" --held-out-count "$HELD_OUT_COUNT" \
        --ticks "$TICKS" --shards "$SHARDS" --decoder authored-flip-thrust --out "$B/null-flip-thrust-raw.json"
    fi
    run npm run null:report -- --authored "$B/null-flip-thrust-raw.json" --variant-out "$B/variant-flip-thrust.json"
  fi
  check_sha "variant-flip-thrust.json sourceGraphSha256" "$B/variant-flip-thrust.json" '.sourceGraphSha256' "$GRAPH_SHA"

  if [[ "$DRY_RUN" != 1 && -f "$B/variant-flip-yaw.json" ]]; then
    log "step 2b (flip-yaw null): $B/variant-flip-yaw.json exists, skipping"
  else
    if [[ "$DRY_RUN" == 1 || ! -f "$B/null-flip-yaw-raw.json" ]]; then
      run npm run null:evaluate -- --biological --graph "$GRAPH" --rewired-index "$B/graphs/index.json" \
        --graphs-dir "$B/graphs" --held-out-start "$HELD_OUT_START" --held-out-count "$HELD_OUT_COUNT" \
        --ticks "$TICKS" --shards "$SHARDS" --decoder authored-flip-yaw --out "$B/null-flip-yaw-raw.json"
    fi
    run npm run null:report -- --authored "$B/null-flip-yaw-raw.json" --variant-out "$B/variant-flip-yaw.json"
  fi
  check_sha "variant-flip-yaw.json sourceGraphSha256" "$B/variant-flip-yaw.json" '.sourceGraphSha256' "$GRAPH_SHA"
else
  log "step 2b (single-axis nulls): mirrored bioPercentile ${BIO_PERCENTILE} is below the predeclared >= ${DECODER_PERCENTILE_THRESHOLD} threshold -- skipping authored-flip-thrust/authored-flip-yaw per .agents/plans/null-explanation/00-overview.md:35"
fi

# --- Step 3: explanation (transfer, features, regime, explain) ---
if [[ "$DRY_RUN" != 1 && -f "$B/transfer.json" ]]; then
  log "step 3a (transfer): exists, skipping"
else
  run uv run python scripts/analysis/transfer.py --index "$B/graphs/index.json" --graphs-dir "$B/graphs" \
    --biological "$GRAPH" --out "$B/transfer.json" --workers "$WORKERS"
fi

if [[ "$DRY_RUN" != 1 && -f "$B/features.json" ]]; then
  log "step 3b (features): exists, skipping"
else
  run uv run python scripts/analysis/features.py --index "$B/graphs/index.json" --graphs-dir "$B/graphs" \
    --biological "$GRAPH" --out "$B/features.json" --workers "$WORKERS"
fi

if [[ "$DRY_RUN" != 1 && -f "$B/regime.json" ]]; then
  log "step 3c (regime): exists, skipping"
else
  run npm run null:regime-check -- --biological --graph "$GRAPH" --rewired-index "$B/graphs/index.json" \
    --graphs-dir "$B/graphs" --steady-state-dir "$B/steady-state" --shards "$SHARDS" --out "$B/regime.json"
fi

# Skip only when both of explain.py's write-order outputs exist (out ->
# report -> manifest, explain.py:982-986) -- a crash between writes would
# otherwise leave null-explanation-report.md (an acceptance artifact)
# silently missing (a dual-review finding, same class as step 2's above)
# -- AND the manifest's own `nullExplanation` entry (written LAST, see
# `null_explanation_manifest_matches`'s doc comment above) is present and
# still matches the file actually on disk.
if [[ "$DRY_RUN" != 1 && -f "$B/null-explanation.json" && -f "$B/null-explanation-report.md" ]] \
  && null_explanation_manifest_matches; then
  log "step 3d (explain --selection-mode): both outputs exist and manifest nullExplanation entry matches, skipping"
else
  # Every path flag given explicitly (never a default), per the plan --
  # the public/docs refusal in explain.py applies to --out/--report-out/
  # --manifest alike, and --selection-mode makes
  # --features-exploratory-unrestricted optional (omitted here: this
  # study's historical snapshot has no per-selection counterpart).
  #
  # --variant-flip-thrust/--variant-flip-yaw are passed exactly when step
  # 2b's threshold check triggered them: explain.py itself requires
  # both-or-neither, keyed off the same bioPercentile it reads from
  # variant-flip-both.json (explain.py:757-777) -- passing them
  # unconditionally (or omitting them) here would just make explain.py
  # throw, so SINGLE_AXIS_TRIGGERED (set in step 2b, still in scope) is the
  # single source of truth for whether they exist on disk at all.
  EXPLAIN_SINGLE_AXIS_ARGS=()
  if [[ "$SINGLE_AXIS_TRIGGERED" == 1 ]]; then
    EXPLAIN_SINGLE_AXIS_ARGS+=(--variant-flip-thrust "$B/variant-flip-thrust.json" --variant-flip-yaw "$B/variant-flip-yaw.json")
  fi
  run uv run python scripts/analysis/explain.py --selection-mode \
    --rewiring-null "$B/rewiring-null.json" --variant-flip-both "$B/variant-flip-both.json" \
    "${EXPLAIN_SINGLE_AXIS_ARGS[@]}" \
    --transfer "$B/transfer.json" --features "$B/features.json" --regime "$B/regime.json" \
    --out "$B/null-explanation.json" --report-out "$B/null-explanation-report.md" --manifest "$MANIFEST"
fi

# --- Step 4: coverage ---
if [[ "$DRY_RUN" != 1 && -f "$B/coverage.json" ]]; then
  log "step 4 (coverage): exists, skipping"
else
  run uv run python scripts/selections/coverage.py --graph "$GRAPH" --manifest "$MANIFEST" --out "$B/coverage.json"
fi
check_sha "coverage.json sourceGraphSha256" "$B/coverage.json" '.sourceGraphSha256' "$GRAPH_SHA"

# --- Step 5: interventions (rebuild P/C/M from this selection's own null/explanation, then score + stats) ---
# Skip only when both of interventions.py's outputs exist (index.json ->
# attribution.json, interventions.py:879,901) -- attribution.json holds P's
# search-budget disclosure (k, targetReached) that WP3 needs; a crash
# between writes would otherwise leave it silently missing.
if [[ "$DRY_RUN" != 1 && -f "$B/interventions/index.json" && -f "$B/interventions/attribution.json" ]]; then
  log "step 5a (interventions): both outputs exist, skipping"
else
  run uv run python scripts/analysis/interventions.py --biological "$GRAPH" --null "$B/rewiring-null.json" \
    --explanation "$B/null-explanation.json" --out-dir "$B/interventions"
fi
check_sha "interventions/index.json sourceSha256" "$B/interventions/index.json" '.sourceSha256' "$GRAPH_SHA"

if [[ "$DRY_RUN" != 1 && -f "$B/interventions/authored.json" ]]; then
  log "step 5b (intervention scoring): exists, skipping"
else
  run npm run null:evaluate -- --biological --graph "$GRAPH" --graph-list "$B/interventions/index.json" \
    --held-out-start "$HELD_OUT_START" --held-out-count "$HELD_OUT_COUNT" --ticks "$TICKS" \
    --shards "$SHARDS" --out "$B/interventions/authored.json"
fi

if [[ "$DRY_RUN" != 1 && -f "$B/intervention-stats.json" ]]; then
  log "step 5c (intervention-report --stats-only): exists, skipping"
else
  run npm run intervention:report -- --authored "$B/interventions/authored.json" \
    --index "$B/interventions/index.json" --null "$B/rewiring-null.json" \
    --stats-only --arena-task default --out "$B/intervention-stats.json"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  log "dry-run-fixture complete for selection=${SELECTION} (no compute ran)"
  exit 0
fi

# --- Final safety check: public/ and docs/ must be byte-unchanged ---
if [[ -d "$REPO_ROOT/.git" ]] || git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  DIRTY="$(git -C "$REPO_ROOT" status --porcelain -- public docs)"
  if [[ -n "$DIRTY" ]]; then
    echo "run-chain.sh: public/ or docs/ changed during this chain -- refusing to report success:" >&2
    echo "$DIRTY" >&2
    exit 1
  fi
fi

log "chain complete for selection=${SELECTION}: $B/{rewiring-null,variant-flip-both,null-explanation,coverage,intervention-stats}.json"
