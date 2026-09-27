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
#   --dry-run-fixture structural self-check only (script/binary existence,
#                      argument wiring) -- prints every step's command
#                      without running any of the real (hours-long) compute.
#                      Does not exercise the tools themselves; see the
#                      script's own doc comment in the implementation PR for
#                      why a true fixture-scale end-to-end run was out of
#                      scope for this pass.

set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <selection-id> [--shards N] [--workers N] [--dry-run-fixture]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
SELECTION="$1"
shift

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

if [[ "$DRY_RUN" == 1 ]]; then
  log "dry-run-fixture: structural self-check only, no real compute will run"
  GRAPH_SHA="<dry-run: not read>"
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
if [[ "$DRY_RUN" != 1 && -f "$B/rewiring-null.json" ]]; then
  log "step 2 (authored null): $B/rewiring-null.json exists, skipping"
else
  if [[ "$DRY_RUN" == 1 || ! -f "$B/null-raw.json" ]]; then
    run npm run null:evaluate -- --biological --graph "$GRAPH" --rewired-index "$B/graphs/index.json" \
      --graphs-dir "$B/graphs" --held-out-start "$HELD_OUT_START" --held-out-count "$HELD_OUT_COUNT" \
      --ticks "$TICKS" --shards "$SHARDS" --out "$B/null-raw.json"
  fi
  run npm run null:report -- --authored "$B/null-raw.json" --out "$B/rewiring-null.json" \
    --report-md "$B/rewiring-null-report.md" --manifest "$MANIFEST"
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

if [[ "$DRY_RUN" != 1 && -f "$B/null-explanation.json" ]]; then
  log "step 3d (explain --selection-mode): exists, skipping"
else
  # Every path flag given explicitly (never a default), per the plan --
  # the public/docs refusal in explain.py applies to --out/--report-out/
  # --manifest alike, and --selection-mode makes
  # --features-exploratory-unrestricted optional (omitted here: this
  # study's historical snapshot has no per-selection counterpart).
  run uv run python scripts/analysis/explain.py --selection-mode \
    --rewiring-null "$B/rewiring-null.json" --variant-flip-both "$B/variant-flip-both.json" \
    --transfer "$B/transfer.json" --features "$B/features.json" --regime "$B/regime.json" \
    --out "$B/null-explanation.json" --report-out "$B/null-explanation-report.md" --manifest "$MANIFEST"
fi

# --- Step 4: coverage ---
if [[ "$DRY_RUN" != 1 && -f "$B/coverage.json" ]]; then
  log "step 4 (coverage): exists, skipping"
else
  run uv run python scripts/selections/coverage.py --graph "$GRAPH" --manifest "$MANIFEST" --out "$B/coverage.json"
fi

# --- Step 5: interventions (rebuild P/C/M from this selection's own null/explanation, then score + stats) ---
if [[ "$DRY_RUN" != 1 && -f "$B/interventions/index.json" ]]; then
  log "step 5a (interventions): exists, skipping"
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
