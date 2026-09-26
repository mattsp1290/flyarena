#!/usr/bin/env bash
# WP2's GPU search driver (.agents/plans/repertoire-null/02-runs-and-reevaluation.md).
#
# For every entry in `repertoire-plan.ts`'s planned 46 (graph, seed) pairs
# (the same enumeration `repertoire-evaluate.ts` re-derives for TS
# re-evaluation, so the two never drift out of agreement about which pairs
# exist): verify the arm bundle's identity (the same rule
# `verify-search-graph.ts` applies to a search JSON's embedded bundle,
# applied here directly to the bundle file before any search exists --
# `repertoire-plan.ts --mode verify-bundle`), then run the shipped
# MAP-Elites search on it with the *shipped* population/generations/ticks
# budget (read from the shipped atlas artifact itself -- never hardcoded,
# never shrunk). Runs sequentially (one GPU, one search at a time) and is
# resumable: an existing output file is skipped, matching
# `flyarena_training.atlas_cli`'s own refusal to overwrite an existing
# `--output`.
#
# Usage:
#   scripts/atlas/repertoire-search.sh                  # run every planned search
#   scripts/atlas/repertoire-search.sh 'rewired-0@1729'  # run just one entry (calibration)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.nvm/versions/node/v22.22.3/bin:$PATH"
# Keep BLAS/OpenMP threading out of torch's own thread pool's way -- the
# host's pytorch controls its own CUDA/CPU threading via
# `torch.set_num_threads(2)` (atlas_cli.py); a BLAS library spawning its own
# thread pool on top of that on a many-core Spark host would oversubscribe
# CPU for no benefit to a GPU-bound search.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

DATA_DIR="${DATA_DIR:-public/data}"
GRAPHS_INDEX="${GRAPHS_INDEX:-training/runs/repertoire/graphs/index.json}"
ARMS_DIR="${ARMS_DIR:-training/runs/repertoire/arms}"
SEARCH_DIR="${SEARCH_DIR:-training/runs/repertoire/search}"

mkdir -p "$SEARCH_DIR"

ATLAS_ARTIFACT="$DATA_DIR/behavior-atlas-v1.json"
if [ ! -f "$ATLAS_ARTIFACT" ]; then
  echo "repertoire-search: shipped atlas artifact not found at $ATLAS_ARTIFACT" >&2
  exit 1
fi
# Never hardcode/shrink the search budget: read it off the shipped atlas
# artifact's own recorded search options every run. `-e` fails loudly (jq
# exits non-zero) on a missing/null field instead of handing atlas_cli.py a
# literal "null" population/generations/ticks argument.
POPULATION=$(jq -er '.source.options.population' "$ATLAS_ARTIFACT")
GENERATIONS=$(jq -er '.source.options.generations' "$ATLAS_ARTIFACT")
TICKS=$(jq -er '.source.options.ticks' "$ATLAS_ARTIFACT")
echo "repertoire-search: shipped budget population=$POPULATION generations=$GENERATIONS ticks=$TICKS (from $ATLAS_ARTIFACT)"

PLAN_JSON=$(npx tsx scripts/atlas/repertoire-plan.ts --mode emit \
  --data "$DATA_DIR" --graphs-index "$GRAPHS_INDEX" --arms-dir "$ARMS_DIR" --search-dir "$SEARCH_DIR")
PLAN_COUNT=$(echo "$PLAN_JSON" | jq 'length')
echo "repertoire-search: $PLAN_COUNT planned (graph, seed) pairs"

ONLY_KEY="${1:-}"

RAN=0
SKIPPED=0
MATCHED=0
while IFS= read -r ENTRY; do
  GRAPH_ID=$(echo "$ENTRY" | jq -r '.graphId')
  SEED=$(echo "$ENTRY" | jq -r '.searchSeed')
  BUNDLE=$(echo "$ENTRY" | jq -r '.bundlePath')
  ARM=$(echo "$ENTRY" | jq -r '.arm')
  BINARY_SHA=$(echo "$ENTRY" | jq -r '.expected.binarySha256')
  PARENT_SHA=$(echo "$ENTRY" | jq -r '.expected.parentGzipSha256')
  OUTPUT=$(echo "$ENTRY" | jq -r '.searchOutputPath')
  KEY="${GRAPH_ID}@${SEED}"

  if [ -n "$ONLY_KEY" ] && [ "$KEY" != "$ONLY_KEY" ]; then
    continue
  fi
  MATCHED=$((MATCHED + 1))

  if [ -f "$OUTPUT" ]; then
    echo "repertoire-search: [skip] $KEY (output already exists at $OUTPUT)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "repertoire-search: [verify] $KEY bundle=$BUNDLE"
  npx tsx scripts/atlas/repertoire-plan.ts --mode verify-bundle \
    --bundle "$BUNDLE" --arm "$ARM" --binary-sha256 "$BINARY_SHA" --parent-gzip-sha256 "$PARENT_SHA"

  echo "repertoire-search: [search] $KEY -> $OUTPUT (seed=$SEED)"
  SEARCH_STARTED=$(date +%s)
  training/scripts/run.sh python -m flyarena_training.atlas_cli \
    --graph "$BUNDLE" \
    --output "$OUTPUT" \
    --device cuda \
    --seed "$SEED" \
    --population "$POPULATION" \
    --generations "$GENERATIONS" \
    --ticks "$TICKS"
  SEARCH_ELAPSED=$(($(date +%s) - SEARCH_STARTED))
  echo "repertoire-search: [done] $KEY in ${SEARCH_ELAPSED}s"
  RAN=$((RAN + 1))
done < <(echo "$PLAN_JSON" | jq -c '.[]')

# A mistyped calibration key (e.g. a transposed digit) would otherwise match
# nothing, run zero searches, and still exit 0 -- indistinguishable from a
# genuinely already-complete run (a dual-review finding).
if [ -n "$ONLY_KEY" ] && [ "$MATCHED" -eq 0 ]; then
  echo "repertoire-search: no planned entry matches '$ONLY_KEY'" >&2
  echo "$PLAN_JSON" | jq -r '.[] | .graphId + "@" + (.searchSeed | tostring)' >&2
  exit 1
fi

echo "repertoire-search: ran $RAN search(es), skipped $SKIPPED already-present"
