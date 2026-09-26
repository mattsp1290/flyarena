# Behavior repertoire, biological vs. degree-preserving rewirings

## Question

Does the measured MaleCNS topology support a wider or narrower range of behaviors than degree-preserving
rewirings of the same graph, under this model's shipped MAP-Elites behavior-discovery search? This is a
comparison of behavioral *repertoire* (how many distinct behavior cells are reached, and how good the best
controller in each is), not of a single behavior score.

## Method

The shipped behavior-atlas MAP-Elites search (population 64, generations 24,
ticks 900, discovery seeds [61001,61002,61003,61004,61005,61006,61007,61008], held-out seeds
[62001,62002,62003,62004,62005,62006,62007,62008,62009,62010,62011,62012] -- the same search budget and descriptors as the shipped
`behavior-atlas-v1.json`, never a tuned or shrunk one) was run once per graph at search seed
1729 on: the biological graph, the disconnected control, and 20
degree-preserving rewirings (seeds `0..19`). Biological and rewirings
`0..4` were additionally searched at seeds
1730, 1731, 1732, 1733 for search-seed robustness -- **rewired seeds
5..19 were searched at a single search seed
(1729) only**, never at the extra seeds. Every search was re-evaluated in TypeScript
(authoritative), re-binned into the 36-cell coverage x turning grid, with the diversity gate disabled
for null graphs (a narrow repertoire is a valid result, not an error). The discovery seeds and held-out seeds
above are pinned globally, shared by every graph and every search seed; only population/generations/ticks are
checked per entry against the shipped atlas budget (a maintainability review, Suggestion: this doc comment
previously left that unstated, which could be misread as a per-entry seed check).

### Predeclared metrics

- `occupied`: the number of occupied cells among 36, after TS rebinning and collision resolution.
- `qd`: the sum over occupied cells of `max(0, quality)`.
- `span`: the number of distinct coverage bins plus the number of distinct turning bins that are occupied.
- `heldoutOwnMedian`: the median over occupied cells of the searched graph's own held-out mean score
  (seeds [62001,62002,62003,62004,62005,62006,62007,62008,62009,62010,62011,62012], trained decoder on that graph). Reported, not categorized.
- Audit fields per graph: `gpuArchiveSize`, `occupied` after TS rebinning, and `collisions`.

### Predeclared categories

- **Wider repertoire:** biological is at or above the rewired 75th percentile on both `occupied` and `qd`.
- **Narrower repertoire:** biological is at or below the 25th percentile on both.
- **Typical:** otherwise.
- Tie rule (predeclared): **Wider** requires, on both metrics, `bio ≥ p75` **and** `bio > p25`. **Narrower** requires, on both metrics, `bio ≤ p25` **and** `bio < p75`. With a degenerate rewired distribution (`p25 = p75 = bio`) neither holds, and the result is **Typical** with `tie: true`. Wider and Narrower can never both hold.

Search-seed robustness: the category is **robust** only if it is the same for all 5
biological search seeds against the seed-matched rewired distribution. Rewirings
`0..4` are the seed-matched sample for seeds
1730, 1731, 1732, 1733 -- **that sample has only 5 points (a 20% percentile
resolution)**, far coarser than the 20-point primary comparison at seed 1729.

## Results

### Primary comparison (search seed 1729)

| metric | biological | rewired distribution (n=20) |
| --- | --- | --- |
| occupied | 28 | rewired: n=20, p25=28.00, p50=29.00, p75=29.00 |
| qd | 637.49 | rewired: n=20, p25=728.15, p50=763.00, p75=828.44 |
| span | 11 | -- (not categorized) |
| heldoutOwnMedian | 16.55 | -- (not categorized) |

**Category at seed 1729: narrower**.

### Search-seed robustness

| search seed | seed-matched rewired n | occupied p25/p50/p75 | qd p25/p50/p75 | category |
| --- | --- | --- | --- | --- |
| 1729 | 20 | 28/29/29 | 728.15/763.00/828.44 | narrower |
| 1730 | 5 | 29/29/30 | 736.26/827.89/852.72 | narrower |
| 1731 | 5 | 29/30/31 | 823.41/846.38/857.41 | typical |
| 1732 | 5 | 28/28/29 | 750.24/801.55/830.96 | typical |
| 1733 | 5 | 26/28/29 | 692.96/712.91/766.62 | typical |

**Robust across all 5 biological search seeds: no.**
The category is **not** the same at every search seed (seed 1729: narrower, seed 1730: narrower, seed 1731: typical, seed 1732: typical, seed 1733: typical) -- the primary seed 1729 result (**narrower**) alone must not be
read as the study's headline without this disclosure: it is not robust across search seeds, and seeds
1730, 1731, 1732, 1733 compare against only 5 seed-matched rewirings each.

### Occupancy maps (search seed 1729, row = turning bin (bottom = most negative), column = coverage bin (left = least covered))

#### Biological (1 = occupied, 0 = empty)

| turning \ coverage | 0%+ | 5%+ | 10%+ | 20%+ | 35%+ | 60%+ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.67..1.00 | 1 | 1 | 1 | 1 | 0 | 0 |
| 0.33..0.67 | 1 | 1 | 1 | 1 | 1 | 0 |
| 0.00..0.33 | 1 | 1 | 1 | 1 | 1 | 0 |
| -0.33..0.00 | 1 | 1 | 1 | 1 | 1 | 0 |
| -0.67..-0.33 | 1 | 1 | 1 | 1 | 1 | 0 |
| -1.00..-0.67 | 1 | 1 | 1 | 1 | 0 | 0 |

#### Rewired occupancy frequency (of 20 rewirings)

| turning \ coverage | 0%+ | 5%+ | 10%+ | 20%+ | 35%+ | 60%+ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.67..1.00 | 20 | 20 | 20 | 14 | 0 | 0 |
| 0.33..0.67 | 20 | 20 | 19 | 20 | 18 | 4 |
| 0.00..0.33 | 20 | 20 | 20 | 20 | 20 | 10 |
| -0.33..0.00 | 20 | 20 | 20 | 20 | 20 | 7 |
| -0.67..-0.33 | 20 | 20 | 20 | 20 | 19 | 3 |
| -1.00..-0.67 | 20 | 20 | 20 | 19 | 0 | 0 |

### Disconnected control (reference only, not part of the null)

| occupied | qd | span | heldoutOwnMedian | gpuArchiveSize | collisions |
| --- | --- | --- | --- | --- | --- |
| 24 | 343.90 | 11 | 9.17 | 24 | 0 |

The disconnected graph may produce degenerate behavior; it is reported as a reference point, not evaluated
against the predeclared categories above.

### Audit table

#### Biological, every search seed

| graph | seed | occupied | qd | span | heldoutOwnMedian | gpuArchiveSize | collisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| biological | 1729 | 28 | 637.49 | 11 | 16.55 | 28 | 0 |
| biological | 1730 | 28 | 618.73 | 11 | 15.38 | 28 | 0 |
| biological | 1731 | 30 | 751.41 | 12 | 15.49 | 30 | 0 |
| biological | 1732 | 29 | 693.84 | 11 | 17.74 | 29 | 0 |
| biological | 1733 | 30 | 711.37 | 12 | 18.42 | 30 | 0 |

#### Rewired, every searched (rewiring seed, search seed) pair

| graph | seed | occupied | qd | span | heldoutOwnMedian | gpuArchiveSize | collisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rewired-0 | 1729 | 29 | 763.37 | 12 | 19.23 | 30 | 1 |
| rewired-0 | 1730 | 29 | 827.89 | 12 | 24.26 | 29 | 0 |
| rewired-0 | 1731 | 29 | 846.38 | 12 | 22.66 | 29 | 0 |
| rewired-0 | 1732 | 30 | 853.77 | 12 | 18.47 | 30 | 0 |
| rewired-0 | 1733 | 25 | 692.96 | 11 | 18.96 | 28 | 3 |
| rewired-1 | 1729 | 27 | 725.27 | 11 | 21.49 | 27 | 0 |
| rewired-1 | 1730 | 31 | 879.80 | 12 | 17.83 | 31 | 0 |
| rewired-1 | 1731 | 32 | 946.21 | 12 | 28.06 | 32 | 0 |
| rewired-1 | 1732 | 28 | 750.24 | 12 | 21.39 | 30 | 2 |
| rewired-1 | 1733 | 29 | 712.91 | 12 | 20.87 | 29 | 0 |
| rewired-2 | 1729 | 28 | 695.11 | 12 | 17.88 | 28 | 0 |
| rewired-2 | 1730 | 29 | 722.91 | 12 | 16.62 | 29 | 0 |
| rewired-2 | 1731 | 29 | 803.22 | 12 | 22.58 | 29 | 0 |
| rewired-2 | 1732 | 28 | 830.96 | 12 | 22.61 | 28 | 0 |
| rewired-2 | 1733 | 28 | 766.62 | 12 | 19.47 | 28 | 0 |
| rewired-3 | 1729 | 28 | 749.82 | 12 | 16.70 | 28 | 0 |
| rewired-3 | 1730 | 28 | 736.26 | 12 | 18.72 | 28 | 0 |
| rewired-3 | 1731 | 31 | 857.41 | 12 | 22.05 | 31 | 0 |
| rewired-3 | 1732 | 27 | 727.55 | 11 | 20.42 | 27 | 0 |
| rewired-3 | 1733 | 26 | 592.04 | 11 | 15.09 | 26 | 0 |
| rewired-4 | 1729 | 29 | 733.14 | 12 | 20.75 | 29 | 0 |
| rewired-4 | 1730 | 30 | 852.72 | 12 | 22.67 | 30 | 0 |
| rewired-4 | 1731 | 30 | 823.41 | 12 | 22.77 | 30 | 0 |
| rewired-4 | 1732 | 29 | 801.55 | 12 | 19.30 | 29 | 0 |
| rewired-4 | 1733 | 31 | 835.83 | 12 | 17.10 | 31 | 0 |
| rewired-5 | 1729 | 29 | 893.03 | 12 | 26.60 | 29 | 0 |
| rewired-6 | 1729 | 28 | 702.16 | 11 | 17.98 | 28 | 0 |
| rewired-7 | 1729 | 29 | 728.15 | 12 | 20.74 | 29 | 0 |
| rewired-8 | 1729 | 26 | 616.61 | 11 | 15.32 | 26 | 0 |
| rewired-9 | 1729 | 27 | 743.88 | 11 | 17.80 | 27 | 0 |
| rewired-10 | 1729 | 28 | 808.45 | 11 | 22.78 | 28 | 0 |
| rewired-11 | 1729 | 29 | 878.05 | 12 | 26.88 | 30 | 1 |
| rewired-12 | 1729 | 31 | 828.44 | 12 | 21.55 | 31 | 0 |
| rewired-13 | 1729 | 30 | 845.26 | 12 | 20.26 | 30 | 0 |
| rewired-14 | 1729 | 29 | 786.08 | 12 | 25.65 | 29 | 0 |
| rewired-15 | 1729 | 26 | 759.78 | 11 | 21.20 | 27 | 1 |
| rewired-16 | 1729 | 30 | 922.39 | 12 | 27.94 | 30 | 0 |
| rewired-17 | 1729 | 29 | 688.32 | 12 | 16.11 | 29 | 0 |
| rewired-18 | 1729 | 29 | 896.59 | 12 | 24.03 | 29 | 0 |
| rewired-19 | 1729 | 32 | 763.00 | 12 | 14.31 | 32 | 0 |

## Limitations

- This experiment covers this model only: the authored encoder, dynamics, arena, descriptors, and the atlas
  readout family. There is no biological claim about fly behavior.
- One search budget only (population 64, generations 24, ticks
  900); a larger budget was declined and may change the result.
- `n = 20` rewirings at the primary seed is a 5% percentile resolution; only
  5 rewirings are seed-matched for the search-seed robustness check above
  (a 20% percentile resolution at those seeds).
- MAP-Elites coverage depends on the search budget and the descriptor space; a different budget or descriptor
  set could occupy different cells.
- Every claim here is descriptive and bound to this model and this search budget; no causal claim is made
  about the real fly.

## Provenance

- `sources.biologicalSha`: `f1a0f982ffdfba12ecb2206d064c3dc1ceaae7cffa06a22049093d69a79098f7`
- `sources.rewiringNullSha`: `06294ee5d631e26669ae52c999534b959a4156fa1a558f47c335033a58c4005f`
- `sources.evaluatedSha256`: `a1f18138a26f64fe569a0ca8a0aad4b01ea4f2a32089e088db88079932682ba5`
- `sources.atlasSha256`: `3cf39f80d017655a425d89416f989a093474a6628d2221eb91c839e36033b276`
- `sources.graphsIndexSha256`: `c887b555c559be56ef3b7c733609a5b672bd4e7e7db5ad3f2c23d1591574f812`
- Producer: `scripts/atlas/repertoire-report.ts`, sourceSha256 `57a8a76edf038e4843b8e6cb4d99ccee6d374e664de1dfe0f4a4c6575cd98971` (59 files)
- Host: arm64 / node v22.22.3
