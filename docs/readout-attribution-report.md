# Readout attribution (under this model)

**Question.** Why do CEM-retrained readouts erase the biological graph's deficit against its degree-preserving rewirings (trained biological scores 62.7-72.2 versus rewired scores 68.0-74.1), and why does the pathway intervention P gain nothing under trained readouts? This report tests three predeclared hypotheses against per-readout input saliency, readout-input ablation, input-independence share, and pathway linkage, computed on the 23 default-task readouts this artifact covers. Every result below is descriptive and bound to this model; none is a claim about the real fly.

## Predeclared analyses

Quoted verbatim from `.agents/plans/readout-attribution/00-overview.md`:

1. **Input saliency per descending neuron:** for each readout and each of the D = 48 output-neuron inputs, compute the mean absolute input-gradient of the readout's thrust and yaw outputs along that readout's own held-out trajectories (authoritative TS episodes, seeds 30001-30010, T 1800). The full chain rule is `d(out)/d(r_d) = diag(f_out'(W2*h + b2)) * W2 * diag(1 - h^2) * W1[:, d]`, where `f_out'` is `1 - tanh^2` for thrust and yaw. A secondary **variance-weighted** saliency (gradient x that input's trajectory std) is also reported.
2. **Readout-input ablation:** for the top-8 and bottom-8 descending inputs by saliency, per readout, zero that input in the **readout's input only**, leaving the dynamics intact. Measure the paired score change on 100 held-out seeds. This needs a new trained-decoder input mask. Zero is the rate model's equilibrium (leak toward 0; rates clamped to +/-2). Each ablated input's empirical trajectory mean and std are reported next to its effect, so an ablation that pushes a neuron outside its operating range is visible.
3. **Input-independence share:** per readout, `mean(silenced) / mean(trained)` over the 100 seeds (a ratio of means, never per-episode). If `mean(trained) <= 1`, the share is reported as undefined, which the report discloses. It measures how much of the policy ignores the circuit.
4. **Pathway linkage:** per descending input d, the steady-state clearance transfer `|T_clear,d|`, which is the d-row analogue of the explanation's transfer from the clearance channels to that neuron, computed on the readout's graph.

## Predeclared hypotheses and outcome rules

Quoted verbatim from `.agents/plans/readout-attribution/00-overview.md`:

- **H1 (consistent with routing around):** in biological readouts, saliency correlates with the clearance signal that bridge paths deliver. Spearman rho between per-neuron saliency and `|T_clear,d|` is >= 0.3, with a **cluster bootstrap** CI excluding 0. The resampling unit is the annotated `group` from the WP1c sidecar, or the output population if `group` is missing, not the 48 neurons treated as independent. H1 must hold in **all 3** biological trainer seeds, matching the repo's unanimous-seed convention (`trainedRobust`). If the cluster-bootstrap and neuron-level-bootstrap CIs disagree on excluding 0, H1 is inconclusive. **Regime gate:** H1 is evaluated only if the trained-readout trajectories pass the explanation study's regime thresholds (clamp fraction <= 20%, steady-state distance <= 0.5, measured by `regime-task.ts` logic on those trajectories). Otherwise H1 is inconclusive (regime-invalid). H1's rule defines only two outcomes, "supported" or "inconclusive" -- it has no "not-supported" branch.
- **H2 (constant policy, equivalence):** the input-independence share is equivalent between biological and rewired. Pairing is same-trainer-seed (101/202/303). "Supported" means the 90% CI of the mean paired difference lies inside +/-0.10, which is the predeclared smallest share difference of interest, a TOST-style test. "Not supported" means the CI excludes 0 and lies outside +/-0.10. Anything else is inconclusive. With n = 3 pairs the resolution is coarse, and the report says so.
- **H3 (P redundancy, equivalence or less):** in P's readouts (seeds 101/202/303), saliency on P's **newly connected thrust neurons**, identified from the archived P swap edges, is not larger than in the same-seed biological readouts. "Supported" means the upper 90% bound of the mean paired ratio (P/biological) is <= 1.25. "Not supported" means the lower bound is above 1.25. Anything else is inconclusive.
- No multiple-comparison correction is applied, and the number of hypotheses is disclosed. H1 is always described in correlational language ("consistent with"). Only the ablation results support causal wording, and then only about the readout's reliance on its inputs.

## Coverage

This artifact covers 23 default-task archived readouts. The committed archive (`training/archive/trained-readouts-v1.json`) holds 75 entries total; 52 `kind: "task-intervention"` entries across 4 non-default arena tasks (`crowded`/`hazard-heavy`/`no-movement`/`sparse-food`, from WP1b) are **excluded** here -- every predeclared analysis and H1-H3 rule in `00-overview.md` is scoped to fixed default-task ids (`biological-seed101`, `P-seed202`, ...) and was never defined over the per-task entries. `coverage.perTaskIncluded` is `false` in the JSON artifact.

| id | arm | graphId | trainerSeed | arenaTask |
| --- | --- | --- | --- | --- |
| `C000-seed101` | rewired | C000 | 101 | default |
| `C001-seed101` | rewired | C001 | 101 | default |
| `C002-seed101` | rewired | C002 | 101 | default |
| `C003-seed101` | rewired | C003 | 101 | default |
| `C004-seed101` | rewired | C004 | 101 | default |
| `M1000-seed101` | rewired | M1000 | 101 | default |
| `M1001-seed101` | rewired | M1001 | 101 | default |
| `M1002-seed101` | rewired | M1002 | 101 | default |
| `M1003-seed101` | rewired | M1003 | 101 | default |
| `M1004-seed101` | rewired | M1004 | 101 | default |
| `P-seed101` | rewired | P | 101 | default |
| `P-seed202` | rewired | P | 202 | default |
| `P-seed303` | rewired | P | 303 | default |
| `biological-seed101` | biological | biological | 101 | default |
| `biological-seed101-gpurerun` | biological | biological | 101 | default |
| `biological-seed202` | biological | biological | 202 | default |
| `biological-seed303` | biological | biological | 303 | default |
| `disconnected-seed101` | disconnected | disconnected | 101 | default |
| `disconnected-seed202` | disconnected | disconnected | 202 | default |
| `disconnected-seed303` | disconnected | disconnected | 303 | default |
| `rewired-seed0-seed101` | rewired | rewired-seed0 | 101 | default |
| `rewired-seed0-seed202` | rewired | rewired-seed0 | 202 | default |
| `rewired-seed0-seed303` | rewired | rewired-seed0 | 303 | default |

## Saliency

Analysis 1 (`00-overview.md`): mean absolute input-gradient of thrust/yaw along each readout's own 10 held-out trajectories (seeds 30001-30010, 1800 ticks). Named against `descending-types-v1.json`'s pinned cell-type annotations -- labels only, not a claim about fly descending-neuron function. Below, per readout, the top 10 of the 48 descending inputs by combined (thrust + yaw) saliency score, in descending order:

| id | top 10 by combined thrust+yaw saliency |
| --- | --- |
| `C000-seed101` | DNg100_R (#4, 2.094); DNp103_R (#22, 1.855); DNg75_R (#43, 1.033); DNg105_L (#8, 1.030); DNg74_a_L (#17, 0.997); DNg97_L (#38, 0.846); DNg35_R (#13, 0.790); DNge050_L (#24, 0.762); DNpe053_R (#39, 0.735); DNge050_R (#41, 0.680) |
| `C001-seed101` | DNg100_R (#4, 1.105); DNge122_R (#35, 1.038); DNg16_L (#10, 0.968); DNg75_R (#43, 0.930); DNg74_b_L (#30, 0.864); pIP1_L (#1, 0.762); DNg98_L (#27, 0.735); DNp31_R (#19, 0.675); DNge050_R (#41, 0.563); DNpe053_R (#39, 0.541) |
| `C002-seed101` | DNg35_R (#13, 1.058); DNg100_L (#3, 0.789); DNpe053_R (#39, 0.539); DNge122_R (#35, 0.388); DNa03_R (#34, 0.333); DNge050_L (#24, 0.331); DNbe001_R (#45, 0.322); aSP22_L (#6, 0.318); DNp103_R (#22, 0.316); DNg16_L (#10, 0.312) |
| `C003-seed101` | DNge122_R (#35, 0.400); DNp35_L (#16, 0.367); DNg105_R (#37, 0.341); DNb05_L (#7, 0.338); DNb05_R (#5, 0.327); DNp31_L (#18, 0.311); DNge079_L (#47, 0.305); DNp31_R (#19, 0.305); DNge037_L (#14, 0.285); DNa02_L (#44, 0.251) |
| `C004-seed101` | DNg16_R (#29, 0.778); DNa03_L (#42, 0.737); DNp13_R (#26, 0.582); DNp35_R (#31, 0.572); DNg100_R (#4, 0.533); DNg97_L (#38, 0.481); DNg35_R (#13, 0.478); DNge122_R (#35, 0.403); DNg108_R (#21, 0.352); DNbe001_R (#45, 0.300) |
| `M1000-seed101` | DNp35_L (#16, 0.989); DNg108_L (#20, 0.562); aSP22_L (#6, 0.553); DNg105_L (#8, 0.515); DNg100_R (#4, 0.476); DNpe053_L (#23, 0.465); DNpe053_R (#39, 0.437); DNg16_R (#29, 0.400); DNg105_R (#37, 0.378); DNg74_a_R (#9, 0.361) |
| `M1001-seed101` | DNg105_R (#37, 2.733); DNg105_L (#8, 2.559); DNge050_L (#24, 2.116); DNp31_R (#19, 1.783); DNp103_R (#22, 1.376); DNg108_L (#20, 1.240); DNg35_R (#13, 1.179); DNpe042_R (#36, 0.752); DNpe053_L (#23, 0.716); DNg100_L (#3, 0.683) |
| `M1002-seed101` | DNge122_R (#35, 1.175); DNpe053_R (#39, 1.133); DNge079_R (#46, 0.976); DNp103_L (#28, 0.929); DNg16_L (#10, 0.741); DNp31_R (#19, 0.700); DNg105_R (#37, 0.669); DNge037_L (#14, 0.655); DNp06_L (#15, 0.611); DNg105_L (#8, 0.602) |
| `M1003-seed101` | DNg35_R (#13, 1.262); DNg97_R (#40, 1.185); DNp06_R (#32, 0.825); aSP22_L (#6, 0.798); DNa03_L (#42, 0.765); DNp01_L (#0, 0.752); DNg108_L (#20, 0.743); DNp31_L (#18, 0.722); DNg98_R (#33, 0.710); DNg16_R (#29, 0.670) |
| `M1004-seed101` | DNg16_R (#29, 0.744); DNp06_R (#32, 0.560); DNg75_L (#11, 0.539); aSP22_L (#6, 0.526); DNg100_L (#3, 0.522); DNg105_L (#8, 0.502); DNb05_R (#5, 0.475); pIP1_R (#2, 0.445); DNp31_L (#18, 0.443); DNp31_R (#19, 0.394) |
| `P-seed101` | DNge122_R (#35, 1.305); DNge050_L (#24, 1.264); DNg74_a_R (#9, 0.595); DNg16_R (#29, 0.582); DNg74_b_L (#30, 0.535); DNa02_L (#44, 0.482); DNp06_L (#15, 0.432); DNp01_L (#0, 0.430); DNg97_L (#38, 0.425); DNge037_R (#12, 0.410) |
| `P-seed202` | DNge037_L (#14, 0.698); DNp06_R (#32, 0.647); DNbe001_R (#45, 0.481); DNge050_R (#41, 0.414); DNge122_R (#35, 0.409); DNpe053_L (#23, 0.387); DNg105_L (#8, 0.384); DNa02_R (#25, 0.382); DNa03_R (#34, 0.342); DNg97_L (#38, 0.310) |
| `P-seed303` | DNpe053_L (#23, 0.429); DNge122_R (#35, 0.389); DNpe053_R (#39, 0.363); DNge037_R (#12, 0.346); DNp103_L (#28, 0.301); DNp103_R (#22, 0.251); DNg74_a_R (#9, 0.251); DNge037_L (#14, 0.244); DNg98_R (#33, 0.235); DNa03_R (#34, 0.233) |
| `biological-seed101` | DNge122_R (#35, 1.213); DNb05_R (#5, 0.845); DNge079_R (#46, 0.797); DNpe053_R (#39, 0.778); DNp31_R (#19, 0.751); DNg74_a_L (#17, 0.738); DNg16_L (#10, 0.686); DNge037_L (#14, 0.613); DNg74_a_R (#9, 0.573); DNpe053_L (#23, 0.485) |
| `biological-seed101-gpurerun` | DNge079_L (#47, 1.429); DNge050_L (#24, 1.295); DNg100_L (#3, 1.207); DNb05_L (#7, 0.871); DNg105_R (#37, 0.849); DNg108_L (#20, 0.822); DNp13_R (#26, 0.775); DNp103_R (#22, 0.756); DNp31_R (#19, 0.750); DNg108_R (#21, 0.713) |
| `biological-seed202` | DNge079_L (#47, 0.639); DNp13_R (#26, 0.622); DNg75_R (#43, 0.484); DNp01_L (#0, 0.441); DNg108_L (#20, 0.393); DNp103_R (#22, 0.378); DNg74_b_L (#30, 0.367); DNge050_L (#24, 0.361); DNa03_R (#34, 0.349); DNg74_a_L (#17, 0.338) |
| `biological-seed303` | DNg108_R (#21, 1.387); DNge122_R (#35, 1.115); DNg98_R (#33, 0.811); DNge079_L (#47, 0.602); pIP1_L (#1, 0.601); aSP22_L (#6, 0.551); DNge050_L (#24, 0.526); DNg105_R (#37, 0.518); DNp13_R (#26, 0.502); DNa03_R (#34, 0.495) |
| `disconnected-seed101` | DNg97_R (#40, 0.754); aSP22_L (#6, 0.537); DNp01_L (#0, 0.527); DNge037_R (#12, 0.500); pIP1_R (#2, 0.464); pIP1_L (#1, 0.448); DNp13_R (#26, 0.437); DNp06_L (#15, 0.433); DNg74_a_R (#9, 0.426); DNge050_L (#24, 0.423) |
| `disconnected-seed202` | DNge079_L (#47, 0.854); DNg74_a_L (#17, 0.815); DNpe042_R (#36, 0.765); DNg105_R (#37, 0.722); DNge079_R (#46, 0.694); DNb05_R (#5, 0.691); DNa02_L (#44, 0.591); DNg75_L (#11, 0.556); DNg98_L (#27, 0.507); DNg105_L (#8, 0.490) |
| `disconnected-seed303` | DNg108_L (#20, 1.390); DNge050_L (#24, 1.290); DNg98_L (#27, 0.943); DNge050_R (#41, 0.931); DNa03_L (#42, 0.886); DNg97_R (#40, 0.780); DNa02_L (#44, 0.770); DNg74_a_L (#17, 0.757); DNg105_L (#8, 0.712); DNg100_L (#3, 0.673) |
| `rewired-seed0-seed101` | DNg105_R (#37, 1.687); DNg100_R (#4, 1.169); DNb05_R (#5, 1.036); DNg100_L (#3, 0.860); DNg74_a_R (#9, 0.841); DNp06_L (#15, 0.764); DNg108_L (#20, 0.737); DNg98_L (#27, 0.713); DNg105_L (#8, 0.670); DNg108_R (#21, 0.624) |
| `rewired-seed0-seed202` | DNge122_R (#35, 1.698); DNg105_R (#37, 0.921); DNpe042_R (#36, 0.886); DNp13_R (#26, 0.787); DNge079_R (#46, 0.664); DNg100_L (#3, 0.645); DNge037_L (#14, 0.626); DNg75_R (#43, 0.583); DNg98_L (#27, 0.551); DNg74_a_R (#9, 0.535) |
| `rewired-seed0-seed303` | DNg100_R (#4, 1.578); DNg35_R (#13, 1.509); DNpe042_R (#36, 1.251); DNg108_R (#21, 1.215); DNge050_L (#24, 1.158); DNp31_L (#18, 1.044); DNpe053_L (#23, 1.019); DNg100_L (#3, 0.825); DNg97_L (#38, 0.745); DNg75_R (#43, 0.731) |

Full per-input thrust/yaw/variance-weighted saliency arrays (all 48 inputs, every readout) are in the JSON artifact's `readouts[].saliency`/`saliencyVarWeighted`.

## Ablation

Analysis 2 (`00-overview.md`): for the top-8 and bottom-8 descending inputs by saliency (thrust+yaw summed), zero that input in the readout's input only (dynamics untouched), and measure the paired score change on 100 held-out seeds (seeds 30001-30100). Full detail below for the 6 headline readouts central to H1/H3 (biological and P, all three trainer seeds each); every readout's full 16-input ablation table is in the JSON artifact's `readouts[].ablation`.

#### `P-seed101` (baseline held-out score: see `independence.json`'s `trainedMean`; n=100 paired seeds per input)

| rank | input | saliency score | input mean | input std | effect (paired mean diff) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| top | DNge122_R (#35) | 1.305 | 0.6371 | 0.3355 | -47.351 | [-51.574, -43.312] |
| top | DNge050_L (#24) | 1.264 | 0.0033 | 0.0063 | -1.844 | [-6.887, 3.581] |
| top | DNg74_a_R (#9) | 0.595 | 0.0137 | 0.0087 | -1.232 | [-6.304, 4.197] |
| top | DNg16_R (#29) | 0.582 | 0.0283 | 0.0198 | 2.837 | [-1.869, 7.596] |
| top | DNg74_b_L (#30) | 0.535 | -0.0036 | 0.0080 | -0.732 | [-4.910, 3.668] |
| top | DNa02_L (#44) | 0.482 | 0.0002 | 0.0006 | -1.403 | [-4.767, 1.970] |
| top | DNp06_L (#15) | 0.432 | 0.0149 | 0.0056 | -2.224 | [-6.496, 1.776] |
| top | DNp01_L (#0) | 0.430 | 0.0010 | 0.0016 | -2.026 | [-6.401, 2.176] |
| bottom | pIP1_R (#2) | 0.075 | 0.0657 | 0.0319 | -1.239 | [-5.465, 3.204] |
| bottom | DNg105_R (#37) | 0.070 | 0.0058 | 0.0108 | 2.660 | [-2.577, 7.187] |
| bottom | DNg98_R (#33) | 0.050 | 0.0036 | 0.0021 | 0.145 | [-3.593, 3.828] |
| bottom | DNg100_L (#3) | 0.038 | 0.0165 | 0.0208 | -3.563 | [-8.613, 1.415] |
| bottom | DNg75_L (#11) | 0.036 | 0.0030 | 0.0060 | 0.491 | [-3.668, 4.754] |
| bottom | DNg108_L (#20) | 0.035 | 0.0642 | 0.0289 | -1.015 | [-5.188, 3.225] |
| bottom | DNg108_R (#21) | 0.008 | 0.0612 | 0.0267 | -1.788 | [-6.145, 2.454] |
| bottom | DNg74_a_L (#17) | 0.005 | 0.0009 | 0.0071 | -1.994 | [-5.081, 1.073] |

#### `P-seed202` (baseline held-out score: see `independence.json`'s `trainedMean`; n=100 paired seeds per input)

| rank | input | saliency score | input mean | input std | effect (paired mean diff) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| top | DNge037_L (#14) | 0.698 | 0.2685 | 0.0606 | -13.543 | [-18.858, -8.667] |
| top | DNp06_R (#32) | 0.647 | 0.0065 | 0.0025 | 2.501 | [-0.936, 5.681] |
| top | DNbe001_R (#45) | 0.481 | -0.0001 | 0.0000 | -2.506 | [-5.153, -0.091] |
| top | DNge050_R (#41) | 0.414 | 0.0016 | 0.0032 | 0.914 | [-2.274, 4.119] |
| top | DNge122_R (#35) | 0.409 | 0.5619 | 0.1981 | -33.120 | [-38.056, -27.776] |
| top | DNpe053_L (#23) | 0.387 | 0.0032 | 0.0029 | 3.159 | [-0.604, 6.894] |
| top | DNg105_L (#8) | 0.384 | 0.0081 | 0.0051 | 0.274 | [-3.767, 4.497] |
| top | DNa02_R (#25) | 0.382 | 0.0078 | 0.0018 | 3.255 | [-0.401, 6.761] |
| bottom | pIP1_L (#1) | 0.040 | -0.0006 | 0.0030 | -0.629 | [-3.593, 2.625] |
| bottom | DNge079_L (#47) | 0.034 | 0.0090 | 0.0073 | 1.582 | [-2.307, 5.645] |
| bottom | DNg74_a_L (#17) | 0.032 | 0.0027 | 0.0035 | 0.143 | [-2.855, 3.232] |
| bottom | DNg108_L (#20) | 0.031 | 0.0615 | 0.0188 | 2.211 | [-1.577, 6.105] |
| bottom | DNg74_b_L (#30) | 0.030 | -0.0015 | 0.0041 | 1.291 | [-1.723, 4.553] |
| bottom | DNge079_R (#46) | 0.025 | 0.0117 | 0.0070 | 0.095 | [-3.102, 3.466] |
| bottom | DNp01_L (#0) | 0.025 | 0.0008 | 0.0009 | 1.526 | [-1.011, 4.026] |
| bottom | DNp06_L (#15) | 0.005 | 0.0145 | 0.0040 | -0.163 | [-3.161, 3.043] |

#### `P-seed303` (baseline held-out score: see `independence.json`'s `trainedMean`; n=100 paired seeds per input)

| rank | input | saliency score | input mean | input std | effect (paired mean diff) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| top | DNpe053_L (#23) | 0.429 | 0.0043 | 0.0036 | -1.484 | [-4.955, 2.070] |
| top | DNge122_R (#35) | 0.389 | 0.6200 | 0.2356 | -33.035 | [-37.203, -28.801] |
| top | DNpe053_R (#39) | 0.363 | 0.0032 | 0.0033 | 0.818 | [-3.108, 4.569] |
| top | DNge037_R (#12) | 0.346 | 0.2134 | 0.0647 | -4.094 | [-8.918, 0.332] |
| top | DNp103_L (#28) | 0.301 | 0.0250 | 0.0062 | 2.345 | [-2.008, 7.036] |
| top | DNp103_R (#22) | 0.251 | 0.0230 | 0.0093 | 0.765 | [-3.312, 5.084] |
| top | DNg74_a_R (#9) | 0.251 | 0.0178 | 0.0052 | 0.659 | [-3.256, 4.703] |
| top | DNge037_L (#14) | 0.244 | 0.2714 | 0.0655 | -2.212 | [-6.774, 2.268] |
| bottom | DNg75_L (#11) | 0.030 | 0.0034 | 0.0036 | 1.001 | [-2.390, 4.203] |
| bottom | DNp06_R (#32) | 0.022 | 0.0075 | 0.0031 | 2.265 | [-1.334, 6.009] |
| bottom | pIP1_L (#1) | 0.019 | -0.0014 | 0.0037 | 1.483 | [-1.458, 4.442] |
| bottom | DNg16_L (#10) | 0.019 | 0.0301 | 0.0144 | 0.174 | [-3.160, 3.601] |
| bottom | DNg35_R (#13) | 0.014 | -0.0074 | 0.0107 | 0.745 | [-2.813, 4.104] |
| bottom | DNp31_R (#19) | 0.012 | 0.0001 | 0.0000 | 0.734 | [-1.123, 2.689] |
| bottom | DNp06_L (#15) | 0.008 | 0.0157 | 0.0047 | 1.922 | [-1.550, 5.480] |
| bottom | DNg100_R (#4) | 0.006 | 0.0232 | 0.0114 | 2.469 | [-0.735, 5.568] |

#### `biological-seed101` (baseline held-out score: see `independence.json`'s `trainedMean`; n=100 paired seeds per input)

| rank | input | saliency score | input mean | input std | effect (paired mean diff) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| top | DNge122_R (#35) | 1.213 | 0.6680 | 0.3579 | -45.439 | [-50.007, -41.483] |
| top | DNb05_R (#5) | 0.845 | 0.0002 | 0.0003 | 0.934 | [-2.880, 4.972] |
| top | DNge079_R (#46) | 0.797 | 0.0047 | 0.0154 | 2.526 | [-2.981, 7.744] |
| top | DNpe053_R (#39) | 0.778 | 0.0045 | 0.0054 | 2.144 | [-2.827, 6.844] |
| top | DNp31_R (#19) | 0.751 | 0.0000 | 0.0001 | 0.096 | [-3.507, 3.663] |
| top | DNg74_a_L (#17) | 0.738 | -0.0013 | 0.0081 | 3.648 | [-1.426, 8.420] |
| top | DNg16_L (#10) | 0.686 | 0.0270 | 0.0239 | -0.412 | [-5.910, 5.268] |
| top | DNge037_L (#14) | 0.613 | 0.0141 | 0.0098 | 2.765 | [-2.531, 7.860] |
| bottom | DNp01_L (#0) | 0.074 | 0.0009 | 0.0018 | -0.406 | [-3.627, 2.895] |
| bottom | DNg74_b_L (#30) | 0.047 | -0.0043 | 0.0093 | 0.503 | [-4.174, 4.970] |
| bottom | pIP1_R (#2) | 0.044 | -0.0083 | 0.0052 | 1.937 | [-2.805, 6.314] |
| bottom | DNge037_R (#12) | 0.037 | 0.0210 | 0.0108 | -1.603 | [-5.917, 2.795] |
| bottom | DNg105_R (#37) | 0.030 | -0.0103 | 0.0138 | -0.150 | [-4.352, 4.182] |
| bottom | DNpe042_R (#36) | 0.029 | 0.0136 | 0.0056 | 0.898 | [-3.432, 5.166] |
| bottom | DNp06_R (#32) | 0.029 | 0.0079 | 0.0041 | 2.393 | [-1.374, 6.121] |
| bottom | DNg97_R (#40) | 0.006 | 0.0163 | 0.0149 | -0.327 | [-3.819, 3.011] |

#### `biological-seed202` (baseline held-out score: see `independence.json`'s `trainedMean`; n=100 paired seeds per input)

| rank | input | saliency score | input mean | input std | effect (paired mean diff) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| top | DNge079_L (#47) | 0.639 | 0.0085 | 0.0149 | 0.328 | [-4.492, 5.191] |
| top | DNp13_R (#26) | 0.622 | 0.0020 | 0.0011 | -3.305 | [-6.609, 0.020] |
| top | DNg75_R (#43) | 0.484 | 0.0039 | 0.0067 | -2.191 | [-6.889, 2.181] |
| top | DNp01_L (#0) | 0.441 | 0.0003 | 0.0016 | -2.214 | [-5.760, 1.108] |
| top | DNg108_L (#20) | 0.393 | 0.0530 | 0.0333 | -4.698 | [-9.357, -0.175] |
| top | DNp103_R (#22) | 0.378 | 0.0173 | 0.0137 | -2.130 | [-6.683, 2.235] |
| top | DNg74_b_L (#30) | 0.367 | -0.0020 | 0.0084 | 0.320 | [-3.607, 4.462] |
| top | DNge050_L (#24) | 0.361 | 0.0000 | 0.0067 | -0.748 | [-4.885, 3.390] |
| bottom | DNge079_R (#46) | 0.092 | 0.0107 | 0.0142 | 2.283 | [-0.941, 5.556] |
| bottom | pIP1_R (#2) | 0.089 | -0.0057 | 0.0054 | 0.111 | [-2.108, 2.287] |
| bottom | DNa02_L (#44) | 0.073 | 0.0002 | 0.0006 | -0.041 | [-1.407, 1.317] |
| bottom | DNp06_L (#15) | 0.037 | 0.0137 | 0.0065 | -2.206 | [-5.343, 0.848] |
| bottom | DNg16_R (#29) | 0.033 | 0.0216 | 0.0225 | -0.216 | [-3.290, 2.837] |
| bottom | DNg97_R (#40) | 0.030 | 0.0107 | 0.0154 | -0.469 | [-2.919, 1.919] |
| bottom | DNg74_a_R (#9) | 0.022 | 0.0143 | 0.0092 | -1.512 | [-4.312, 1.266] |
| bottom | DNg98_L (#27) | 0.009 | 0.0014 | 0.0039 | -0.851 | [-2.420, 0.613] |

#### `biological-seed303` (baseline held-out score: see `independence.json`'s `trainedMean`; n=100 paired seeds per input)

| rank | input | saliency score | input mean | input std | effect (paired mean diff) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| top | DNg108_R (#21) | 1.387 | 0.0616 | 0.0282 | -3.800 | [-9.113, 1.322] |
| top | DNge122_R (#35) | 1.115 | 0.6343 | 0.3544 | -44.235 | [-48.998, -39.399] |
| top | DNg98_R (#33) | 0.811 | 0.0036 | 0.0023 | -0.528 | [-5.787, 4.600] |
| top | DNge079_L (#47) | 0.602 | 0.0040 | 0.0153 | -1.507 | [-6.536, 3.625] |
| top | pIP1_L (#1) | 0.601 | -0.0018 | 0.0060 | -3.721 | [-8.303, 1.092] |
| top | aSP22_L (#6) | 0.551 | -0.0009 | 0.0006 | 0.939 | [-3.589, 5.278] |
| top | DNge050_L (#24) | 0.526 | 0.0038 | 0.0070 | -2.212 | [-6.778, 2.641] |
| top | DNg105_R (#37) | 0.518 | -0.0085 | 0.0130 | -0.808 | [-5.583, 3.849] |
| bottom | DNb05_R (#5) | 0.100 | 0.0002 | 0.0003 | 0.792 | [-1.917, 3.329] |
| bottom | DNp103_R (#22) | 0.077 | 0.0240 | 0.0134 | -1.009 | [-6.320, 3.858] |
| bottom | DNg35_R (#13) | 0.059 | -0.0201 | 0.0187 | -3.470 | [-7.761, 0.606] |
| bottom | pIP1_R (#2) | 0.058 | -0.0085 | 0.0050 | 0.813 | [-3.323, 4.967] |
| bottom | DNb05_L (#7) | 0.053 | 0.0001 | 0.0006 | 0.948 | [-1.413, 3.329] |
| bottom | DNg97_L (#38) | 0.038 | 0.0207 | 0.0159 | -0.915 | [-5.479, 3.457] |
| bottom | DNpe053_L (#23) | 0.022 | 0.0055 | 0.0059 | 0.598 | [-3.155, 4.219] |
| bottom | DNge037_L (#14) | 0.018 | 0.0136 | 0.0097 | -2.905 | [-6.540, 0.607] |

## Input independence

Analysis 3 (`00-overview.md`): `mean(silenced) / mean(trained)` over 100 held-out seeds (a ratio of means, never per-episode); undefined when `mean(trained) <= 1`.

| id | trained mean | silenced mean | ratio | defined |
| --- | --- | --- | --- | --- |
| `C000-seed101` | 70.549 | 30.321 | 0.4298 | yes |
| `C001-seed101` | 62.441 | 17.390 | 0.2785 | yes |
| `C002-seed101` | 71.582 | 34.882 | 0.4873 | yes |
| `C003-seed101` | 70.033 | 35.632 | 0.5088 | yes |
| `C004-seed101` | 70.494 | 35.105 | 0.4980 | yes |
| `M1000-seed101` | 72.360 | 41.494 | 0.5734 | yes |
| `M1001-seed101` | 72.859 | 38.119 | 0.5232 | yes |
| `M1002-seed101` | 65.901 | 17.304 | 0.2626 | yes |
| `M1003-seed101` | 75.487 | 23.527 | 0.3117 | yes |
| `M1004-seed101` | 69.615 | 35.520 | 0.5102 | yes |
| `P-seed101` | 65.534 | 17.795 | 0.2715 | yes |
| `P-seed202` | 70.271 | 31.681 | 0.4508 | yes |
| `P-seed303` | 70.667 | 34.140 | 0.4831 | yes |
| `biological-seed101` | 62.963 | 17.300 | 0.2748 | yes |
| `biological-seed101-gpurerun` | 69.317 | 35.631 | 0.5140 | yes |
| `biological-seed202` | 72.230 | 34.827 | 0.4822 | yes |
| `biological-seed303` | 62.742 | 18.468 | 0.2943 | yes |
| `disconnected-seed101` | 35.559 | 35.559 | 1.0000 | yes |
| `disconnected-seed202` | 35.781 | 35.781 | 1.0000 | yes |
| `disconnected-seed303` | 36.188 | 36.188 | 1.0000 | yes |
| `rewired-seed0-seed101` | 68.724 | 37.710 | 0.5487 | yes |
| `rewired-seed0-seed202` | 74.067 | 21.365 | 0.2885 | yes |
| `rewired-seed0-seed303` | 68.022 | 35.480 | 0.5216 | yes |

## Pathway linkage

Analysis 4 (`00-overview.md`): per descending input d, the steady-state clearance transfer `|T_clear,d|`, correlated (Spearman) against saliency, with a cluster bootstrap over the pinned annotations' `group` (28 clusters for the 48 neurons: 20 bilateral pairs and 8 singletons) and a neuron-level bootstrap reported as a sensitivity check.

| id | rho (thrust) | rho (yaw) | 90% CI (cluster) | 90% CI (neuron) | cluster count |
| --- | --- | --- | --- | --- | --- |
| `C000-seed101` | 0.200 | 0.042 | [-0.093, 0.440] | [-0.095, 0.451] | 28 |
| `C001-seed101` | -0.182 | 0.183 | [-0.438, 0.110] | [-0.475, 0.140] | 28 |
| `C002-seed101` | -0.185 | 0.070 | [-0.429, 0.091] | [-0.454, 0.099] | 28 |
| `C003-seed101` | -0.112 | -0.264 | [-0.444, 0.215] | [-0.402, 0.177] | 28 |
| `C004-seed101` | 0.096 | 0.290 | [-0.136, 0.356] | [-0.209, 0.370] | 28 |
| `M1000-seed101` | -0.071 | 0.061 | [-0.347, 0.247] | [-0.357, 0.247] | 28 |
| `M1001-seed101` | -0.073 | 0.057 | [-0.360, 0.239] | [-0.331, 0.192] | 28 |
| `M1002-seed101` | 0.004 | 0.030 | [-0.258, 0.289] | [-0.255, 0.276] | 28 |
| `M1003-seed101` | 0.268 | 0.205 | [0.036, 0.498] | [-0.034, 0.533] | 28 |
| `M1004-seed101` | -0.066 | 0.012 | [-0.349, 0.265] | [-0.369, 0.232] | 28 |
| `P-seed101` | -0.116 | -0.078 | [-0.382, 0.167] | [-0.414, 0.178] | 28 |
| `P-seed202` | -0.002 | -0.065 | [-0.291, 0.291] | [-0.274, 0.270] | 28 |
| `P-seed303` | 0.168 | 0.148 | [-0.079, 0.410] | [-0.093, 0.413] | 28 |
| `biological-seed101` | -0.238 | -0.058 | [-0.533, 0.063] | [-0.501, 0.038] | 28 |
| `biological-seed101-gpurerun` | -0.109 | 0.118 | [-0.356, 0.161] | [-0.388, 0.178] | 28 |
| `biological-seed202` | -0.014 | -0.186 | [-0.240, 0.238] | [-0.277, 0.262] | 28 |
| `biological-seed303` | -0.024 | -0.093 | [-0.375, 0.268] | [-0.333, 0.277] | 28 |
| `disconnected-seed101` | degenerate | n/a | n/a | n/a | 28 |
| `disconnected-seed202` | degenerate | n/a | n/a | n/a | 28 |
| `disconnected-seed303` | degenerate | n/a | n/a | n/a | 28 |
| `rewired-seed0-seed101` | -0.166 | 0.059 | [-0.407, 0.082] | [-0.454, 0.155] | 28 |
| `rewired-seed0-seed202` | -0.215 | -0.011 | [-0.441, -0.001] | [-0.468, 0.032] | 28 |
| `rewired-seed0-seed303` | 0.108 | 0.174 | [-0.198, 0.364] | [-0.189, 0.373] | 28 |

## Regime check (H1's gate)

The same clamp-fraction/steady-state-distance statistic, at the same granularity, as the explanation study (`regime-metrics.ts`, extracted from `regime-task.ts`); H1 is evaluated only for a biological seed whose trajectories pass both thresholds (clamp fraction <= 20%, steady-state distance <= 0.5).

| id | clamp fraction | steady-state distance | valid |
| --- | --- | --- | --- |
| `C000-seed101` | 0.0203 | 0.4201 | yes |
| `C001-seed101` | 0.0183 | 0.4350 | yes |
| `C002-seed101` | 0.0202 | 0.4251 | yes |
| `C003-seed101` | 0.0202 | 0.4317 | yes |
| `C004-seed101` | 0.0205 | 0.4293 | yes |
| `M1000-seed101` | 0.0208 | 0.3973 | yes |
| `M1001-seed101` | 0.0206 | 0.4206 | yes |
| `M1002-seed101` | 0.0175 | 0.4474 | yes |
| `M1003-seed101` | 0.0210 | 0.4094 | yes |
| `M1004-seed101` | 0.0203 | 0.4128 | yes |
| `P-seed101` | 0.0179 | 0.4458 | yes |
| `P-seed202` | 0.0209 | 0.4037 | yes |
| `P-seed303` | 0.0203 | 0.4253 | yes |
| `biological-seed101` | 0.0180 | 0.4455 | yes |
| `biological-seed101-gpurerun` | 0.0202 | 0.4216 | yes |
| `biological-seed202` | 0.0245 | 0.4352 | yes |
| `biological-seed303` | 0.0181 | 0.4454 | yes |
| `disconnected-seed101` | 0.0220 | 0.3590 | yes |
| `disconnected-seed202` | 0.0215 | 0.3619 | yes |
| `disconnected-seed303` | 0.0218 | 0.3630 | yes |
| `rewired-seed0-seed101` | 0.0180 | 0.4366 | yes |
| `rewired-seed0-seed202` | 0.0196 | 0.4277 | yes |
| `rewired-seed0-seed303` | 0.0180 | 0.4339 | yes |

## Hypothesis outcomes

Hypothesis count: 3. Multiple-comparison correction: none.

### H1 (consistent with routing around)

**Outcome: inconclusive** (threshold-not-met-in-all-seeds). Threshold: rho >= 0.3, unanimous across all 3 biological trainer seeds, cluster-bootstrap CI excluding 0, agreeing with the neuron-level bootstrap.

| seed | regime valid | rho (thrust) | 90% CI (cluster) | 90% CI (neuron) | meets threshold |
| --- | --- | --- | --- | --- | --- |
| `biological-seed101` | yes | -0.238 | [-0.533, 0.063] | [-0.501, 0.038] | no |
| `biological-seed202` | yes | -0.014 | [-0.240, 0.238] | [-0.277, 0.262] | no |
| `biological-seed303` | yes | -0.024 | [-0.375, 0.268] | [-0.333, 0.277] | no |

None of the 3 biological trainer seeds' rho meets the >= 0.3 threshold (seed101: -0.238, seed202: -0.014, seed303: -0.024) -- two of the three are even negative. H1's rule has no "not-supported" branch (see the predeclared rules above); with the threshold unmet in all seeds, the mechanical result is **inconclusive**, stated exactly as the plan's rule defines it, never read as a stronger "refuted" claim.

### H2 (constant policy, equivalence)

**Outcome: inconclusive**. Equivalence bound: +/-0.1. Paired differences (biological minus rewired independence share, same trainer seed): -0.2739, 0.1937, -0.2272 (n=3). t-based 90% CI of the mean paired difference: [-0.5367, 0.3318].

The CI ([-0.5367, 0.3318]) neither sits entirely inside +/-0.1 (which "supported" would require) nor entirely outside it while excluding 0 (which "not supported" would require) -- it straddles both boundaries. With n = 3 paired seeds the resolution is coarse, exactly as `00-overview.md` anticipates; the mechanical result is **inconclusive**, read as inconclusive, never as "not consistent" or any stronger claim.

### H3 (P redundancy, equivalence or less)

**Outcome: inconclusive**. Ratio bound: <= 1.25. P's newly connected thrust neurons (D-space indices, from the archived swap edges): 2, 12, 14.

| seed | P mean saliency | biological mean saliency | ratio (P / biological) |
| --- | --- | --- | --- |
| 101 | 0.000412 | 0.000288 | 1.43 |
| 202 | 0.007991 | 0.000035 | 227.85 |
| 303 | 0.002958 | 0.005632 | 0.53 |

t-based 90% CI of the mean paired ratio: [-144.22, 297.42]. The per-seed ratios are highly skewed: seed 202's ratio (227.85) is driven by a near-zero biological mean saliency denominator (0.000035) in that one seed, not by a genuinely large P effect -- stated plainly, not smoothed over. The CI's negative lower bound (-144.22) is an artifact of fitting a t-interval to 3 highly skewed values, not evidence that the ratio is ever negative (a mean saliency ratio cannot be negative by construction). With the CI this wide and straddling the <= 1.25 bound in both directions, the mechanical result is **inconclusive**.

## What remains unexplained

This study set out to explain, under this model, why CEM-retrained readouts erase the biological graph's deficit: trained biological scores 62.7-72.2 versus rewired scores 68.0-74.1 (`docs/trained-readout-report.md`), and why the pathway intervention P gains nothing under trained readouts (`docs/pathway-interventions-report.md`). All three predeclared hypotheses came back **inconclusive**, so this evidence does not account for either result -- not partially, not with caveats folded in, but genuinely not.

For the ~5-10 point score-convergence gap: H1 (routing-around) found no rho >= 0.3 in any of the 3 biological trainer seeds -- two of three seeds even show a small negative correlation between saliency and the clearance transfer. H2 (constant-policy equivalence) could neither confirm nor rule out that biological and rewired trained readouts rely on their circuit to the same degree; its CI straddles both the equivalence and non-equivalence boundaries at n=3. Neither result points at a mechanism for why training closes the gap -- the combined evidence leaves that question exactly where it started, not partially resolved.

For P's null trained result: H3 (P-redundancy) could neither confirm nor rule out that saliency on P's newly-connected thrust neurons is no larger than biological's, again at the coarse n=3 resolution and with one seed's ratio dominated by a near-zero denominator, not a real effect. This study offers no positive account of why P gains nothing under trained readouts either.

What this study does establish: the readout-input ablation results (a causal claim about a readout's own reliance on its inputs, not merely correlational) are real, measured effects on each readout's own held-out score, reported per readout above and in the JSON artifact -- these are the one part of this study's evidence that supports causal language, and even they speak only to each readout's reliance on its own inputs, not to why training converges the two topologies' scores.

## Limitations

- Trainer-seed variance is large and every hypothesis rule is evaluated per seed; the seed count (3) is disclosed throughout.
- Readouts are the only trained parameters -- the dynamics and topology are fixed and authored/measured, not trained.
- An analytic saliency (the exact chain-rule gradient of the readout's own forward pass) is not a causal claim about the circuit; the readout-input ablation results are included specifically because they, unlike saliency, do support causal language about a readout's own reliance on its inputs.
- Cell-type names (`descending-types-v1.json`) are annotation labels copied from the pinned MaleCNS table, never a claim about fly descending-neuron function.
- Three hypotheses (H1, H2, H3) are evaluated with no multiple-comparison correction; this is disclosed, not adjusted for.
- H2 and H3 use n = 3 paired trainer seeds and predeclared equivalence bounds (+/-0.10 independence-share difference for H2, <= 1.25 mean saliency ratio for H3) -- a coarse resolution, stated in `00-overview.md` itself, and confirmed coarse in practice by this study's own straddling CIs.
- H1 is stated only in correlational language ("consistent with"), never as a causal claim.
- The linkage cluster bootstrap resamples 28 clusters (20 bilateral pairs, 8 singletons), not the 48 neurons independently; a neuron-level bootstrap is reported alongside as a sensitivity check, and H1 is inconclusive whenever the two disagree on excluding 0.
- The 90% CIs for H2/H3 use a one-sample t interval (df = 2) rather than a percentile bootstrap: at n = 3, a percentile bootstrap of the mean cannot extend beyond the sample's own min/max (only 10 distinct resample multisets exist), which would understate real sampling uncertainty and make a "supported" verdict artificially easy to reach. The t-interval is the conventional TOST construction for a small paired sample and is used here instead -- a deliberate deviation from a percentile-bootstrap default, disclosed here.
- Coverage is scoped to the 23 default-task archive entries; the 52 per-task entries (`kind: "task-intervention"`, WP1b) are excluded because none of this study's predeclared analyses or hypothesis rules were ever defined over them (see Coverage above).
- This model only: no claim is made about the real fly's descending-neuron function or the biological plausibility of CEM training.
