# Behavior atlas validation

Status: implementation and maintainability review complete; integrated verification
and deployment remain in progress.

## Planning and hardware

The implementation-plan workflow completed two independent reviews and a fresh
adversarial review. Accepted findings added atlas-hash pinning between selection
and worker execution, direct trained-counterfactual/episode parity, exact tested
deployment-checkout requirements, and actual CUDA repeat-search measurements.
The adversarial review returned no material findings. User context: no active
users, no backward-compatibility requirement, no feature flags. Deployment is
authorized only after review and merge.

Actual NVIDIA GB10 search: PyTorch 2.11.0+cu128, CUDA 12.8; 64 candidates × 24
generations × eight seeds × 900 ticks. Wall time was 135.734 seconds; peak PyTorch
tensor allocation was 27,753,984 bytes (26.47 MiB). TF32 was disabled. This is a
working-set measurement, not an end-to-end CPU/GPU speedup claim or whole-process
memory measurement. Other agents' processes were not stopped or altered.

The search produced 30 occupied cells. Canonical TypeScript evaluation retained
30, spanning all six coverage bins and all six turning bins. The artifact is
2,747,730 bytes. The manifest records its exact SHA-256 and byte length.

Two short fixture CUDA searches with identical settings produced identical results.
CPU/CUDA and singleton/batched fixture rollouts passed. A real-graph smoke comparison
had zero coverage error, maximum absolute score error 8.57e-9 and turning error
2.44e-8. The full 900-tick candidate comparison had maximum mean-score discrepancy
0.464213, coverage discrepancy 0.0025 and turning discrepancy 0.001359. This is
explicitly not an exact GPU/TypeScript trajectory-parity claim. Public values and
replays are recomputed canonically in TypeScript.

The default controller, 1316, was chosen by discovery quality, not held-out
performance. Its held-out mean score is 34.4718 intact, 15.5448 disconnected and
15.5448 with zero readout inputs. This is descriptive evidence about this model,
not biological inference or a claim that all behavior depends on the circuit.

## Completed checks

- Svelte/TypeScript: zero errors and warnings.
- TypeScript unit/integration suite: 55 files, 579 tests passed.
- Training suite: 128 tests passed, including actual CPU and CUDA paths.
- Selected-controller engine tests compare warmup and subsequent sampled worlds
  against the canonical episode runner for two published readouts; baseline/sham
  equality and bias-preserving all-rate silencing also pass.

- Full atlas verifier exactly reproduced all 30 controllers, discovery metrics,
  held-out controls and replay frames.
- Graph compiler: 66 tests passed.
- Browser atlas selection/replay/probe/cancel/retry, tampering, deployment-race
  rejection and exported-evidence comparison passed. Browser/Node comparison
  differed by at most 1.11e-16; the numerical diagnostic passed without loosening
  its existing tolerance. It is not labeled exact reproduction.
- A Node-produced 8-seed controller-1316 output-group probe (fork 120, horizon 300)
  reproduced exactly. The paired score effect was −8.6790 and sham effect zero;
  this is one exploratory example, not a biological significance claim.

## Live Cursor maintainability review

The requested upstream rubric was retrieved on 2026-09-24 from
[Cursor's live skill](https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md).
Its raw-file SHA-256 was `7faca08b51b643b2ddd0836f92af15574444024685dcc1e677dbbb39ae8c9e8f`.
The implementing agent reviewed the full change against this rubric; this code
review is distinct from the independent planning reviews above.

Accepted structural fixes:

1. **Important:** atlas and CEM rollouts duplicated the observe/model/readout/world
   sequence. Extracted `step_readout_world` into the existing rollout owner and
   made both consumers use it. Full training tests, including CUDA, passed again.
2. **Important:** flat weight decoding and snapshot capture risked independent
   contracts. Both now reuse the canonical readout conversion and counterfactual
   frame capture; read-only snapshot inputs are accepted without casts.
3. **Minor:** dense boundary code and ad-hoc UI key casts obscured the contract.
   Formatted the new TypeScript modules, used typed control names, and named the
   discriminated evidence header. Unknown JSON is validated at the boundary;
   controller identity is resolved against verified graph/atlas data before use.

The revised implementation has one causal fork engine, one worker lifecycle,
shared trained rollout stepping, bounded artifact loaders, and no new file near
1,000 lines. No material structural finding remains. Artifact/manifest publication
is offline; concurrent development fetches fail closed during a partial update.
Production activation switches the complete release atomically.

Still pending: final integrated browser/subpath gates, merge, deployment and the
public browser smoke.
