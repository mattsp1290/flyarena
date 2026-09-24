# Execution and merge readiness

Implementation is complete; current validation is recorded in docs/counterfactual-validation.md. Independent code reviews and the live Cursor thermonuclear review passed after fixes. Final merge follows the gates below.

## Ordered packages
1. Complete two independent plan reviews and one fresh adversarial review. Reconcile findings in these files. Preserve the user's application context and scope. Then freeze the plan as ready. Main ignores `.agents/plans`; force-add only this plan directory, without changing the ignore rule or including another agent's plans.
2. Implement `01-integration.md`: selective import, entry shell, scoped styles, additive offline packaging. Run check/unit/build, existing arena e2e and imported backend tests. Do not modify another worktree or any deployment environment.
3. Implement `02-causal-engine.md`: target contract, deterministic fork engine, controls, evidence and CLI. Run focused invariants, existing evaluator/product parity tests and real graph CLI regeneration.
4. Implement `03-browser.md`: Worker/client lifecycle, controls, paired replay and export; complete both direct navigation and the arena setup handoff. Run real browser journeys without backend and with temporary CUDA sandbox.
5. Reconcile current main, review implementation, verify all gates, record evidence and commit only related changes. Run the live Cursor thermonuclear review, fix its material findings and merge all our related work to main. The user explicitly requested this after planning; deployment via scripts/deploy.sh is allowed only afterward. Preserve unrelated dirty files and coordinate with current main state before the merge.

Keep implementation sequential where files overlap. No parallel implementation is required. Independent reviewer subagents are required by the invoked implementation-plan skill.

## Merge gate
- Inspect `git status`, latest main log and Beans plan execution read-only before final reconciliation. Other agents may land trained toggle, null distribution, anatomy or deployment changes. Preserve their public contracts and new user journeys.
- Incorporate latest main into this integration branch; do not checkout or mutate the occupied main worktree. Resolve additive shell/App/docs/script conflicts against current code and rerun affected suites. Do not overwrite current main with the old lab's App/CSS/deployment script.
- `npm ci`, `npm run check`, `npm run test:unit`, `npm run build`, and `CI=1 npm run test:e2e` pass with Node 22.22.3. Do not use `--pass-with-no-tests` or accept skipped counterfactual coverage.
- `env -u PYTHONPATH DD_IAST_ENABLED=false uv run --locked pytest tests_python` passes. Run `training/` pytest similarly using its own environment; do not reconfigure its runtime or stop a concurrent production training job. If optional training tools cannot run, the merge gate remains incomplete and reports the precise missing evidence.
- Backend tests, rebuilt image, actual CUDA experiment and synthetic browser journey pass via an isolated local container. No deployed container is restarted. Stop every newly created server/container.
- `bash -n scripts/deploy.sh scripts/lab.sh`; offline backend bundle content check; static production build served under `/fly/` exercises all routes and base-relative graph loading. Do not call deploy mode before the reviewed merge to main.
- Run independent code review focused on scientific invariants and lifecycle/integration; fix material findings and rerun relevant checks. Perform a mergeability check against the current local main SHA and record that SHA with commands, test counts, browser/hardware results and limitations in new `docs/counterfactual-validation.md` under existing docs.
- If main moves after verification, the recorded SHA explicitly bounds the claim. Reconcile and rerun gates for changed surfaces before declaring readiness. Do not claim green remote CI or an actual merge without evidence.

## Definition of done
One app retains main's default connectome arena, adds an actual graph-based fork/silence/compare experiment with reproducible evidence, and makes the existing GPU synthetic sandbox accessible without weakening provenance or requiring a server for the static modes. All plan success criteria and merge gates have authoritative current evidence. All our work is merged to main after the live Cursor thermonuclear review and fixes. The resulting worktrees are clean. No Beans mutation has occurred. Deployment is permitted only after merging and is reported separately with actual verification evidence.

Deferred: trained-decoder causal runs once its public browser contract lands, anatomical target picking, saved live-arena checkpoints, arbitrary neuron sets, large offline intervention atlases and running main's real graph on the lab server. They are not substitutes for the required working causal experiment and integration in this plan.

## User-required final review
Read and apply the live rubric at https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md to the complete branch diff. Record the retrieved content hash, review findings, corrections and relevant rerun evidence before merging. This is required in addition to scientific/lifecycle verification.
