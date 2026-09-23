# Model ledger vocabulary

FlyArena separates source evidence from engineering choices. These labels must remain visible and unambiguous in the product and its documentation.

| Surface | Required label | Meaning |
| --- | --- | --- |
| Graph topology | **Measured** | Connections and contact counts come from the pinned source dataset. |
| Biological annotations | **Annotated** | Cell classes, transmitter assignments, and similar metadata are source annotations rather than direct measurements of simulated behavior. |
| Network dynamics | **Authored / literature-derived** | The rate equations and update rules are engineering choices informed by literature; they are not measured dynamics from the source animal. |
| Global parameters | **Calibrated** | Global gains or thresholds may be adjusted under a documented, shared calibration procedure. |
| Sensory encoder and action decoder | **Authored** | Input/output mappings are designed for this POC and must remain identical across experimental arms. |
| 3D presentation | **Synthetic** | Arena geometry, agents, effects, and camera are visual presentation, not biological anatomy or biomechanics. |

The application must not describe the POC as a brain emulation or imply that authored behavior is biological. `src/lib/ui/LedgerPanel.svelte` renders this table in-product next to the experiment's live results, plus links to the compiled artifact's manifest and ledger JSON (`public/data/malecns-arena-v1.manifest.json`/`.ledger.json` — exact dataset version, source-file hashes, compiler-revision hash, node/edge/contact counts, inclusion rules, and retained/dropped counts) and the CC BY 4.0 attribution for the source dataset (Janelia FlyEM Project (HHMI), MRC Laboratory of Molecular Biology, Google Research). See `docs/data-provenance.md` for the full sourcing detail those links point at.

### Topology labeling vs. the renderer's fixed arm identity

The 3D renderer (`src/lib/render/ArenaScene.ts`) gives the left agent a fixed "BIO" shape/label (a smooth icosahedron) and the right agent a fixed "REWIRED" shape/label (an angular octahedron with a wireframe overlay) — this is a stable *visual identifier for an arena position*, not a live claim about which topology that arm is currently running. `ExperimentPanel.svelte`'s topology selectors let either arm run biological, rewired, or disconnected; `TelemetryPanel.svelte` always shows each arm's actual selected topology as text next to its "(BIO shape)"/"(REWIRED shape)" label, so a user can never mistake a re-topologized arm for its default assignment.
