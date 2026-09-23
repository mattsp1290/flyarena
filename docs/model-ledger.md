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

The application must not describe the POC as a brain emulation or imply that authored behavior is biological. The eventual ledger will add exact dataset versions, hashes, inclusion rules, compiler revision, retained and dropped counts, license, and citations.
