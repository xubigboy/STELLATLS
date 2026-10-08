# STELLA-TLS public algorithm code

This package exposes the public three-stage implementation described by
STELLA-TLS: Spatial Tri-stage Explainable Learning for Lymphoid Architecture of
Tertiary Lymphoid Structures. The code is self-contained and does not read the
locked H-drive checkpoints or datasets.

## Algorithm names

The public names follow the manuscript workflow:

- `RAPIDTLS` (`RAPID-TLS`): TLS instance segmentation with primary and
  auxiliary masks, boundary and center maps, objectness, and box regression.
- `GATETB` (`GATE-TB`): gated tumor-bed semantic segmentation with an auxiliary
  boundary head.
- `ORDMoE` (`ORD-MoE`): dual-view maturity classifier with morphology experts,
  categorical prediction, and cumulative ordinal prediction for E-TLS, P-TLS,
  and S-TLS.
- `STELLATLS` (`STELLA-TLS`): wrapper preserving the Module B to Module C
  order. Module A and Module D are implemented by `src/spatial.py` utilities.

The previous implementation names remain available for compatibility:
`TLSInstanceResearchNet`, `TumorBedBoundaryResearchNet`, and
`MaturityContextOrdinalResearchNet`.

## Workflow alignment

Module A uses deterministic tissue-qualified, coverage-preserving candidate
tiles. `generate_candidate_tiles` accepts a low-resolution tissue mask and
retains each tile's origin, scale, and tissue coverage. `map_patch_point_to_wsi`
maps local coordinates back to the WSI frame using `x_wsi=x0+scale*x` and
`y_wsi=y0+scale*y`.

Module B runs RAPID-TLS and GATE-TB in parallel on the same coordinate-preserving
tile grid. `apply_tls_cascade` performs TTA aggregation, uncertainty-aware
thresholding, objectness gating, connected-component separation, and area
filtering.

Module C sends the same TLS crop and its Gaussian-blurred second view to
ORD-MoE. The three maturity states are ordered as E-TLS, P-TLS, and S-TLS;
`ordinal_prediction` and `select_ordinal_thresholds` implement the cumulative
ordinal decision path.

Module D uses `assign_spatial_compartment` for the 500-μm boundary rule,
`build_descriptive_tls_graph` for a non-trainable coordinate-aware kNN graph,
and `patient_level_endpoints` for TLS count, density, mature ratio, spatial
entropy, and SM-TLS score. The graph utility is descriptive and does not imply
a trainable graph neural network.

`UnifiedTLSResearchModel` remains available as an optional cross-stage research
extension with hierarchical token compression, pathology-aware MoE, coordinate
graph attention, and transformer refinement. It is not required for the
paper-aligned STELLA-TLS three-stage wrapper.

## Install

```bash
pip install -r requirements.txt
```

## Data format

Use one `.npz` file per task.

```text
TLS:       image, mask, optional objectness, optional box
Tumor-bed: image, mask
Maturity:  local, context, label
```

Images may be `NCHW` or `NHWC`. Numeric maturity labels are `0=E-TLS`,
`1=P-TLS`, and `2=S-TLS`.

## Train

```bash
python train.py --task tls --data tls_train.npz --out checkpoints/rapid_tls.pt --epochs 1
python train.py --task tumor_bed --data tumor_bed_train.npz --out checkpoints/gate_tb.pt --epochs 1
python train.py --task maturity --data maturity_train.npz --out checkpoints/ord_moe.pt --epochs 1
```

The driver trains RAPID-TLS, GATE-TB, and ORD-MoE by default. Use
`--variant compact` only for the small compatibility models in `src/models.py`.
Use `--device cuda` when CUDA is available.

The locked production TLS baseline remains `YOLO11n-seg`; it is only a
comparison baseline. The new STELLA-TLS models do not call YOLO or depend on
`ultralytics`.

## Public module map

```text
src/models.py        compact compatibility models
src/research.py      RAPID-TLS, GATE-TB, ORD-MoE and research losses
src/spatial.py       Module A tile mapping and Module D spatial endpoints
src/ops.py           segmentation and maturity inference post-processing
src/research_ops.py  mining, TTA, ordinal and slide calibration utilities
train.py             NPZ training driver
source_map.json      mapping from the six internal scripts to public modules
```

