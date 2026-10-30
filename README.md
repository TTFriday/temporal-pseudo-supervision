# Temporal pseudo-supervision for leaf instance segmentation

This repository contains the training, selection, propagation and evaluation code,
together with the archived experiment records, for a study of low-label leaf
instance segmentation in time-lapse plant imagery. The central idea is to spend a
small human annotation budget on a few well-chosen time points per plant and then
propagate masks to neighbouring frames with SAM2, using either a simple confidence
rule or a reliability check based on IoU agreement between forward and backward
propagation passes.

Six method configurations are compared:

| ID  | Selection             | Supervision beyond the human labels    |
| --- | --------------------- | -------------------------------------- |
| M1  | time coverage         | none                                   |
| M2  | per-frame uncertainty | spatial self-training                  |
| M3  | time coverage         | confidence-based temporal propagation  |
| M4  | event aware           | reliability-based temporal propagation |
| M5  | time coverage         | reliability-based temporal propagation |
| M6  | event aware           | confidence-based temporal propagation  |

M4 is the full method; M5 and M6 isolate the contribution of event-aware
selection and of the reliability check, respectively.

## Repository layout

```
run_pilot.py                  pilot runner: 6 methods x seeds at a given label budget
evaluate_checkpoints.py       COCO evaluation of trained student checkpoints
training/                     Mask R-CNN student, partial supervision, pipeline
selection/event_selector.py   time-coverage, uncertainty and event-aware query strategies
propagation/sam2_adapter.py   SAM2-based bidirectional mask propagation
evaluation/coco_eval.py       AP/AP50/AP75/SBD/count-MAE evaluation
analysis/                     gate summaries, formal budget sweep, random control
controls/run_random_query.py  random-query control runner (R1/R2 at 10% budget)
data_audit/                   dataset audit tools and the archived audit verdicts
splits/                       split construction and the archived plant-disjoint plans
configs/                      run configurations and the SAM2.1 model config
results/                      archived protocol/provenance/gate/result files per group
docs/                         budget-curve report, sweep record, pre-registrations
tests/                        regression tests (pytest)
```

## Dataset

Experiments use the PhenoLeaf-TS dataset (ECCV 2026; DOI:
10.1007/978-3-032-37359-5_23). Images and masks are **not** redistributed here.
Place the dataset under `data/phenoleaf_ts/` with this layout:

```
data/phenoleaf_ts/extracted/PhenoLeaf-TS_DS/01_raw_dataset/   RGB frames
data/phenoleaf_ts/extracted/PhenoLeaf-TS_DS/02_leaf_labels/   instance masks
data/phenoleaf_ts/PhenoLeaf-TS-main/phenoleaf_ts/             dataset toolkit
```

Then regenerate the frame manifest and the consistency audit with
`python data_audit/audit_dataset.py`; `data_audit/audit_report.json` archives
the verdicts of the original audit runs. The plant-disjoint development and
confirmation plans are checked in under `splits/` and can be regenerated with
`python splits/create_splits.py`.

## Environment

- Python 3.10; experiments ran on one RTX 3090 (24 GB).

- PyTorch 2.6.0 / torchvision 0.21.0 (CUDA 12.4 builds) and the pinned
  dependencies listed in `requirements.txt`.

- `sam2` 1.1.0, the pip packaging of Meta's Segment Anything Model 2
  (Apache-2.0), with the official SAM2.1 tiny checkpoint:

  ```
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
  SHA256: 7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
  ```

  Place the checkpoint in `checkpoints/`.

All path references in the scripts, run configurations and archived records are
relative to the repository root: the dataset under `data/phenoleaf_ts/`, the
SAM2.1 checkpoint under `checkpoints/`, and the archived experiment groups
under `results/`. Run the commands from the repository root.

## Running

```bash
pip install -r requirements.txt
pytest                                              # regression tests
python run_pilot.py --output_dir pilot_results --seeds 17 42
python analysis/confirmation_gate.py --results <run_directory>
python evaluate_checkpoints.py
```

## Archived results

`results/` keeps the protocol, provenance, gate and result files of every
experiment group in the directory layout used during the runs:

- `pilot_validated_20261025_v3/`, `pilot_validated_20261026_v4/` — 10% budget on
  the development split, seeds 17/42 and then the pre-registered extension to
  seeds 73/101/202/303 (`SEED_EXPANSION.md` records why the extension happened);
- `pilot_budget05/`, `pilot_budget20/`, `pilot_full100/` — formal budget sweep at
  5%, 20% and 100%;
- `random_query_control/` — 10% budget with random frame selection, R1/R2 x 3 seeds;
- `pilot_confirmation_20261027/` — confirmation-split replication, 6 seeds;
- `evidence.json`, `all_runs.csv`, `paired_comparisons.csv` — consolidated
  per-run metrics and paired comparisons over all 105 runs.

Headline numbers: on the development split, confidence-based propagation with
time-coverage selection (M3) improves AP over human-labels-only training (M1) by
1.19 points at the 10% budget (6 paired seeds, 6/6 positive); the confirmation
split replicates the improvement (+1.53 points, 6/6 positive). The AP gains do
not transfer to fixed-threshold SBD or counting metrics, which the study
discusses. Pre-registration documents are under `docs/` and inside
`results/pilot_confirmation_20261027/`.

##
