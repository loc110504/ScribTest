# UNet3D + pCE baseline

This directory intentionally contains one training implementation for ACDC,
MSCMR and WORD. It follows fixed published splits rather than sampling a new
holdout at runtime, so its results can be compared to prior scribble-supervised
work.

The optimizer sees only `labelsTr`. In those masks, the dataset's number of
classes (4 for ACDC/MSCMR, 17 for WORD) is the unlabeled/ignore value. Partial
cross-entropy is the ordinary multiclass cross-entropy averaged only over voxels
whose value is not the ignore value. `labelsTr_dense` is used only to calculate
Dice on the published validation partition and select `best.pth`.
The official test split is untouched during training.

| Dataset | Train | Validation | Test | Fixed protocol |
| --- | ---: | ---: | ---: | --- |
| ACDC | 70 patients / 140 phases | 15 patients / 30 phases | 15 patients / 30 phases | ScribFormer `MAAGfold70` |
| MSCMR | 23 volumes | 5 volumes | 15 volumes | CycleMix; subject2 and subject4 omitted because no dense GT exists in ScribbleBench |
| WORD | 100 scans | 20 scans | 30 scans | original WORD-V0.1.0 `imagesTr` / `imagesVal` / `imagesTs` |

The exact group IDs are in `legacy_splits.py` and each run writes resolved IDs
to `split.json`. Training aborts if the cases on disk are not exactly the
published train+validation partition.

From the repository root, train one dataset:

```bash
python code/train/train_pce_3d.py --dataset ACDC --amp
python code/train/train_pce_3d.py --dataset MSCMR --amp
python code/train/train_pce_3d.py --dataset WORD --amp
```

Or run all three sequentially:

```bash
bash code/train/run.sh --amp
```

Set `SCRIBBLE_PCE_OUTPUT_ROOT` to change the common output root used by
`run.sh`; each dataset always receives its own subdirectory.

Outputs default to `checkpoints/ScribbleBench_pCE/<DATASET>/`:

- `best.pth`: highest dense-validation mean foreground Dice; use this for test.
- `last.pth`: latest resumable optimizer/model/scaler state.
- `split.json`: exact patient-level train/validation split.
- `validation.jsonl`, `train.log`, and `tensorboard/`: metrics and logs.

Test the best checkpoint with the existing 3D evaluator:

```bash
python code/test/test_pce_3d.py \
  --checkpoint checkpoints/ScribbleBench_pCE/ACDC/best.pth \
  --amp --save_predictions
```

Replace `ACDC` in the checkpoint path for MSCMR or WORD. The evaluator reads
the dataset, class count, patch size and feature channels directly from the
checkpoint and loads weights strictly.

Resume an interrupted run with the same dataset, published split, architecture
and output directory:

```bash
python code/train/train_pce_3d.py \
  --dataset ACDC \
  --resume checkpoints/ScribbleBench_pCE/ACDC/last.pth \
  --amp
```

The implementation uses the standard partial-cross-entropy formulation: only
scribble voxels affect the loss. It remains a plain UNet3D+pCE baseline, not a
reproduction of pseudo-label, consistency, or multi-branch methods.

## References used

- [CycleMix paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_CycleMix_A_Holistic_Strategy_for_Medical_Image_Segmentation_From_Scribble_CVPR_2022_paper.pdf)
  and [official repository](https://github.com/BWGZK/CycleMix): MSCMR public
  split, and the conventional 70/15/15 ACDC protocol.
- [ScribFormer repository](https://github.com/HUANGLIZI/ScribFormer/blob/main/acdc/dataset.py):
  exact `MAAGfold70` ACDC subject IDs.
- [DMSPS paper](https://www.sciencedirect.com/science/article/pii/S1361841524001993)
  and [official repository](https://github.com/HiLab-git/DMSPS): 100/20/30 WORD
  and 70/15/15 ACDC experiments.
- [Official WORD repository](https://github.com/HiLab-git/WORD): source archive
  whose `imagesTr`, `imagesVal`, and `imagesTs` memberships are retained here.

## UNet3D + VoxTrust-3D

Implements `VoxTrust3D_CVPR2026_Proposed_Method.pdf`: a single 3D U-Net
student + EMA teacher (Mean Teacher), where the mechanism that decides
training-time behavior is not the network but *which* teacher pseudo-labels
the student is allowed to learn from. See `code/utils/voxtrust3d.py`'s module
docstring for the algorithm and `train_voxtrust3d_3d.py`'s module docstring
for the engineering choices made where the paper leaves an implementation
detail open (scribble "block" definition, KD-tree-based physical transfer
distance instead of dense per-volume distance maps, coordinate tracking
through crop/flip/rotate augmentation).

Training uses `labelsTr`, split once at the start of a run into a directly
supervised part and a held-out calibration part (`--holdout_fraction`,
default 0.15); the calibration part is never used for gradient-based
supervision, only to test whether the reliability score predicts teacher
correctness. `labelsTr_dense` is used only for checkpoint selection, exactly
as in the pCE baseline above.

```bash
python code/train/train_voxtrust3d_3d.py --dataset ACDC --amp
python code/train/train_voxtrust3d_3d.py --dataset MSCMR --amp
python code/train/train_voxtrust3d_3d.py --dataset WORD --amp
```

Outputs default to `checkpoints/ScribbleBench_VoxTrust3D/<DATASET>/`, with the
same `best.pth` / `last.pth` / `split.json` / `validation.jsonl` / `train.log`
/ `tensorboard/` layout as the other 3D scripts. Per the paper's Sec. 5
("only one EMA network is required at inference; the calibrator is
removed"), the checkpointed `model_state_dict` is the **EMA teacher's**
weights in the same schema the other baselines use, so it needs **no
separate test script** -- evaluate it exactly like SDT-Net or CycleMix:

```bash
python code/test/test_pce_3d.py \
  --checkpoint checkpoints/ScribbleBench_VoxTrust3D/ACDC/best.pth \
  --amp --save_predictions
```

Resume with `--resume checkpoints/ScribbleBench_VoxTrust3D/ACDC/last.pth`
(student weights, teacher weights, optimizer, and the rolling calibration
memory are all restored; the distance-stratum bin edges/`d_c^max` table is
cheap and deterministic, so it is recomputed rather than persisted).

Recommended starting hyperparameters (paper Sec. 5) are the argparse
defaults: EMA decay `--ema_decay 0.99`, holdout fraction `--holdout_fraction
0.15`, `--distance_strata 3`, target precision `--target_precision 0.95`,
Wilson `--wilson_delta 0.05`, `--calibration_min_samples 32`,
`--calibration_buffer_size 4096`, `--calibration_block_cap 64`, and a
scribble-only warm-up (`--warmup_frac 0.1`) followed by a sigmoid pseudo-loss
ramp-up (`--rampup_frac 0.2`) so the full ramp completes by ~30% of training.
