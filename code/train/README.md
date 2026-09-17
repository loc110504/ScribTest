# UNet2D/VNet3D + pCE baseline

This directory contains one training implementation per method, split into a
2D slice pipeline for ACDC/MSCMR (`train_<method>_2d.py`, `UNet2D`/`UNetCCT2D`
backbone) and a full-3D pipeline for WORD (`train_<method>_3d.py`,
`VNet3D`/`VNetCCT3D` backbone). ACDC and MSCMR are cardiac MRI with strongly
anisotropic voxel spacing (thin in-plane, thick through-plane), so they train
as independent 2D slices, stitched back into a volume only at
validation/test time (`train/common_2d.py`'s `predict_volume_2d`) -- the
standard protocol across the scribble-supervision literature (WSL4MIS, DMSPS,
CycleMix, ScribFormer). WORD (abdominal CT, near-isotropic) stays a full-3D
VNet pipeline. Both follow fixed published splits rather than sampling a new
holdout at runtime, so results can be compared to prior scribble-supervised
work.

The optimizer sees only `labelsTr`. In those masks, the dataset's number of
classes (4 for ACDC/MSCMR, 8 for WORD's 7-organ subset) is the unlabeled/ignore
value. Partial cross-entropy is the ordinary multiclass cross-entropy averaged
only over voxels whose value is not the ignore value
(`train/common_3d.py::partial_cross_entropy`, shape-agnostic over 2D/3D).
`labelsTr_dense` is used only to calculate Dice on the published validation
partition and select `best.pth`. The official test split is untouched during
training.

| Dataset | Train | Validation | Test | Fixed protocol |
| --- | ---: | ---: | ---: | --- |
| ACDC | 70 patients / 140 phases | 15 patients / 30 phases | 15 patients / 30 phases | ScribFormer `MAAGfold70` |
| MSCMR | 23 volumes | 5 volumes | 15 volumes | CycleMix; subject2 and subject4 omitted because no dense GT exists in ScribbleBench |
| WORD | 100 scans | 20 scans | 30 scans | original WORD-V0.1.0 `imagesTr` / `imagesVal` / `imagesTs` |

The exact group IDs are in `legacy_splits.py` (patient-level, shared by both
pipelines) and each run writes resolved IDs to `split.json`. Training aborts
if the cases on disk are not exactly the published train+validation
partition. For ACDC/MSCMR, this case-level split is then translated into flat
per-slice training indices via
`dataloader.scribblebench_2d.ScribbleBench2DDataset.slice_positions_for_volumes`
(see `train_pce_2d.py`'s `resolve_case_split`).

From the repository root, train one dataset:

```bash
python code/train/train_pce_2d.py --dataset ACDC --amp
python code/train/train_pce_2d.py --dataset MSCMR --amp
python code/train/train_pce_3d.py --dataset WORD --amp
```

Or run pCE on all three sequentially:

```bash
bash code/train/run.sh --amp
```

Set `SCRIBBLE_PCE_OUTPUT_ROOT` to change the common output root used by
`run.sh`; each dataset always receives its own subdirectory. Note `run.sh`
forwards `"$@"` to whichever script a dataset uses, so flags that differ in
shape between the two pipelines (`--patch_size` takes two values for 2D,
three for 3D) must be passed per-dataset instead of through that wrapper.

Outputs default to `checkpoints/ScribbleBench_pCE/<DATASET>/`:

- `best.pth`: highest dense-validation mean foreground Dice; use this for test.
- `last.pth`: latest resumable optimizer/model/scaler state.
- `split.json`: exact patient-level train/validation split.
- `validation.jsonl`, `train.log`, and `tensorboard/`: metrics and logs.

Test the best checkpoint with the matching evaluator:

```bash
# ACDC/MSCMR
python code/test/test_pce_2d.py \
  --checkpoint checkpoints/ScribbleBench_pCE/ACDC/best.pth \
  --amp --save_predictions

# WORD
python code/test/test_pce_3d.py \
  --checkpoint checkpoints/ScribbleBench_pCE/WORD/best.pth \
  --amp --save_predictions
```

Replace `ACDC` in the checkpoint path for MSCMR. The evaluator reads the
dataset, class count, patch size and backbone width directly from the
checkpoint and loads weights strictly.

Resume an interrupted run with the same dataset, published split, architecture
and output directory:

```bash
python code/train/train_pce_2d.py \
  --dataset ACDC \
  --resume checkpoints/ScribbleBench_pCE/ACDC/last.pth \
  --amp
```

The implementation uses the standard partial-cross-entropy formulation: only
scribble voxels affect the loss. It remains a plain pCE baseline, not a
reproduction of pseudo-label, consistency, or multi-branch methods.

## References used

- [WSL4MIS repository](https://github.com/HiLab-git/WSL4MIS): the 2D `UNet`
  architecture, `RandomGenerator` slice augmentation, and per-slice
  resize-then-stitch validation (`val_2D.py`) that
  `networks/unet_2d.py`/`dataloader/scribblebench_2d.py`/`train/common_2d.py`
  are ported from.
- [CycleMix paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_CycleMix_A_Holistic_Strategy_for_Medical_Image_Segmentation_From_Scribble_CVPR_2022_paper.pdf)
  and [official repository](https://github.com/BWGZK/CycleMix): MSCMR public
  split, and the conventional 70/15/15 ACDC protocol.
- [ScribFormer repository](https://github.com/HUANGLIZI/ScribFormer/blob/main/acdc/dataset.py):
  exact `MAAGfold70` ACDC subject IDs.
- [DMSPS paper](https://www.sciencedirect.com/science/article/pii/S1361841524001993)
  and [official repository](https://github.com/HiLab-git/DMSPS): 100/20/30 WORD
  and 70/15/15 ACDC experiments, and the 2D `RandomGenerator`/VNet source
  `networks/vnet_3d.py` and `dataloader/scribblebench_2d.py::RandomGenerator2D`
  are verified against.
- [Official WORD repository](https://github.com/HiLab-git/WORD): source archive
  whose `imagesTr`, `imagesVal`, and `imagesTs` memberships are retained here.
- [EFFDNet paper](https://link.springer.com/chapter/10.1007/978-3-032-05334-9)
  and [official repository](https://github.com/Aurora-003-web/EFFDNet):
  `code/utils/effdnet.py`'s Mean-Teacher EMA warm-up, FBSL contrastive
  formula, and FADC copy-paste augmentation are verified against
  `code/train_weakly_supervised_2D.py` and `code/utils/supcon_loss.py`.
- [ModelMix paper](https://arxiv.org/abs/2406.13237) and
  [official repository](https://github.com/BWGZK/ModelMix):
  `code/utils/modelmix.py`'s image/model mixup, encoder-layer selection, and
  the model-mixup branch's detached-virtual-encoder gradient behavior are
  verified against `multi_train.py`.

## DMSPS (2-stage)

`train_dmsps_2d.py`/`train_dmsps_3d.py` reuse `dmsps_step` from the 3D script
in both pipelines (dual-branch pCE + dynamically mixed soft pseudo-label
consistency is elementwise/softmax math with no 3D-specific assumption -- see
`utils/dmsps.py`). Stage 1 trains the dual-decoder DB-Net (`UNetCCT2D` /
`VNetCCT3D`) directly on the sparse scribble; stage 2 re-initializes from
stage 1 and retrains after expanding the scribble with high-confidence,
largest-connected-component predictions from stage 1
(`--stage 2 --init_checkpoint <stage1 best.pth>`). For the 2D pipeline this
expansion still runs *per case* (its whole slice stack, 26-connectivity
across slices, matching the 3D pipeline exactly) via
`utils.dmsps.dual_branch_slice_stack_probs`, which forwards each slice
through the 2D model instead of a 3D sliding window.

```bash
python code/train/train_dmsps_2d.py --dataset ACDC --stage 1 --amp
python code/train/train_dmsps_2d.py --dataset ACDC --stage 2 --amp \
  --init_checkpoint checkpoints/ScribbleBench_DMSPS/ACDC/stage1/best.pth

python code/train/train_dmsps_3d.py --dataset WORD --stage 1 --amp
python code/train/train_dmsps_3d.py --dataset WORD --stage 2 --amp \
  --init_checkpoint checkpoints/ScribbleBench_DMSPS/WORD/stage1/best.pth
```

Test with `test_dmsps_2d.py` (ACDC/MSCMR) or `test_dmsps_3d.py` (WORD), never
`test_pce_*.py` -- the checkpoint holds a dual-decoder network, not a plain
one.

## SDT-Net

`train_sdtnet_2d.py`/`train_sdtnet_3d.py` reuse `sdtnet_step` from the 3D
script: Dynamic Teacher Switching, Pick Reliable Pixels, and Hierarchical
Consistency are elementwise/feature-flatten operations with no 3D-specific
assumption (`utils/sdtnet.py`). `UNet2D`/`VNet3D` both support the
`return_features=True` contract HiCo needs (`{"encoder": [...], "decoder":
[...]}`, `decoder[0]` the coarsest stage, `decoder[-1]` full resolution).

```bash
python code/train/train_sdtnet_2d.py --dataset ACDC --amp
python code/train/train_sdtnet_3d.py --dataset WORD --amp
```

## UNet2D/VNet3D + VoxTrust-3D

Implements `VoxTrust3D_CVPR2026_Proposed_Method.pdf`: a single student + EMA
teacher (Mean Teacher), where the mechanism that decides training-time
behavior is not the network but *which* teacher pseudo-labels the student is
allowed to learn from. See `code/utils/voxtrust3d.py`'s module docstring for
the algorithm, and the `train_voxtrust3d_2d.py`/`train_voxtrust3d_3d.py`
module docstrings for the engineering choices made where the paper leaves an
implementation detail open (scribble "block" definition, KD-tree-based
physical transfer distance instead of dense per-volume distance maps,
coordinate tracking through augmentation). `voxtrust_step`
(`train_voxtrust3d_3d.py`) is reused unchanged by the 2D script -- every
calculation it performs is shape-agnostic; only the student's strong-view
augmentation differs (`strong_intensity_augment_2d` vs. `_3d`).

The 3D pipeline crops a fixed-size patch out of each (much larger) WORD
volume, so Omega_sup/Omega_cal calibration is partitioned once **per case**
and unlabeled-voxel coordinates are tracked through that crop
(`choose_patch_origin`/`build_patch_coordinates`/`gather_patch`). ACDC/MSCMR
slices are already small enough that training uses the *whole* native slice
resized to `patch_size` -- there is no cropping, so calibration is instead
partitioned once **per slice** (matching the paper's own block definition,
"all scribble voxels of one class on one annotated slice"), the physical
transfer distance is an in-plane 2D distance (dropping the through-plane
spacing component), and coordinate tracking goes through
`random_flip_rotate_resize_2d` (the 2D counterpart of the 3D crop/gather
machinery) instead.

Training uses `labelsTr`, split once at the start of a run into a directly
supervised part and a held-out calibration part (`--holdout_fraction`,
default 0.15); the calibration part is never used for gradient-based
supervision, only to test whether the reliability score predicts teacher
correctness. `labelsTr_dense` is used only for checkpoint selection, exactly
as in the pCE baseline above.

```bash
python code/train/train_voxtrust3d_2d.py --dataset ACDC --amp
python code/train/train_voxtrust3d_2d.py --dataset MSCMR --amp
python code/train/train_voxtrust3d_3d.py --dataset WORD --amp
```

Outputs default to `checkpoints/ScribbleBench_VoxTrust3D/<DATASET>/`, with the
same `best.pth` / `last.pth` / `split.json` / `validation.jsonl` / `train.log`
/ `tensorboard/` layout as the other scripts. Per the paper's Sec. 5
("only one EMA network is required at inference; the calibrator is
removed"), the checkpointed `model_state_dict` is the **EMA teacher's**
weights in the same schema the other baselines use, so it needs **no
separate test script** -- evaluate it exactly like SDT-Net or CycleMix:

```bash
python code/test/test_pce_2d.py \
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

## UNet2D/VNet3D + EFFDNet

Implements EFFDNet (Liu et al., MICCAI 2025), verified against the official
`Aurora-003-web/EFFDNet` repository. A Mean-Teacher framework (student +
an **independently initialized** EMA teacher -- the source calls its model
constructor twice with no state-dict copy, unlike SDT-Net/VoxTrust-3D above)
adds two losses on top of partial cross-entropy and a dense cross-entropy
against the teacher's (Gaussian-noise-perturbed) pseudo-label:

- **FBSL** (Foreground-Background Separation Loss): the student's last
  decoder feature map is aggregated into a coarse `K x K` (`K^3` in 3D) grid,
  each cell labeled foreground/background from whether it contains any
  foreground-class scribble pixel, and a modified SupCon-style contrastive
  loss pulls same-label cells together while separating the two groups. See
  `code/utils/effdnet.py` for the exact (verified-against-source) contrastive
  formula, which differs subtly from the vanilla SupCon paper's.
- **FADC** (Foreground Augmentation with Diverse Context): a batch-level
  copy-paste augmentation that swaps each sample's own annotated bounding-box
  region for a resized crop of another sample's own annotated region.

Both are shape-agnostic over 2D/3D (`code/utils/effdnet.py`), so
`train_effdnet_2d.py` reuses `effdnet_step` from `train_effdnet_3d.py`
unchanged, exactly like the other baselines above.

```bash
python code/train/train_effdnet_2d.py --dataset ACDC --amp
python code/train/train_effdnet_2d.py --dataset MSCMR --amp
python code/train/train_effdnet_3d.py --dataset WORD --amp
```

Outputs default to `checkpoints/ScribbleBench_EFFDNet/<DATASET>/`, same
layout as the other scripts. The deployed/checkpointed model is the
**student** (matching the source, not the EMA teacher), so evaluate it
exactly like pCE/CycleMix:

```bash
python code/test/test_pce_2d.py \
  --checkpoint checkpoints/ScribbleBench_EFFDNet/ACDC/best.pth \
  --amp --save_predictions
```

Recommended starting hyperparameters (paper's Implementation Details) are the
argparse defaults: EMA decay `--ema_alpha 0.99` (with a Mean-Teacher warm-up
ramp on alpha, not fixed from step 0), pseudo-label weight `--lambda_value
0.6`, FBSL weight `--delta 0.3`, grid resolution `--num_regions 8`. FBSL and
FADC are both **on** by default (`--use_fbsl 1 --use_fadc 1`); pass `0` to
ablate either one.

## UNet2D + ModelMix

Implements ModelMix (Zhang & Patel, MICCAI 2024), verified against the
official `BWGZK/ModelMix` repository. Unlike every other method above,
ModelMix always trains a **pair** of tasks jointly -- two independently
initialized `UNet2D` models (same architecture, no shared weights), one per
task -- so `train_modelmix_2d.py` has **no `--dataset` flag**: one run always
produces both ACDC's and MSCMR's checkpoints, the only pair in this benchmark
sharing a compatible 4-class cardiac label space (matching the paper's own
primary experiment; see `code/utils/modelmix.py`'s module docstring for why
WORD has no legitimate substitute and is not covered).

Besides each task's own scribble supervision (partial CE + Dice, restricted
to annotated pixels), every iteration adds:

- **Image-level mixup**: each task's image is linearly blended with its own
  batch-reversed counterpart (`Beta(0.5, 0.5)` ratio); the mixed image's
  prediction is supervised against the correspondingly blended one-hot label
  and regularized (cosine similarity) to match the same blend of the two
  individual predictions. Verified against the source: it does *not*
  implement the paper text's cutout-then-mix description.
- **Model-level mixup** (the method's namesake): one randomly selected
  *encoder* convolutional layer's weight+bias are linearly blended between
  the two tasks' own encoders into a virtual encoder, paired with the task's
  own decoder and run on a randomly rotated copy of the same image;
  supervised on the (rotated) scribble label, and regularized (negative
  cosine similarity) so the de-rotated prediction matches that task's own
  individual (non-mixed) model.

The virtual encoder is built as a `torch.no_grad()` deep copy (matching the
source's raw `.data =` assignment), so backpropagating through it updates
only the decoder called on top of it, never either encoder's real weights
directly -- each encoder is still updated every step, just through the
*other* loss terms (own supervision, image-level mixup) that call it
directly. See `mix_one_encoder_layer`'s docstring in
`code/utils/modelmix.py` before treating this as a bug.

```bash
python code/train/train_modelmix_2d.py --amp
```

Outputs default to `checkpoints/ScribbleBench_ModelMix/{ACDC,MSCMR}/`, each
with the usual `best.pth` / `last.pth` / `split.json` / `validation.jsonl`
layout (no shared/joint optimizer state is persisted, since one optimizer
spans both tasks' parameters at once -- `--resume_acdc`/`--resume_mscmr`
restore model weights and step/best-score bookkeeping, not exact optimizer
momentum). Both checkpoints are plain `UNet2D` models, evaluated exactly like
pCE:

```bash
python code/test/test_pce_2d.py \
  --checkpoint checkpoints/ScribbleBench_ModelMix/ACDC/best.pth --amp
python code/test/test_pce_2d.py \
  --checkpoint checkpoints/ScribbleBench_ModelMix/MSCMR/best.pth --amp
```
