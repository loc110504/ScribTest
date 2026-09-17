# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Scribble-supervised medical image segmentation benchmark on **ACDC**, **MSCMR**, and **WORD**
(collectively "ScribbleBench"). ACDC and MSCMR (cardiac MRI, strongly anisotropic spacing) train as
independent **2D slices** (`UNet2D` backbone, the standard protocol in the scribble-supervision
literature — WSL4MIS, DMSPS, CycleMix, ScribFormer), stitched back into a volume only at
evaluation time. WORD (abdominal CT, near-isotropic) trains as full **3D volumes** with a `VNet3D`
backbone. Seven methods share these two pipelines, differing only in how they turn scribbles into a
loss / pseudo-label:

- **pCE** — partial cross-entropy baseline (loss only on annotated voxels)
- **CycleMix** (Zhang & Zhuang, CVPR 2022)
- **DMSPS** (Han et al., MedIA 2024) — two-stage training (dual-branch network, then re-init)
- **SDT-Net** (Nguyen et al. 2026) — dual-teacher/single-student
- **VoxTrust-3D** (this project's proposed method, `VoxTrust3D_CVPR2026_Proposed_Method.pdf`) —
  single student + EMA (Mean Teacher) where reliability calibration decides which pseudo-labels
  the student learns from; see `code/utils/voxtrust3d.py` and the `train_voxtrust3d_2d.py` /
  `train_voxtrust3d_3d.py` module docstrings for the algorithm and the implementation choices made
  where the paper leaves details open (block/KD-tree granularity differs between the 2D per-slice
  and 3D per-volume pipelines — see the 2D script's docstring)
- **EFFDNet** (Liu et al., MICCAI 2025) — Mean Teacher + a grid-based Foreground-Background
  Separation Loss (FBSL, a modified SupCon-style contrastive loss) and a Foreground Augmentation
  with Diverse Context (FADC) copy-paste augmentation; see `code/utils/effdnet.py`. Runs on all 3
  datasets like the methods above.
- **ModelMix** (Zhang & Patel, MICCAI 2024) — the one method that is *not* single-dataset: it always
  jointly trains a **pair** of tasks (separate encoder+decoder per task, same encoder architecture),
  periodically blending one random encoder layer between them and regularizing the blend to agree
  with each task's own individual model. ACDC+MSCMR are the only pair here sharing a compatible
  4-class cardiac label space (matching the paper's own primary experiment); WORD has no comparable
  partner and ModelMix does not run on it. See `code/utils/modelmix.py` and
  `train_modelmix_2d.py` (no `--dataset` flag — one run always produces both ACDC's and MSCMR's
  checkpoints).

Each single-dataset method has a `train_<method>_2d.py` (ACDC/MSCMR only) and `train_<method>_3d.py`
(WORD only) script; ModelMix is the one exception (`train_modelmix_2d.py` only, no 3D counterpart).
All scripts are run from the **repository root**.

## Commands

Install deps:
```bash
pip install -r requirements.txt
```

Run tests (plain `unittest`, no pytest config; each file is runnable directly and manipulates
`sys.path` itself — run one file to run its tests, no test selection flag needed for a whole file):
```bash
python code/test/test_networks_2d.py       # CPU regression: UNet2D shapes/backprop/return_features
python code/test/test_networks_3d.py       # CPU regression: UNet3D/ResUNet3D/NNUNet3D
python code/test/test_vnet_3d.py           # VNet3D/VNetCCT3D (WORD backbone)
python code/test/test_scribblebench_2d.py  # ACDC/MSCMR slice dataset + RandomGenerator2D
python code/test/test_cyclemix_utils.py
python code/test/test_dmsps_utils.py
python code/test/test_sdtnet_utils.py
python code/test/test_voxtrust3d_utils.py
python code/test/test_effdnet_utils.py     # FBSL contrastive loss, FADC augmentation, EMA warm-up
python code/test/test_modelmix_utils.py    # image/model mixup, encoder layer mixing, rotation
python code/test/test_train_pce_2d.py      # split resolution, checkpoint round-trip
python code/test/test_train_dmsps_2d.py    # stage-2 expanded-label dataset wiring
python code/test/test_train_voxtrust3d_2d.py  # 2D calibration dataset + full voxtrust_step
python code/test/test_train_effdnet_3d.py  # full effdnet_step, 2D and 3D
python code/test/test_train_modelmix_2d.py # full modelmix_task_step, gradient-flow properties
python code/test/test_common_2d.py         # per-slice resize-then-stitch inference/validation
python code/test/test_common_3d.py
python code/test/test_metrics_3d.py
python code/test/test_word_label_remap.py
```
To run a single test method, use `unittest`'s selector, e.g.:
```bash
python -m unittest code.test.test_networks_2d.Network2DShapeTests.test_unet_2d
```

Train one method/dataset. ACDC/MSCMR use the `_2d.py` script, WORD uses the `_3d.py` script (swap
`pce` for `cyclemix` / `sdtnet` / `voxtrust3d` / `effdnet` / `dmsps`):
```bash
python code/train/train_pce_2d.py --dataset ACDC --amp     # ACDC | MSCMR
python code/train/train_pce_3d.py --dataset WORD --amp     # WORD only
python code/train/train_pce_2d.py --help                    # full argument list
```
DMSPS is two-stage; stage 2 re-initializes from stage 1's best checkpoint:
```bash
python code/train/train_dmsps_2d.py --dataset ACDC --stage 1 --amp
python code/train/train_dmsps_2d.py --dataset ACDC --stage 2 --amp \
  --init_checkpoint checkpoints/ScribbleBench_DMSPS/ACDC/stage1/best.pth
```
ModelMix is the one exception: no `--dataset` flag, always jointly trains both ACDC and MSCMR in
one run, producing two checkpoints:
```bash
python code/train/train_modelmix_2d.py --amp
# -> checkpoints/ScribbleBench_ModelMix/ACDC/best.pth and .../MSCMR/best.pth
```

Evaluate a checkpoint (pCE/CycleMix/SDT-Net/VoxTrust-3D/EFFDNet/ModelMix all deploy a plain
UNet2D/VNet3D checkpoint and share `test_pce_{2d,3d}.py`; DMSPS's dual-decoder network needs
`test_dmsps_{2d,3d}.py`):
```bash
python code/test/test_pce_2d.py --checkpoint checkpoints/ScribbleBench_pCE/ACDC/best.pth --amp
python code/test/test_dmsps_3d.py --checkpoint checkpoints/ScribbleBench_DMSPS/WORD/stage2/best.pth --amp
```

Run everything (train + test all methods x all datasets, one summary CSV; ACDC/MSCMR routed
through the 2D scripts, WORD through the 3D/VNet scripts):
```bash
bash code/train/run_baselines.sh
```
Configurable via env vars (`SCRIBBLE_DATASETS`, `SCRIBBLE_BATCH_SIZE`, `SCRIBBLE_AMP_FLAG`,
`SCRIBBLE_DEVICE`, `SCRIBBLE_ROOT_PATH`, checkpoint/results root overrides, extra train/test args)
— see the header comment of `code/train/run_baselines.sh` for the full list, and
`code/train/run_voxtrust3d.sh` / `run_voxtrust3d_nowarmup.sh` for VoxTrust-3D-only runs with a
different (no-)warm-up schedule.

Quick CPU smoke test of the full pipeline (e.g. before a real run):
```bash
SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 \
SCRIBBLE_EXTRA_TRAIN_ARGS="--max_iterations 4 --early_interval 2 --late_interval 2 --num_workers 0" \
SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
bash code/train/run_baselines.sh
```

## Architecture

```
code/
  dataloader/
    scribblebench_3d.py   # Shared 3D per-case Dataset for ACDC/MSCMR/WORD: augment, WORD label remap
    scribblebench_2d.py   # ACDC/MSCMR flat per-slice Dataset built on scribblebench_3d.py,
                           # RandomGenerator2D augmentation (rot90+flip / rotate / resize)
  networks/
    net_factory.py         # net_factory(net_type, spatial_dims, ...) — construction entry point
    unet_2d.py              # UNet2D / UNetCCT2D backbone (ACDC/MSCMR: pCE/CycleMix/SDT-Net/
                             # VoxTrust-3D use UNet2D; DMSPS uses UNetCCT2D)
    unet_3d.py, unet_cct_3d.py   # 3D counterparts -- no longer used by any train script
                                  # (kept for reuse/reference; WORD uses VNet, see below)
    vnet_3d.py               # VNet3D / VNetCCT3D backbone (WORD: all 5 methods)
    resunet_3d.py, nnunet_3d.py   # Alternative 3D backbones, not wired into any train script
  utils/
    sliding_window_3d.py     # Sliding-window inference for full 3D volumes (WORD)
    cyclemix.py, dmsps.py, sdtnet.py, voxtrust3d.py, effdnet.py   # Per-method loss/pseudo-label
                              # algorithms, shape-agnostic over 2D [B,C,H,W] / 3D [B,C,D,H,W]
                              # (dispatched on tensor.ndim) so the exact same functions back both
                              # pipelines
    modelmix.py               # ModelMix building blocks (2D-only: image/model mixup, encoder-layer
                               # mixing, rotation invariance) -- ACDC+MSCMR is the one task pair here
  train/
    common_3d.py             # Shared infra: split resolution, checkpointing, seeding,
                              # partial_cross_entropy() (also 2D/3D shape-agnostic), sliding-window
                              # validate() for WORD -- imported directly by 2D scripts too
    common_2d.py              # validate_2d()/predict_volume_2d(): per-slice resize-then-stitch
                               # inference for ACDC/MSCMR, mirrors common_3d.validate()
    legacy_splits.py          # Fixed published train/val/test subject IDs (do not resample)
    train_pce_2d.py / train_cyclemix_2d.py / train_dmsps_2d.py / train_sdtnet_2d.py /
      train_voxtrust3d_2d.py / train_effdnet_2d.py    # ACDC/MSCMR only; each *_step()/loss
                                 # function is reused directly from its *_3d.py sibling
                                 # (shape-agnostic), only the dataset, network and augmentation differ
    train_pce_3d.py / train_cyclemix_3d.py / train_dmsps_3d.py / train_sdtnet_3d.py /
      train_voxtrust3d_3d.py / train_effdnet_3d.py    # WORD only (--dataset choices restricted)
    train_modelmix_2d.py       # No 3D counterpart; no --dataset flag -- always jointly trains
                               # ACDC+MSCMR, producing two checkpoints in one run
    run_baselines.sh            # Full train+test sweep, all methods x all datasets, routes
                                 # ACDC/MSCMR through *_2d.py and WORD through *_3d.py, plus a
                                 # separate ModelMix(ACDC+MSCMR) block
    run.sh, run_voxtrust3d.sh, run_voxtrust3d_nowarmup.sh   # Narrower/variant sweeps
  test/
    test_pce_2d.py, test_dmsps_2d.py   # Evaluators for ACDC/MSCMR (per-slice stitch inference) --
                                        # also used for EFFDNet/ModelMix (plain UNet2D checkpoints)
    test_pce_3d.py, test_dmsps_3d.py   # Evaluators for WORD (sliding-window inference) -- also
                                        # used for EFFDNet (plain VNet3D checkpoint)
    metrics_3d.py                       # Dice / HD95 / ASSD -- dimension-agnostic, shared by all
                                         # 4 evaluators above
    append_metrics_csv.py               # Appends one evaluator run's metrics to the summary CSV
    test_*.py                           # Unit tests (CPU, fast)

dataset/ScribbleBench/<ACDC|MSCMR|WORD>/   # Not committed to git
  imagesTr/ labelsTr/ labelsTr_dense/ imagesTs/ labelsTs/

checkpoints/   # best.pth, last.pth, split.json, train.log, tensorboard/ (train output)
results/       # metrics.json, baselines_summary.csv (test output)
```

### Key invariants

- **Scribble vs. dense labels**: the optimizer only ever sees `labelsTr` (scribbles; the
  per-dataset class count — 4 for ACDC/MSCMR, 8 for WORD's 7-organ subset — is the ignore/unlabeled
  value). `labelsTr_dense` is used *only* to compute Dice for checkpoint selection (`best.pth`) and
  must never be used for gradient-based supervision or model selection beyond that. The official
  test split (`imagesTs`/`labelsTs`) is untouched during training.
- **2D vs. 3D split is per-dataset, not a runtime flag**: ACDC/MSCMR always go through the `_2d.py`
  scripts and `ScribbleBench2DDataset`; WORD always goes through the `_3d.py` scripts (their
  `--dataset` choices are restricted accordingly). There is no `--dim` switch inside one script —
  keeping the pipelines as separate files was a deliberate choice over a unified script (see
  `AGENTS.md`/commit history if present) to avoid one file juggling two datasets, two networks and
  two validation strategies.
- **Loss functions are shape-dispatched, not duplicated**: `partial_cross_entropy`
  (`train/common_3d.py`) and every per-method loss in `utils/{cyclemix,dmsps,sdtnet,voxtrust3d}.py`
  branch on `tensor.ndim` (4 → 2D `[B,C,H,W]`, 5 → 3D `[B,C,D,H,W]`) rather than having separate 2D
  and 3D copies. The `*_step()` orchestration functions (`cyclemix_step`, `dmsps_step`,
  `sdtnet_step`, `voxtrust_step`) are themselves dimension-agnostic and are imported directly from
  the `_3d.py` script into the `_2d.py` script — when changing one of these, check both pipelines.
- **Fixed splits, not resampled**: `code/train/legacy_splits.py` hardcodes published
  train/val/test subject/volume/scan IDs per dataset (matching ScribFormer's ACDC `MAAGfold70`,
  CycleMix's MSCMR split, and the original WORD-V0.1.0 `imagesTr`/`imagesVal`/`imagesTs`). Training
  aborts if the data on disk doesn't exactly match the published train+val partition. Change these
  IDs only when deliberately introducing and documenting a new experimental protocol — never to
  make a run "just work" against a different dataset layout. The 2D scripts resolve this same
  case-level split (via `train_pce_2d.py`'s `resolve_case_split`, which wraps
  `common_3d.make_published_split`) and then translate it into flat per-slice training indices
  with `ScribbleBench2DDataset.slice_positions_for_volumes`.
- **Checkpoint compatibility**: within one pipeline, pCE, CycleMix, SDT-Net, and VoxTrust-3D all
  checkpoint a plain `UNet2D`/`VNet3D` `model_state_dict` (for VoxTrust-3D this is the EMA
  teacher's weights, per the paper's "only one EMA network required at inference") and are
  evaluated with the same `test_pce_{2d,3d}.py`. DMSPS uses a dual-decoder network
  (`UNetCCT2D`/`VNetCCT3D`) and has its own evaluator, `test_dmsps_{2d,3d}.py`. Every evaluator
  reads dataset, class count, patch size, and backbone width from the checkpoint itself and loads
  weights strictly — don't hand-edit a checkpoint's shape metadata. A 2D checkpoint's
  `model_config`/`data_config` keys differ slightly from a 3D one (`patch_size_hw` + no
  `feature_chns` list — 2D uses the fixed HiLab-style `UNet2D` width, 3D's VNet uses a single
  `n_filters` int instead of the legacy `feature_chns` list).
- **Crop/resize policy**: WORD defaults to `--foreground_crop_prob 0` (uniform random 3D crop,
  matching the official CycleMix/DMSPS/SDT-Net recipes). ACDC/MSCMR use `RandomGenerator2D`'s fixed
  policy (50% rot90+flip, else 25% random rotate ±20°, else identity, always resized to
  `patch_size` via nearest-neighbor) — the same augmentation for every method, so results stay
  comparable across methods within each dataset.
- **Everything under `checkpoints/`, `results/`, and `dataset/`** is generated/data, not source —
  do not commit volumes, checkpoints, TensorBoard logs, or predictions.
- **ModelMix is the one cross-dataset method**: `train_modelmix_2d.py` has no `--dataset` flag and
  always jointly trains ACDC+MSCMR (`utils/modelmix.py`'s module docstring explains why no other
  pair in this benchmark is a legitimate substitute). Its model-mixup branch builds the "virtual"
  mixed encoder as a `torch.no_grad()` deep copy (verified against the official source's raw
  `.data =` assignment) — backpropagating through it updates only the *decoder* called on top of
  it, never either encoder's real weights directly; each encoder is still updated every step, just
  through that same task's *other* loss terms (own supervision, image-level mixup), which do call
  the real encoder. This is intentional, not a bug — see `mix_one_encoder_layer`'s docstring before
  "fixing" it.

## Coding style

Four-space indentation, `snake_case` for modules/functions/variables/CLI flags, `PascalCase` for
classes. Prefer explicit type/shape validation at public data or model boundaries (see
`partial_cross_entropy` in `code/train/common_3d.py` for the pattern). No repository-wide
formatter/linter — match surrounding style. Import order: standard library, then third-party, then
local modules. Keep scripts runnable from the repository root; use `pathlib.Path` for filesystem
paths.

## Testing guidelines

Add a focused regression test for every network, loss, split, or checkpoint behavior change. Name
unittest methods `test_<behavior>`. Keep default tests small enough for CPU execution — dataset
tests build tiny synthetic nii.gz fixtures with `nibabel`/`tempfile` rather than touching
`dataset/ScribbleBench/` (gitignored, not guaranteed present). Never use dense test labels
(`labelsTs`) during training or model selection — only `labelsTr_dense` on the validation
partition, and only for picking `best.pth`.

## Known pre-existing issues (not part of the 2D/VNet split)

`code/test/test_evaluation_3d.py` and `code/test/test_unet.py` fail even on a clean checkout —
remnants of an earlier iteration of this codebase (`test_evaluation_3d.py` loads a nonexistent
`code/test/test.py`; `test_unet.py` expects a `../../data/ACDC/test.txt` that was never part of
ScribbleBench). `code/train/train_sample2d.py` and `code/dataloader/wavelet_scribble.py` are
similarly dead code from that era. Leave these alone unless specifically asked to clean them up.
