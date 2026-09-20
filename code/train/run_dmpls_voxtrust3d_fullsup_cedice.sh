#!/usr/bin/env bash
# Train + test three specific runs, appending each result to one summary CSV:
#   1. DMPLS (Luo et al., MICCAI 2022)               -- MSCMR only.
#   2. VoxTrust-3D's 2D pipeline, fixed-rate EMA      -- ACDC + MSCMR, pinned
#      (train_voxtrust3d_2d.py) with --ta_ema 0          to --ta_ema 0 (Trust-
#                                                         Advantage EMA off).
#   3. FullSup, the dense-mask upper bound            -- ACDC + MSCMR, now
#      (train_fullsup_2d.py)                             trained with a CE +
#                                                         soft-Dice compound
#                                                         loss (see below).
#
# This is a narrower, ad hoc sibling of run_dmpls_bayes_wss.sh/run_voxtrust3d.sh/
# run_fullsup.sh -- it does not run Bayes-WSS, and it deliberately runs DMPLS
# on MSCMR only (not ACDC): pass SCRIBBLE_DMPLS_DATASETS="ACDC MSCMR" to widen
# it if an ACDC DMPLS row is wanted too.
#
# --- VoxTrust-3D / --ta_ema 0 -----------------------------------------------
# --ta_ema 0 is already train_voxtrust3d_2d.py's own default (Trust-Advantage
# EMA is opt-in via --ta_ema 1); it is passed explicitly here so the run is
# self-documenting and stays correct even if that default ever changes. This
# uses the standard 30000-iteration matched protocol (train_voxtrust3d_2d.py's
# own defaults), NOT run_voxtrust3d_taema_ablation.sh's 35000-iteration
# schedule -- the two are not meant to be compared iteration-for-iteration.
# To avoid silently clobbering (or colliding with) a checkpoint written by a
# plain run_voxtrust3d.sh run (which writes to
# <checkpoint_root>/ScribbleBench_VoxTrust3D/<dataset>/best.pth directly),
# this script writes to a ta_ema_off subdirectory instead and records
# stage=ta_ema_off in the summary CSV, mirroring
# run_voxtrust3d_taema_ablation.sh's naming for the same arm.
#
# --- FullSup / CE + Dice -----------------------------------------------------
# train_fullsup_2d.py's loss changed from plain cross-entropy to an unmasked,
# unweighted CE + soft-Dice compound loss (loss = ce + dice, via
# utils/sdtnet.py's soft_dice_loss -- the same combination
# train_modelmix_2d.py/train_sdtnet_3d.py already use); see that script's
# module docstring. This writes to the SAME canonical directory
# run_fullsup.sh uses (<checkpoint_root>/ScribbleBench_FullSup/<dataset>) and
# records stage=ce_dice, since the method identity (FullSup) hasn't changed,
# only its loss implementation -- but if you already have a best.pth there
# from BEFORE this change (plain-CE FullSup), common_3d.guard_fresh_output_dir
# will refuse this run with a loud error rather than overwrite it; remove/
# rename that directory or pass --output_dir yourself to point elsewhere.
#
# All three methods checkpoint a plain network compatible with the shared
# evaluators: DMPLS's dual-decoder UNetCCT2D uses test_dmpls_2d.py; VoxTrust-
# 3D and FullSup checkpoint a plain UNet2D and use test_pce_2d.py (VoxTrust-
# 3D's --eval_target defaults to "student", matching every other baseline
# here -- see CLAUDE.md's "Checkpoint compatibility" note).
#
# Environment variable overrides (all optional, same names as
# run_baselines.sh/run_voxtrust3d.sh so scripts can share a shell environment):
#   SCRIBBLE_DMPLS_DATASETS          space-separated subset, default "MSCMR"
#   SCRIBBLE_VOXTRUST3D_DATASETS     space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_FULLSUP_DATASETS        space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_SEED                    default 2026; forwarded as --seed to every train_*.py call
#   SCRIBBLE_MAX_ITERATIONS          default 30000; forwarded as --max_iterations
#   SCRIBBLE_BATCH_SIZE              default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_BASELINES_CHECKPOINT_ROOT default "<repo>/checkpoints"
#   SCRIBBLE_BASELINES_RESULTS_ROOT    default "<repo>/results"
#   SCRIBBLE_BASELINES_CSV             default "<results_root>/baselines_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_*.py call
#
# Example - quick end-to-end smoke run on CPU:
#   SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_MAX_ITERATIONS=8 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--early_interval 4 --late_interval 4" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_dmpls_voxtrust3d_fullsup_cedice.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

dmpls_datasets=(${SCRIBBLE_DMPLS_DATASETS:-MSCMR})
voxtrust3d_datasets=(${SCRIBBLE_VOXTRUST3D_DATASETS:-ACDC MSCMR})
fullsup_datasets=(${SCRIBBLE_FULLSUP_DATASETS:-ACDC MSCMR})

seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-30000}"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_BASELINES_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_BASELINES_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_BASELINES_CSV:-$results_root/baselines_summary.csv}"

append_row() {
  # args: method dataset stage checkpoint metrics_json
  python "$test_dir/append_metrics_csv.py" \
    --method "$1" --dataset "$2" --stage "$3" --checkpoint "$4" --metrics_json "$5" --csv "$csv_path"
}

# --- 1. DMPLS -----------------------------------------------------------
for dataset in "${dmpls_datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: DMPLS is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  dmpls_ckpt_dir="$checkpoint_root/ScribbleBench_DMPLS/$dataset"
  dmpls_results_dir="$results_root/ScribbleBench_DMPLS/$dataset"
  echo "=== [DMPLS] dataset=$dataset seed=$seed max_iterations=$max_iterations: train ==="
  python "$script_dir/train_dmpls_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
    --output_dir "$dmpls_ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args
  echo "=== [DMPLS] dataset=$dataset: test ==="
  python "$test_dir/test_dmpls_2d.py" \
    --checkpoint "$dmpls_ckpt_dir/best.pth" --output_dir "$dmpls_results_dir" \
    $amp_flag $device_flag $root_path_flag $extra_test_args
  append_row DMPLS "$dataset" "" "$dmpls_ckpt_dir/best.pth" "$dmpls_results_dir/metrics.json"
done

# --- 2. VoxTrust-3D (2D pipeline), fixed-rate EMA (--ta_ema 0) ----------
for dataset in "${voxtrust3d_datasets[@]}"; do
  vox_ckpt_dir="$checkpoint_root/ScribbleBench_VoxTrust3D/$dataset/ta_ema_off"
  vox_results_dir="$results_root/ScribbleBench_VoxTrust3D/$dataset/ta_ema_off"
  echo "=== [VoxTrust3D/ta_ema_off] dataset=$dataset seed=$seed max_iterations=$max_iterations: train ==="
  python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
    --ta_ema 0 \
    --output_dir "$vox_ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args
  echo "=== [VoxTrust3D/ta_ema_off] dataset=$dataset: test ==="
  python "$test_dir/test_pce_2d.py" \
    --checkpoint "$vox_ckpt_dir/best.pth" --output_dir "$vox_results_dir" \
    $amp_flag $device_flag $root_path_flag $extra_test_args
  append_row VoxTrust3D "$dataset" ta_ema_off "$vox_ckpt_dir/best.pth" "$vox_results_dir/metrics.json"
done

# --- 3. FullSup (dense-mask upper bound), CE + Dice loss ----------------
for dataset in "${fullsup_datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: FullSup is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  fullsup_ckpt_dir="$checkpoint_root/ScribbleBench_FullSup/$dataset"
  fullsup_results_dir="$results_root/ScribbleBench_FullSup/$dataset"
  echo "=== [FullSup/ce_dice] dataset=$dataset seed=$seed max_iterations=$max_iterations: train ==="
  python "$script_dir/train_fullsup_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
    --output_dir "$fullsup_ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args
  echo "=== [FullSup/ce_dice] dataset=$dataset: test ==="
  python "$test_dir/test_pce_2d.py" \
    --checkpoint "$fullsup_ckpt_dir/best.pth" --output_dir "$fullsup_results_dir" \
    $amp_flag $device_flag $root_path_flag $extra_test_args
  append_row FullSup "$dataset" ce_dice "$fullsup_ckpt_dir/best.pth" "$fullsup_results_dir/metrics.json"
done

echo "Done. Summary CSV: $csv_path"
echo "Rows: DMPLS (dataset=${dmpls_datasets[*]}), VoxTrust3D (stage=ta_ema_off), FullSup (stage=ce_dice)."
