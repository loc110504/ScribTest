#!/usr/bin/env bash
# Full Table 2 ("Ablating Trust Calibration", paper_icassp2027/main.tex)
# ablation -- Dice, PL-Acc, AND PL-Cov for all eight rows -- on ACDC and MSCMR's
# 2D ScribbleBench protocol. New script; does not modify run_voxtrust3d_dcc_ablation.sh
# (that one still only reproduces the five-row Dice-only version via
# test_pce_2d.py) or any train_*.py script.
#
# Reproduces eight rows, all from the SAME --seed, per dataset:
#   pce_only          pCE, no Mean Teacher at all                (train_pce_2d.py)
#   mt_all_pl         MT, all pseudo-labels                      (train_voxtrust3d_2d.py --ablation all_pseudo_labels)
#   mt_global_conf    MT + global confidence threshold           (train_voxtrust3d_2d.py --ablation global_confidence)
#   mt_class_only     MT + held-out calibration (class-only)     (train_voxtrust3d_2d.py --ablation class_only)
#   dcc_full          DCC (full method)                          (train_voxtrust3d_2d.py --ablation full)
#   dcc_wo_wilson     w/o Wilson bound (raw k/n)                 (train_voxtrust3d_2d_ablation_raw_ratio.py)
#   dcc_wo_abstain    w/o abstention (extrapolate)               (train_voxtrust3d_2d_ablation_extrapolate.py)
#   dcc_wo_signal     top-1 confidence instead of R_i            (train_voxtrust3d_2d_ablation_top1conf.py)
#
# The last three arms are each a standalone, never-modified-original train
# script (see their own module docstrings); train_voxtrust3d_2d.py and
# train_voxtrust3d_3d.py are untouched by this project's ablation work.
#
# Every arm is evaluated with test_voxtrust3d_ablation_2d.py, which computes
# Dice, PL-Acc, and PL-Cov together against the held-out VALIDATION split (not
# the official test split -- see that evaluator's own module docstring for
# why: only validation cases carry both real scribble geometry and dense
# reference labels). This is the piece run_voxtrust3d_dcc_ablation.sh's own
# header says it does not yet have.
#
# This project's own Trust-Advantage EMA extension (--ta_ema) is not part of
# the base method Table 2 ablates, so every train_voxtrust3d_2d.py call below
# passes --ta_ema 0 explicitly (matching that script's own default).
#
# Each arm writes to its OWN, never-reused --output_dir and is left to
# complete all --max_iterations uninterrupted before the next one starts
# (guard_fresh_output_dir refuses a fresh run into a directory that already
# holds a checkpoint).
#
# Per dataset, this produces:
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/pce_only/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/mt_all_pl/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/mt_global_conf/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/mt_class_only/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/dcc_full/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/dcc_wo_wilson/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/dcc_wo_abstain/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_full_ablation/<dataset>/dcc_wo_signal/
# and appends one row per arm x dataset to the summary CSV (Dice/PL-Acc/PL-Cov).
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_SEED                    default 2026; SAME seed used for every arm
#   SCRIBBLE_MAX_ITERATIONS          default 30000; forwarded as --max_iterations to every arm
#   SCRIBBLE_GLOBAL_CONFIDENCE_THRESHOLD  default 0.75; forwarded to the mt_global_conf arm only
#   SCRIBBLE_BATCH_SIZE              default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_ABLATION_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_ABLATION_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_ABLATION_CSV              default "<results_root>/dcc_full_ablation_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_EVAL_ARGS          extra args appended to every test_voxtrust3d_ablation_2d.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_MAX_ITERATIONS=8 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--warmup_frac 0.25 --rampup_frac 0.25 --early_interval 4 --late_interval 4 --late_phase_start 4" \
#   SCRIBBLE_EXTRA_EVAL_ARGS="--case_limit 2" \
#   bash code/train/run_voxtrust3d_dcc_full_ablation_2d.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC MSCMR})
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-30000}"
global_confidence_threshold="${SCRIBBLE_GLOBAL_CONFIDENCE_THRESHOLD:-0.75}"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_eval_args="${SCRIBBLE_EXTRA_EVAL_ARGS:-}"

checkpoint_root="${SCRIBBLE_ABLATION_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_ABLATION_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_ABLATION_CSV:-$results_root/dcc_full_ablation_summary.csv}"

evaluate_and_append() {
  # args: dataset configuration_label ckpt_dir results_dir
  local dataset="$1" configuration="$2" ckpt_dir="$3" results_dir="$4"
  echo "=== [$configuration] dataset=$dataset: evaluate (Dice/PL-Acc/PL-Cov, held-out validation split) ==="
  python "$test_dir/test_voxtrust3d_ablation_2d.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" \
    $device_flag $root_path_flag $extra_eval_args

  python "$test_dir/append_ablation_csv.py" \
    --dataset "$dataset" --configuration "$configuration" \
    --checkpoint "$ckpt_dir/best.pth" --metrics_json "$results_dir/metrics.json" --csv "$csv_path"
}

run_pce_arm() {
  local dataset="$1"
  local ckpt_dir="$checkpoint_root/ScribbleBench_VoxTrust3D_dcc_full_ablation/$dataset/pce_only"
  local results_dir="$results_root/ScribbleBench_VoxTrust3D_dcc_full_ablation/$dataset/pce_only"

  echo "=== [pce_only] dataset=$dataset seed=$seed max_iterations=$max_iterations: train ==="
  python "$script_dir/train_pce_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  evaluate_and_append "$dataset" "pCE only" "$ckpt_dir" "$results_dir"
}

run_voxtrust_arm() {
  # args: dataset stage_label configuration_label ablation extra_arm_train_args...
  local dataset="$1" stage_label="$2" configuration="$3" ablation="$4"
  shift 4
  local ckpt_dir results_dir
  ckpt_dir="$checkpoint_root/ScribbleBench_VoxTrust3D_dcc_full_ablation/$dataset/$stage_label"
  results_dir="$results_root/ScribbleBench_VoxTrust3D_dcc_full_ablation/$dataset/$stage_label"

  echo "=== [$stage_label] dataset=$dataset ablation=$ablation seed=$seed" \
       "max_iterations=$max_iterations: train ==="
  python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" --ta_ema 0 \
    --ablation "$ablation" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag "$@" $extra_train_args

  evaluate_and_append "$dataset" "$configuration" "$ckpt_dir" "$results_dir"
}

run_knockout_arm() {
  # args: dataset stage_label configuration_label train_script
  local dataset="$1" stage_label="$2" configuration="$3" train_script="$4"
  local ckpt_dir results_dir
  ckpt_dir="$checkpoint_root/ScribbleBench_VoxTrust3D_dcc_full_ablation/$dataset/$stage_label"
  results_dir="$results_root/ScribbleBench_VoxTrust3D_dcc_full_ablation/$dataset/$stage_label"

  echo "=== [$stage_label] dataset=$dataset seed=$seed max_iterations=$max_iterations: train ==="
  python "$script_dir/$train_script" --dataset "$dataset" --seed "$seed" \
    --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  evaluate_and_append "$dataset" "$configuration" "$ckpt_dir" "$results_dir"
}

for dataset in "${datasets[@]}"; do
  run_pce_arm "$dataset"
  run_voxtrust_arm "$dataset" mt_all_pl "MT, all pseudo-labels" all_pseudo_labels --holdout_fraction 0
  run_voxtrust_arm "$dataset" mt_global_conf "MT + global confidence threshold" global_confidence --holdout_fraction 0 \
    --global_confidence_threshold "$global_confidence_threshold"
  run_voxtrust_arm "$dataset" mt_class_only "MT + held-out calibration (class-only)" class_only
  run_voxtrust_arm "$dataset" dcc_full "DCC (full)" full
  run_knockout_arm "$dataset" dcc_wo_wilson "w/o Wilson bound (raw k/n)" train_voxtrust3d_2d_ablation_raw_ratio.py
  run_knockout_arm "$dataset" dcc_wo_abstain "w/o abstention (extrapolate)" train_voxtrust3d_2d_ablation_extrapolate.py
  run_knockout_arm "$dataset" dcc_wo_signal "top-1 confidence instead of R_i" train_voxtrust3d_2d_ablation_top1conf.py
done

echo "Done. Full ablation summary CSV (Dice/PL-Acc/PL-Cov, both datasets): $csv_path"
