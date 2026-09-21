#!/usr/bin/env bash
# Full 9-row Table 2 ("Ablating Trust Calibration", paper_icassp2027/main.tex)
# component ablation -- Dice, PL-Acc, AND PL-Cov for every row -- on
# ScribbleBench (ACDC by default, Table 2's own scope; pass
# SCRIBBLE_DATASETS="ACDC MSCMR" to also reproduce it on MSCMR).
#
# Self-contained: writes to its OWN checkpoint/results namespace
# (ScribbleBench_VoxTrust3D_table2_ablation), never reusing
# ScribbleBench_VoxTrust3D_dcc_full_ablation/ (the older, 8-row script,
# run_voxtrust3d_dcc_full_ablation_2d.sh) so this script cannot collide with
# -- or be blocked by guard_fresh_output_dir on top of -- checkpoints that
# script may already have produced.
#
# Reproduces exactly the 9 rows of Table 2, all from the SAME --seed, per
# dataset. Except the first row, every row trains on the IDENTICAL
# Omega_sup (eta=0.15), matching the table's own caption -- isolating the
# acceptance-rule design from the annotation budget:
#   mt_full_budget      MT, full scribble budget          (train_voxtrust3d_2d.py --ablation all_pseudo_labels --holdout_fraction 0)
#   mt_matched_all_pl    MT, matched Omega_sup, all PLs     (train_voxtrust3d_2d.py --ablation all_pseudo_labels --holdout_fraction 0.15)
#   mt_fixed_confidence  MT + fixed confidence              (train_voxtrust3d_2d.py --ablation global_confidence --holdout_fraction 0.15)
#   in_sample_class_only In-sample, class-only              (train_voxtrust3d_2d_ablation_insample.py)
#   held_out_class_only  Held-out, class-only                (train_voxtrust3d_2d.py --ablation class_only)
#   scribcal_full        ScribCal (full)                     (train_voxtrust3d_2d.py --ablation full)
#   wo_wilson            w/o Wilson bound (raw k/n)          (train_voxtrust3d_2d_ablation_raw_ratio.py)
#   wo_abstain           w/o abstention (extrapolate)        (train_voxtrust3d_2d_ablation_extrapolate.py)
#   top1_confidence      top-1 confidence instead of R_i     (train_voxtrust3d_2d_ablation_top1conf.py)
#
# "mt_full_budget" is the sole exception (eta=0, the complete scribble set,
# unfiltered transfer) -- the reference point for the annotation-budget cost
# the other 8 rows all pay identically. Every train_voxtrust3d_2d.py call
# passes --ta_ema 0: Trust-Advantage EMA is this project's own extension to
# the base method, not one of the mechanisms Table 2 ablates.
#
# Every arm is evaluated with test_voxtrust3d_ablation_2d.py (Dice/PL-Acc/
# PL-Cov together, against the held-out VALIDATION split -- see that
# evaluator's own module docstring for why: only validation cases carry
# both real scribble geometry and dense reference labels). This Dice number
# is therefore NOT directly comparable to Table 1's test-set Dice; it is
# only meant to be compared across this table's own 9 rows.
#
# Per dataset, this produces:
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_table2_ablation/<dataset>/{
#     mt_full_budget, mt_matched_all_pl, mt_fixed_confidence,
#     in_sample_class_only, held_out_class_only, scribcal_full,
#     wo_wilson, wo_abstain, top1_confidence}/
# and appends one row per arm x dataset to one summary CSV.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC" (Table 2's own scope)
#   SCRIBBLE_SEED                    default 2026; SAME seed used for every arm
#   SCRIBBLE_MAX_ITERATIONS          default 30000; forwarded as --max_iterations to every arm
#   SCRIBBLE_HOLDOUT_FRACTION        default 0.15; eta for every row except mt_full_budget
#   SCRIBBLE_GLOBAL_CONFIDENCE_THRESHOLD  default 0.75; forwarded to the mt_fixed_confidence arm only
#   SCRIBBLE_BATCH_SIZE              default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_TABLE2_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_TABLE2_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_TABLE2_CSV              default "<results_root>/table2_ablation_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS        extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_EVAL_ARGS         extra args appended to every test_voxtrust3d_ablation_2d.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_MAX_ITERATIONS=8 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--warmup_frac 0.25 --rampup_frac 0.25 --early_interval 4 --late_interval 4 --late_phase_start 4" \
#   SCRIBBLE_EXTRA_EVAL_ARGS="--case_limit 2" \
#   bash code/train/run_table2_component_ablation.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC})
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-30000}"
holdout_fraction="${SCRIBBLE_HOLDOUT_FRACTION:-0.15}"
global_confidence_threshold="${SCRIBBLE_GLOBAL_CONFIDENCE_THRESHOLD:-0.75}"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_eval_args="${SCRIBBLE_EXTRA_EVAL_ARGS:-}"

checkpoint_root="${SCRIBBLE_TABLE2_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_TABLE2_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_TABLE2_CSV:-$results_root/table2_ablation_summary.csv}"
namespace="ScribbleBench_VoxTrust3D_table2_ablation"

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

run_voxtrust_arm() {
  # args: dataset stage_label configuration_label ablation eta extra_arm_train_args...
  local dataset="$1" stage_label="$2" configuration="$3" ablation="$4" eta="$5"
  shift 5
  local ckpt_dir results_dir
  ckpt_dir="$checkpoint_root/$namespace/$dataset/$stage_label"
  results_dir="$results_root/$namespace/$dataset/$stage_label"

  echo "=== [$stage_label] dataset=$dataset ablation=$ablation eta=$eta seed=$seed" \
       "max_iterations=$max_iterations: train ==="
  python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" --ta_ema 0 \
    --ablation "$ablation" --holdout_fraction "$eta" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag "$@" $extra_train_args

  evaluate_and_append "$dataset" "$configuration" "$ckpt_dir" "$results_dir"
}

run_insample_arm() {
  # args: dataset stage_label configuration_label
  local dataset="$1" stage_label="$2" configuration="$3"
  local ckpt_dir results_dir
  ckpt_dir="$checkpoint_root/$namespace/$dataset/$stage_label"
  results_dir="$results_root/$namespace/$dataset/$stage_label"

  echo "=== [$stage_label] dataset=$dataset eta=$holdout_fraction seed=$seed" \
       "max_iterations=$max_iterations: train ==="
  python "$script_dir/train_voxtrust3d_2d_ablation_insample.py" --dataset "$dataset" --seed "$seed" \
    --holdout_fraction "$holdout_fraction" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  evaluate_and_append "$dataset" "$configuration" "$ckpt_dir" "$results_dir"
}

run_knockout_arm() {
  # args: dataset stage_label configuration_label train_script
  local dataset="$1" stage_label="$2" configuration="$3" train_script="$4"
  local ckpt_dir results_dir
  ckpt_dir="$checkpoint_root/$namespace/$dataset/$stage_label"
  results_dir="$results_root/$namespace/$dataset/$stage_label"

  echo "=== [$stage_label] dataset=$dataset eta=$holdout_fraction seed=$seed" \
       "max_iterations=$max_iterations: train ==="
  python "$script_dir/$train_script" --dataset "$dataset" --seed "$seed" \
    --holdout_fraction "$holdout_fraction" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  evaluate_and_append "$dataset" "$configuration" "$ckpt_dir" "$results_dir"
}

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: Table 2's component ablation is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  run_voxtrust_arm "$dataset" mt_full_budget "MT, full scribble budget" all_pseudo_labels 0
  run_voxtrust_arm "$dataset" mt_matched_all_pl "MT, matched Omega_sup, all PLs" all_pseudo_labels "$holdout_fraction"
  run_voxtrust_arm "$dataset" mt_fixed_confidence "MT + fixed confidence" global_confidence "$holdout_fraction" \
    --global_confidence_threshold "$global_confidence_threshold"
  run_insample_arm "$dataset" in_sample_class_only "In-sample, class-only"
  run_voxtrust_arm "$dataset" held_out_class_only "Held-out, class-only" class_only "$holdout_fraction"
  run_voxtrust_arm "$dataset" scribcal_full "ScribCal (full)" full "$holdout_fraction"
  run_knockout_arm "$dataset" wo_wilson "w/o Wilson bound (raw k/n)" train_voxtrust3d_2d_ablation_raw_ratio.py
  run_knockout_arm "$dataset" wo_abstain "w/o abstention (extrapolate)" train_voxtrust3d_2d_ablation_extrapolate.py
  run_knockout_arm "$dataset" top1_confidence "top-1 confidence instead of R_i" train_voxtrust3d_2d_ablation_top1conf.py
done

echo "Done. Table 2 summary CSV: $csv_path"
echo "Fill main.tex's Table 2 (tab:ablation) directly from this CSV's dice/pl_acc/pl_cov columns,"
echo "matched by the 'configuration' column to each LaTeX row label."
