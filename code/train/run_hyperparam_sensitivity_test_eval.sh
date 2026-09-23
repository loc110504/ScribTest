#!/usr/bin/env bash
# Test-split final-metrics pass over run_hyperparam_sensitivity.sh's 9
# already-trained (rho, B) cells (target precision rho x distance strata
# B, ScribCal's two core calibration knobs): re-evaluates each cell's
# EXISTING checkpoint with test_pce_2d.py against the official, never-
# touched-during-training TEST split (imagesTs/labelsTs) -- the same
# evaluator/split every other ScribbleBench baseline reports its headline
# Dice/HD95 from, unlike run_hyperparam_sensitivity.sh's own CSV, which
# reports mean Dice on the held-out VALIDATION split (see that script's
# header comment and test_voxtrust3d_ablation_2d.py's module docstring for
# why: PL-Acc/PL-Cov need real scribble geometry, which only imagesTr's
# held-out split carries -- this test-set pass does not attempt to compute
# PL-Acc/PL-Cov at all, since imagesTs/labelsTs carry no scribbles).
#
# This script does NOT train anything -- it assumes
# run_hyperparam_sensitivity.sh (or an equivalent run) already produced
# best.pth for every (rho, B) cell under
# <checkpoint_root>/ScribbleBench_VoxTrust3D_hparam_sensitivity/<dataset>/rho<RR>_B<B>/.
# A cell whose checkpoint is missing is skipped with a warning, not an
# error.
#
# Per dataset x (rho, B), appends one row to its OWN summary CSV --
# hparam_sensitivity_test_summary.csv, separate from
# hparam_sensitivity_summary.csv (that file's Dice is the held-out-
# validation number; this script's CSV reports test-set Dice/HD95/ASSD
# instead, so the two must not be merged blindly). Plot with
# code/test/plot_hparam_sensitivity.py.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                 space-separated subset, default "ACDC"
#   SCRIBBLE_TARGET_PRECISIONS        space-separated rho values, default "0.90 0.95 0.99"
#   SCRIBBLE_DISTANCE_STRATA          space-separated B values, default "1 3 5"
#   SCRIBBLE_AMP_FLAG                 default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                   e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_ROOT_PATH                ScribbleBench root override (--root_path)
#   SCRIBBLE_EVAL_TARGET              default "student" (matches every cell's own best.pth
#                                     selection criterion); pass "teacher" to instead score
#                                     the EMA teacher half of the Mean Teacher pair
#   SCRIBBLE_HPARAM_CHECKPOINT_ROOT   default "<repo>/checkpoints" (must match the root
#                                     run_hyperparam_sensitivity.sh trained into)
#   SCRIBBLE_HPARAM_RESULTS_ROOT      default "<repo>/results"
#   SCRIBBLE_HPARAM_TEST_CSV          default "<results_root>/hparam_sensitivity_test_summary.csv"
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_pce_2d.py call
#
# Example:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cuda SCRIBBLE_AMP_FLAG="--amp" \
#   bash code/train/run_hyperparam_sensitivity_test_eval.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC})
target_precisions=(${SCRIBBLE_TARGET_PRECISIONS:-0.90 0.95 0.99})
distance_strata_values=(${SCRIBBLE_DISTANCE_STRATA:-1 3 5})
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
eval_target="${SCRIBBLE_EVAL_TARGET:-student}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_HPARAM_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_HPARAM_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_HPARAM_TEST_CSV:-$results_root/hparam_sensitivity_test_summary.csv}"
namespace="ScribbleBench_VoxTrust3D_hparam_sensitivity"

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: this hyperparameter-sensitivity figure is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  for rho in "${target_precisions[@]}"; do
    for strata in "${distance_strata_values[@]}"; do
      ckpt_dir="$checkpoint_root/$namespace/$dataset/rho${rho}_B${strata}"
      best_ckpt="$ckpt_dir/best.pth"

      if [ ! -f "$best_ckpt" ]; then
        echo "=== dataset=$dataset rho=$rho B=$strata: no checkpoint at $best_ckpt -- skipping ===" >&2
        continue
      fi

      test_results_dir="$results_root/$namespace/$dataset/rho${rho}_B${strata}/test_pce"

      echo "=== dataset=$dataset rho=$rho B=$strata: test-set Dice/HD95 (test_pce_2d.py) ===" >&2
      python "$test_dir/test_pce_2d.py" \
        --checkpoint "$best_ckpt" --output_dir "$test_results_dir" \
        --eval_target "$eval_target" $amp_flag $device_flag $root_path_flag $extra_test_args

      python "$test_dir/append_hparam_sensitivity_test_csv.py" \
        --dataset "$dataset" --target_precision "$rho" --distance_strata "$strata" \
        --checkpoint "$best_ckpt" \
        --test_metrics_json "$test_results_dir/metrics.json" \
        --csv "$csv_path"
    done
  done
done

echo "Done. Hyperparameter-sensitivity TEST-set summary CSV: $csv_path"
echo "Plot with: python code/test/plot_hparam_sensitivity.py --csv $csv_path"
