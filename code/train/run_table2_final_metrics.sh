#!/usr/bin/env bash
# Final-metrics pass over Table 2's 9 already-trained arms
# ("Ablating Trust Calibration", paper_icassp2027/main.tex): re-evaluates
# every arm's EXISTING checkpoint (produced by
# run_table2_component_ablation.sh) with two evaluators instead of one --
#
#   - Dice / HD95 / ASSD: test_pce_2d.py against the official, never-touched
#     TEST split (imagesTs/labelsTs) -- the same evaluator/split every other
#     ScribbleBench baseline reports its headline Dice from, so these
#     numbers ARE directly comparable to Table 1.
#   - PL-Acc / PL-Cov: test_voxtrust3d_ablation_2d.py against the held-out
#     VALIDATION split, UNCHANGED from run_table2_component_ablation.sh --
#     that split is the only case pool with both real scribble geometry and
#     dense labels, which the accept-rule replay needs (see that script's
#     module docstring). Its own Dice number is intentionally NOT used here.
#
# This script does NOT train anything -- it assumes
# run_table2_component_ablation.sh (or an equivalent run) already produced
# best.pth for every arm under
# <checkpoint_root>/ScribbleBench_VoxTrust3D_table2_ablation/<dataset>/<stage_label>/.
# An arm whose checkpoint is missing is skipped with a warning, not an error.
#
# Per dataset, appends one row per arm to one summary CSV (separate from
# table2_ablation_summary.csv -- that file's own "dice" column is the
# held-out-validation number; this script's CSV reports test-set Dice/HD95
# instead, so the two must not be confused/merged blindly).
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                 space-separated subset, default "ACDC"
#   SCRIBBLE_AMP_FLAG                 default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                   e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_ROOT_PATH                ScribbleBench root override (--root_path)
#   SCRIBBLE_EVAL_TARGET               default "student" (matches every arm's own best.pth
#                                      selection criterion); pass "teacher" to instead score
#                                      the EMA teacher half of the Mean Teacher pair
#   SCRIBBLE_TABLE2_CHECKPOINT_ROOT   default "<repo>/checkpoints" (must match the root
#                                      run_table2_component_ablation.sh trained into)
#   SCRIBBLE_TABLE2_RESULTS_ROOT      default "<repo>/results"
#   SCRIBBLE_TABLE2_FINAL_CSV         default "<results_root>/table2_final_metrics_summary.csv"
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_pce_2d.py call
#   SCRIBBLE_EXTRA_EVAL_ARGS          extra args appended to every test_voxtrust3d_ablation_2d.py call
#
# Example:
#   SCRIBBLE_DATASETS="ACDC MSCMR" SCRIBBLE_DEVICE=cuda SCRIBBLE_AMP_FLAG="--amp" \
#   bash code/train/run_table2_final_metrics.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC})
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
eval_target="${SCRIBBLE_EVAL_TARGET:-student}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"
extra_eval_args="${SCRIBBLE_EXTRA_EVAL_ARGS:-}"

checkpoint_root="${SCRIBBLE_TABLE2_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_TABLE2_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_TABLE2_FINAL_CSV:-$results_root/table2_final_metrics_summary.csv}"
namespace="ScribbleBench_VoxTrust3D_table2_ablation"

# stage_label:configuration pairs, matching run_table2_component_ablation.sh exactly.
arms=(
  "mt_full_budget:MT, full scribble budget"
  "mt_matched_all_pl:MT, matched Omega_sup, all PLs"
  "mt_fixed_confidence:MT + fixed confidence"
  "in_sample_class_only:In-sample, class-only"
  "held_out_class_only:Held-out, class-only"
  "scribcal_full:ScribCal (full)"
  "wo_wilson:w/o Wilson bound (raw k/n)"
  "wo_abstain:w/o abstention (extrapolate)"
  "top1_confidence:top-1 confidence instead of R_i"
)

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: Table 2's component ablation is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  for arm in "${arms[@]}"; do
    stage_label="${arm%%:*}"
    configuration="${arm#*:}"
    ckpt_dir="$checkpoint_root/$namespace/$dataset/$stage_label"
    best_ckpt="$ckpt_dir/best.pth"

    if [ ! -f "$best_ckpt" ]; then
      echo "=== [$stage_label] dataset=$dataset: no checkpoint at $best_ckpt -- skipping ===" >&2
      continue
    fi

    test_results_dir="$results_root/$namespace/$dataset/$stage_label/test_pce"
    val_results_dir="$results_root/$namespace/$dataset/$stage_label/val_ablation"

    echo "=== [$stage_label] dataset=$dataset: test-set Dice/HD95 (test_pce_2d.py) ===" >&2
    python "$test_dir/test_pce_2d.py" \
      --checkpoint "$best_ckpt" --output_dir "$test_results_dir" \
      --eval_target "$eval_target" $amp_flag $device_flag $root_path_flag $extra_test_args

    echo "=== [$stage_label] dataset=$dataset: val-set PL-Acc/PL-Cov (test_voxtrust3d_ablation_2d.py) ===" >&2
    python "$test_dir/test_voxtrust3d_ablation_2d.py" \
      --checkpoint "$best_ckpt" --output_dir "$val_results_dir" \
      $device_flag $root_path_flag $extra_eval_args

    python "$test_dir/append_table2_final_csv.py" \
      --dataset "$dataset" --configuration "$configuration" \
      --checkpoint "$best_ckpt" \
      --test_metrics_json "$test_results_dir/metrics.json" \
      --val_metrics_json "$val_results_dir/metrics.json" \
      --csv "$csv_path"
  done
done

echo "Done. Table 2 final-metrics summary CSV: $csv_path"
echo "dice_test/hd95_test come from the official test split; pl_acc_val/pl_cov_val from the held-out validation split."
