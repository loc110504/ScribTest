#!/usr/bin/env bash
# Ablation: VoxTrust-3D with Calibration-Gated EMA (--ta_ema 1, the default) vs.
# the fixed-rate EMA baseline (--ta_ema 0), on ACDC and MSCMR.
#
# Both arms use the SAME --seed and write to their OWN, never-reused
# --output_dir, and each run is left to complete all --max_iterations
# uninterrupted before the next one starts. This is deliberate: an earlier
# comparison mixed up two different checkpoint lineages because a fresh
# (non-`--resume`) run was relaunched into an --output_dir that already held a
# better checkpoint from a previous run -- train_*.py resets best_score to
# -inf on every fresh start, so that silently overwrote the better checkpoint
# with a worse one, and the two "before/after" numbers being compared ended up
# coming from unrelated runs. `guard_fresh_output_dir` (code/train/common_3d.py)
# now refuses to let that happen -- if you rerun this script into the same
# output dirs, it will fail loudly with an error telling you to remove/rename
# the old directory or pass --resume, rather than repeat that mistake.
#
# Per dataset, this produces:
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_ablation/<dataset>/ta_ema_off/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_ablation/<dataset>/ta_ema_on/
# and appends one row per arm to the summary CSV (method=VoxTrust3D,
# stage=ta_ema_off/ta_ema_on) so both can be compared side by side.
#
# --max_iterations went from the training scripts' own default of 30000 to
# 35000 here, but --warmup_frac/--rampup_frac (fractions of --max_iterations)
# and --late_phase_start (an absolute iteration count) are all left at their
# *absolute* iteration counts under the original 30000-iteration protocol --
# warm-up still ends at iteration 3000, the pseudo-label weight still finishes
# ramping up by iteration 9000, and the finer late-phase checkpoint cadence
# still starts at iteration 20000. The extra 5000 iterations are pure
# additional training time in that same late phase, not a rescaled schedule.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_SEED                    default 2026; SAME seed used for both arms
#   SCRIBBLE_MAX_ITERATIONS          default 35000; forwarded as --max_iterations
#   SCRIBBLE_WARMUP_ITERS            default 3000 (absolute); converted to --warmup_frac
#   SCRIBBLE_RAMPUP_ITERS            default 6000 (absolute); converted to --rampup_frac
#   SCRIBBLE_LATE_PHASE_START        default 20000 (absolute); forwarded as --late_phase_start
#   SCRIBBLE_BATCH_SIZE              default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_ABLATION_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_ABLATION_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_ABLATION_CSV              default "<results_root>/ta_ema_ablation_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_voxtrust3d_*.py call
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_pce_*.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only (the default
# warmup/rampup/late-phase absolute values only make sense at 35000
# iterations, so scale them down too for a tiny smoke run):
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_MAX_ITERATIONS=8 SCRIBBLE_WARMUP_ITERS=2 SCRIBBLE_RAMPUP_ITERS=2 SCRIBBLE_LATE_PHASE_START=4 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--early_interval 4 --late_interval 4" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_voxtrust3d_taema_ablation.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC MSCMR})
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-35000}"
warmup_iters="${SCRIBBLE_WARMUP_ITERS:-3000}"
rampup_iters="${SCRIBBLE_RAMPUP_ITERS:-6000}"
late_phase_start="${SCRIBBLE_LATE_PHASE_START:-20000}"
# warmup_frac/rampup_frac are fractions of --max_iterations, so they must be
# recomputed here to keep the *absolute* warm-up/ramp-up iteration counts
# fixed even as --max_iterations changes (see header comment).
warmup_frac="$(awk -v w="$warmup_iters" -v m="$max_iterations" 'BEGIN { printf "%.10f", w / m }')"
rampup_frac="$(awk -v r="$rampup_iters" -v m="$max_iterations" 'BEGIN { printf "%.10f", r / m }')"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_ABLATION_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_ABLATION_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_ABLATION_CSV:-$results_root/ta_ema_ablation_summary.csv}"

dim_for_dataset() {
  case "$1" in
    WORD) echo 3d ;;
    *) echo 2d ;;
  esac
}

run_arm() {
  # args: dataset ta_ema_flag stage_label
  local dataset="$1" ta_ema_flag="$2" stage_label="$3"
  local dim ckpt_dir results_dir
  dim="$(dim_for_dataset "$dataset")"
  ckpt_dir="$checkpoint_root/ScribbleBench_VoxTrust3D_ablation/$dataset/$stage_label"
  results_dir="$results_root/ScribbleBench_VoxTrust3D_ablation/$dataset/$stage_label"

  echo "=== [VoxTrust3D/$stage_label] dataset=$dataset ($dim) seed=$seed max_iterations=$max_iterations" \
       "(warmup_iters=$warmup_iters rampup_iters=$rampup_iters late_phase_start=$late_phase_start): train ==="
  python "$script_dir/train_voxtrust3d_${dim}.py" --dataset "$dataset" --seed "$seed" \
    --ta_ema "$ta_ema_flag" --max_iterations "$max_iterations" \
    --warmup_frac "$warmup_frac" --rampup_frac "$rampup_frac" --late_phase_start "$late_phase_start" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  echo "=== [VoxTrust3D/$stage_label] dataset=$dataset ($dim): test ==="
  python "$test_dir/test_pce_${dim}.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args

  python "$test_dir/append_metrics_csv.py" \
    --method VoxTrust3D --dataset "$dataset" --stage "$stage_label" \
    --checkpoint "$ckpt_dir/best.pth" --metrics_json "$results_dir/metrics.json" --csv "$csv_path"
}

for dataset in "${datasets[@]}"; do
  run_arm "$dataset" 0 ta_ema_off
  run_arm "$dataset" 1 ta_ema_on
done

echo "Done. Ablation summary CSV: $csv_path"
echo "Compare rows where stage=ta_ema_off vs stage=ta_ema_on for the same dataset."
