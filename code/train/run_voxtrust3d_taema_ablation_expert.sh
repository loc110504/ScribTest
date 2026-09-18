#!/usr/bin/env bash
# Ablation: VoxTrust-3D with Calibration-Gated EMA (--ta_ema 1, the default) vs.
# the fixed-rate EMA baseline (--ta_ema 0), on the "expert scribble" ACDC/MSCMR
# archive (<repo_root>/data/{ACDC,MSCMR}) -- the *_2d_expert.py counterpart of
# run_voxtrust3d_taema_ablation.sh, mirroring how run_expert_baselines.sh is
# the *_2d_expert.py counterpart of run_baselines.sh. This archive is
# 2D-only (no WORD), so there is no dimension dispatch: every run goes
# through train_voxtrust3d_2d_expert.py / test_pce_2d_expert.py.
#
# VoxTrust-3D is deliberately excluded from run_expert_baselines.sh itself
# (see that script's header), but train_voxtrust3d_2d_expert.py still exists
# and works standalone -- this script drives it directly, the same way
# run_voxtrust3d_taema_ablation.sh drives train_voxtrust3d_2d.py outside of
# run_baselines.sh.
#
# Both arms use the SAME --seed and write to their OWN, never-reused
# --output_dir, and each run completes all --max_iterations uninterrupted
# before the next starts -- see run_voxtrust3d_taema_ablation.sh's header for
# why this matters (a fresh, non-`--resume` run resets best_score to -inf and
# silently overwrites a better checkpoint from a previous run;
# guard_fresh_output_dir in code/train/common_3d.py now refuses this outright
# unless --resume is passed).
#
# --max_iterations defaults to 35000 here (the training scripts' own default
# is 30000), but --warmup_frac/--rampup_frac (fractions of --max_iterations)
# and --late_phase_start (an absolute iteration count) are recomputed to keep
# their *absolute* iteration counts fixed at the original 30000-iteration
# protocol's values -- warm-up still ends at iteration 3000, the pseudo-label
# weight still finishes ramping up by iteration 9000, and the finer
# late-phase checkpoint cadence still starts at iteration 20000. The extra
# 5000 iterations are pure additional late-phase training time, not a
# rescaled schedule.
#
# Per dataset, this produces:
#   <checkpoint_root>/ExpertScribble_VoxTrust3D_ablation/<dataset>/ta_ema_off/
#   <checkpoint_root>/ExpertScribble_VoxTrust3D_ablation/<dataset>/ta_ema_on/
# and appends one row per arm to the summary CSV (method=VoxTrust3D,
# stage=ta_ema_off/ta_ema_on) so both can be compared side by side.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_EXPERT_DATASETS         space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_SEED                    default 2026; SAME seed used for both arms
#   SCRIBBLE_MAX_ITERATIONS          default 35000; forwarded as --max_iterations
#   SCRIBBLE_WARMUP_ITERS            default 3000 (absolute); converted to --warmup_frac
#   SCRIBBLE_RAMPUP_ITERS            default 6000 (absolute); converted to --rampup_frac
#   SCRIBBLE_LATE_PHASE_START        default 20000 (absolute); forwarded as --late_phase_start
#   SCRIBBLE_BATCH_SIZE              default 16 (matches run_expert_baselines.sh); forwarded as --batch_size
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_EXPERT_ROOT_PATH        expert-scribble archive root override (--root_path,
#                                    default <repo_root>/data)
#   SCRIBBLE_ABLATION_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_ABLATION_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_ABLATION_CSV              default "<results_root>/ta_ema_ablation_summary_expert.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_voxtrust3d_2d_expert.py call
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_pce_2d_expert.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only (the default
# warmup/rampup/late-phase absolute values only make sense at 35000
# iterations, so scale them down too for a tiny smoke run):
#   SCRIBBLE_EXPERT_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_MAX_ITERATIONS=8 SCRIBBLE_WARMUP_ITERS=2 SCRIBBLE_RAMPUP_ITERS=2 SCRIBBLE_LATE_PHASE_START=4 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--early_interval 4 --late_interval 4" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_voxtrust3d_taema_ablation_expert.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_EXPERT_DATASETS:-ACDC MSCMR})
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-32000}"
warmup_iters="${SCRIBBLE_WARMUP_ITERS:-3000}"
rampup_iters="${SCRIBBLE_RAMPUP_ITERS:-6000}"
late_phase_start="${SCRIBBLE_LATE_PHASE_START:-20000}"
# warmup_frac/rampup_frac are fractions of --max_iterations, so they must be
# recomputed here to keep the *absolute* warm-up/ramp-up iteration counts
# fixed even as --max_iterations changes (see header comment).
warmup_frac="$(awk -v w="$warmup_iters" -v m="$max_iterations" 'BEGIN { printf "%.10f", w / m }')"
rampup_frac="$(awk -v r="$rampup_iters" -v m="$max_iterations" 'BEGIN { printf "%.10f", r / m }')"
batch_size="${SCRIBBLE_BATCH_SIZE:-16}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
root_path_flag="${SCRIBBLE_EXPERT_ROOT_PATH:+--root_path $SCRIBBLE_EXPERT_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_ABLATION_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_ABLATION_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_ABLATION_CSV:-$results_root/ta_ema_ablation_summary_expert.csv}"

run_arm() {
  # args: dataset ta_ema_flag stage_label
  local dataset="$1" ta_ema_flag="$2" stage_label="$3"
  local ckpt_dir results_dir
  ckpt_dir="$checkpoint_root/ExpertScribble_VoxTrust3D_ablation/$dataset/$stage_label"
  results_dir="$results_root/ExpertScribble_VoxTrust3D_ablation/$dataset/$stage_label"

  echo "=== [VoxTrust3D/$stage_label] dataset=$dataset (expert) seed=$seed max_iterations=$max_iterations" \
       "(warmup_iters=$warmup_iters rampup_iters=$rampup_iters late_phase_start=$late_phase_start): train ==="
  python "$script_dir/train_voxtrust3d_2d_expert.py" --dataset "$dataset" --seed "$seed" \
    --ta_ema "$ta_ema_flag" --max_iterations "$max_iterations" \
    --warmup_frac "$warmup_frac" --rampup_frac "$rampup_frac" --late_phase_start "$late_phase_start" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  echo "=== [VoxTrust3D/$stage_label] dataset=$dataset (expert): test ==="
  python "$test_dir/test_pce_2d_expert.py" \
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
