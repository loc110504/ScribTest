#!/usr/bin/env bash
# Table 2 ("Ablating Trust Calibration", paper_icassp2027/main.tex) component
# ablation, on ScribbleBench (ACDC by default -- Table 2's own scope; pass
# SCRIBBLE_DATASETS="ACDC MSCMR" to also reproduce it on MSCMR).
#
# Reproduces five rows, all from the SAME --seed:
#   pce_only        pCE, no Mean Teacher at all (train_pce_2d.py)
#   mt_all_pl       MT, all pseudo-labels        (--ablation all_pseudo_labels, --holdout_fraction 0)
#   mt_global_conf  MT, global confidence        (--ablation global_confidence, --holdout_fraction 0)
#   mt_class_only   MT, class-only calibration   (--ablation class_only)
#   dcc_full        DCC (full method)            (--ablation full, the training scripts' own default)
#
# The "mt_all_pl"/"mt_global_conf" arms pass --holdout_fraction 0: they never
# consult Omega_cal (see utils/voxtrust3d.py's module docstring), so holding
# out scribbles for them would only waste annotation without being used by
# anything.
#
# Each arm writes to its OWN, never-reused --output_dir and is left to
# complete all --max_iterations uninterrupted before the next one starts
# (see run_voxtrust3d_taema_ablation.sh's header for why this matters --
# guard_fresh_output_dir, code/train/common_3d.py, refuses a fresh run into
# a directory that already holds a checkpoint, but two arms must still never
# share a directory in the first place).
#
# What this script gives you vs. what Table 2 also reports: this produces
# the Dice column (via test_pce_2d.py against the ACDC/MSCMR test split, same
# as every other baseline in this repo). It does NOT yet produce Table 2's
# PL-P (pseudo-label precision, Eq. 9) or ECE (Eq. 10) columns -- those are
# test-time diagnostics computed against ACDC's dense test masks with each
# arm's own acceptance rule reapplied at inference, which needs a dedicated
# evaluator this repo does not have yet. Comparing accepted_ratio (logged to
# TensorBoard/train.log by every arm) across arms is a rough proxy in the
# meantime, but is not a substitute for PL-P/ECE computed on the test set.
#
# Per dataset, this produces:
#   <checkpoint_root>/ScribbleBench_pCE/<dataset>/                                (pce_only; shared with the
#                                                                                   regular pCE baseline -- see below)
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_ablation/<dataset>/mt_all_pl/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_ablation/<dataset>/mt_global_conf/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_ablation/<dataset>/mt_class_only/
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_dcc_ablation/<dataset>/dcc_full/
# and appends one row per arm to the summary CSV (method=pCE/VoxTrust3D,
# stage=""/mt_all_pl/mt_global_conf/mt_class_only/dcc_full).
#
# pce_only reuses the SAME --output_dir convention as run_baselines.sh's pCE
# row (checkpoints/ScribbleBench_pCE/<dataset>), so if you already have a
# pCE baseline checkpoint from that sweep with a matching --seed, set
# SCRIBBLE_SKIP_PCE=1 to skip retraining it and just re-evaluate/append it.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC" (Table 2's own scope)
#   SCRIBBLE_SEED                    default 2026; SAME seed used for every arm
#   SCRIBBLE_MAX_ITERATIONS          default 30000; forwarded as --max_iterations to every arm
#   SCRIBBLE_GLOBAL_CONFIDENCE_THRESHOLD  default 0.75; forwarded to the mt_global_conf arm only
#   SCRIBBLE_SKIP_PCE                default 0; set to 1 to skip the pce_only arm's training step
#   SCRIBBLE_BATCH_SIZE              default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_ABLATION_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_ABLATION_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_ABLATION_CSV              default "<results_root>/dcc_ablation_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_pce_2d.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_MAX_ITERATIONS=8 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--warmup_frac 0.25 --rampup_frac 0.25 --early_interval 4 --late_interval 4 --late_phase_start 4" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_voxtrust3d_dcc_ablation.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC})
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-30000}"
global_confidence_threshold="${SCRIBBLE_GLOBAL_CONFIDENCE_THRESHOLD:-0.75}"
skip_pce="${SCRIBBLE_SKIP_PCE:-0}"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_ABLATION_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_ABLATION_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_ABLATION_CSV:-$results_root/dcc_ablation_summary.csv}"

run_pce_arm() {
  local dataset="$1"
  local ckpt_dir="$checkpoint_root/ScribbleBench_pCE/$dataset"
  local results_dir="$results_root/ScribbleBench_pCE/$dataset"

  if [ "$skip_pce" != "1" ]; then
    echo "=== [pce_only] dataset=$dataset seed=$seed max_iterations=$max_iterations: train ==="
    python "$script_dir/train_pce_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
      --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
      $amp_flag $device_flag $root_path_flag $extra_train_args
  else
    echo "=== [pce_only] dataset=$dataset: SCRIBBLE_SKIP_PCE=1, skipping training ==="
  fi

  echo "=== [pce_only] dataset=$dataset: test ==="
  python "$test_dir/test_pce_2d.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args

  python "$test_dir/append_metrics_csv.py" \
    --method pCE --dataset "$dataset" --stage "" \
    --checkpoint "$ckpt_dir/best.pth" --metrics_json "$results_dir/metrics.json" --csv "$csv_path"
}

run_voxtrust_arm() {
  # args: dataset stage_label ablation extra_arm_train_args...
  local dataset="$1" stage_label="$2" ablation="$3"
  shift 3
  local ckpt_dir results_dir
  ckpt_dir="$checkpoint_root/ScribbleBench_VoxTrust3D_dcc_ablation/$dataset/$stage_label"
  results_dir="$results_root/ScribbleBench_VoxTrust3D_dcc_ablation/$dataset/$stage_label"

  echo "=== [VoxTrust3D/$stage_label] dataset=$dataset ablation=$ablation seed=$seed" \
       "max_iterations=$max_iterations: train ==="
  python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" \
    --ablation "$ablation" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag "$@" $extra_train_args

  echo "=== [VoxTrust3D/$stage_label] dataset=$dataset: test ==="
  python "$test_dir/test_pce_2d.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args

  python "$test_dir/append_metrics_csv.py" \
    --method VoxTrust3D --dataset "$dataset" --stage "$stage_label" \
    --checkpoint "$ckpt_dir/best.pth" --metrics_json "$results_dir/metrics.json" --csv "$csv_path"
}

for dataset in "${datasets[@]}"; do
  run_pce_arm "$dataset"
  run_voxtrust_arm "$dataset" mt_all_pl all_pseudo_labels --holdout_fraction 0
  run_voxtrust_arm "$dataset" mt_global_conf global_confidence --holdout_fraction 0 \
    --global_confidence_threshold "$global_confidence_threshold"
  run_voxtrust_arm "$dataset" mt_class_only class_only
  run_voxtrust_arm "$dataset" dcc_full full
done

echo "Done. Ablation summary CSV: $csv_path"
echo "Compare rows for the same dataset across stage=''(pCE)/mt_all_pl/mt_global_conf/mt_class_only/dcc_full."
echo "Note: this CSV has Dice/HD95/ASSD only -- see this script's header for what it does not yet cover (PL-P/ECE)."
