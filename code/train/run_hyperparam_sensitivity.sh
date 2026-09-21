#!/usr/bin/env bash
# Hyperparameter-sensitivity sweep for ScribCal's two core calibration
# knobs -- target precision rho and the number of distance strata B --
# reproducing the "Sensitivity to target precision rho and number of
# distance strata B" line-chart figure (1-column IEEE figure: x-axis rho,
# one line per B, y-axis validation mean Dice).
#
#   rho (--target_precision) in {0.90, 0.95, 0.99}
#   B   (--distance_strata)  in {1, 3, 5}
#
# 3x3 = 9 configurations per dataset, all trained with --ablation full
# (the proposed held-out + distance-conditioned calibration; rho/B are
# meaningless for the other ablation arms). Every other calibration
# hyperparameter is held at ScribCal's own default and NOT swept here --
# eta (--holdout_fraction) is its own experiment (see
# run_table3_annotation_budget.sh's annotation-allocation study), and
# n_min/delta/buffer/block_cap/EMA decay are deliberately not tuned:
#
#   eta (--holdout_fraction)          0.15
#   n_min (--calibration_min_samples) 32
#   delta (--wilson_delta)            0.05
#   buffer (--calibration_buffer_size) 4096
#   ema_decay                         0.99
#
# Evaluated with test_voxtrust3d_ablation_2d.py against the SAME held-out
# validation split run_table2_component_ablation.sh/
# run_table3_annotation_budget.sh use (see that evaluator's own module
# docstring for why validation, not the official test split). Mean Dice is
# the metric the chart plots; PL-Acc/PL-Cov are also recorded in the CSV
# for reference but are Table 2's story, not this sensitivity figure's --
# see the figure's own caption/discussion in the paper draft for why.
#
# Self-contained: writes to its OWN checkpoint/results namespace
# (ScribbleBench_VoxTrust3D_hparam_sensitivity), independent of
# run_table2_component_ablation.sh/run_table3_annotation_budget.sh (the
# default rho=0.95/B=3 cell here duplicates run_table2_component_ablation.sh's
# scribcal_full arm under a different --output_dir; this script never
# assumes that script's output exists, for the same robustness reason
# run_table3_annotation_budget.sh's SCRIBBLE_SKIP_ETA opt-in exists).
#
# Resumable: before (re)training each (rho, B) cell, checks whether its
# checkpoint already ran to completion (last.pth's global_step reached
# --max_iterations). If so, training is skipped; if a cached metrics.json
# also already exists for it, evaluation is skipped too. If the checkpoint
# is missing/partial (e.g. an interrupted previous run), its
# ckpt_dir/results_dir are removed and the cell is trained+evaluated from
# scratch -- a partial checkpoint would otherwise make
# guard_fresh_output_dir (common_3d.py) refuse the retry. Safe to Ctrl-C
# and rerun this script.
#
# Per dataset x (rho, B), this produces:
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_hparam_sensitivity/<dataset>/rho<RR>_B<B>/
# and appends one row per (dataset, rho, B) to one summary CSV with exactly
# the columns the sensitivity chart needs (dataset, target_precision,
# distance_strata, mean_dice_pct, plus pl_acc_pct/pl_cov_pct for reference).
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC" (the figure's own scope)
#   SCRIBBLE_TARGET_PRECISIONS       space-separated rho values, default "0.90 0.95 0.99"
#   SCRIBBLE_DISTANCE_STRATA         space-separated B values, default "1 3 5"
#   SCRIBBLE_HOLDOUT_FRACTION        default 0.15; eta held fixed for every cell
#   SCRIBBLE_WILSON_DELTA            default 0.05; delta held fixed for every cell
#   SCRIBBLE_CALIBRATION_MIN_SAMPLES default 32; n_min held fixed for every cell
#   SCRIBBLE_CALIBRATION_BUFFER_SIZE default 4096; buffer held fixed for every cell
#   SCRIBBLE_EMA_DECAY               default 0.99; ema_decay held fixed for every cell
#   SCRIBBLE_SEED                    default 2026; SAME seed used for every cell
#   SCRIBBLE_MAX_ITERATIONS          default 30000; forwarded as --max_iterations to every cell
#   SCRIBBLE_BATCH_SIZE              default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_HPARAM_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_HPARAM_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_HPARAM_CSV              default "<results_root>/hparam_sensitivity_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS        extra args appended to every train_voxtrust3d_2d.py call
#   SCRIBBLE_EXTRA_EVAL_ARGS         extra args appended to every test_voxtrust3d_ablation_2d.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only, one (rho, B) cell:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_TARGET_PRECISIONS=0.95 SCRIBBLE_DISTANCE_STRATA=3 \
#   SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_MAX_ITERATIONS=8 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--warmup_frac 0.25 --rampup_frac 0.25 --early_interval 4 --late_interval 4 --late_phase_start 4" \
#   SCRIBBLE_EXTRA_EVAL_ARGS="--case_limit 2" \
#   bash code/train/run_hyperparam_sensitivity.sh

set -euo pipefail
shopt -s inherit_errexit  # without this, `set -e` does not propagate into
                          # command substitutions (e.g. evaluate()'s output
                          # captured below), so a crash inside evaluate()
                          # would otherwise be silently swallowed and only
                          # surface later as a confusing empty-string error

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC})
target_precisions=(${SCRIBBLE_TARGET_PRECISIONS:-0.90 0.95 0.99})
distance_strata_values=(${SCRIBBLE_DISTANCE_STRATA:-1 3 5})
holdout_fraction="${SCRIBBLE_HOLDOUT_FRACTION:-0.15}"
wilson_delta="${SCRIBBLE_WILSON_DELTA:-0.05}"
calibration_min_samples="${SCRIBBLE_CALIBRATION_MIN_SAMPLES:-32}"
calibration_buffer_size="${SCRIBBLE_CALIBRATION_BUFFER_SIZE:-4096}"
ema_decay="${SCRIBBLE_EMA_DECAY:-0.99}"
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-30000}"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_eval_args="${SCRIBBLE_EXTRA_EVAL_ARGS:-}"

checkpoint_root="${SCRIBBLE_HPARAM_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_HPARAM_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_HPARAM_CSV:-$results_root/hparam_sensitivity_summary.csv}"
namespace="ScribbleBench_VoxTrust3D_hparam_sensitivity"

# True iff ckpt_dir holds a checkpoint that ran to completion (last.pth's
# global_step reached its own recorded --max_iterations), not just a
# best.pth/last.pth pair left behind by a run that was interrupted partway.
is_checkpoint_complete() {
  local ckpt_dir="$1"
  [ -f "$ckpt_dir/best.pth" ] && [ -f "$ckpt_dir/last.pth" ] || return 1
  python - "$ckpt_dir/last.pth" <<'PYEOF'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
step = checkpoint.get("global_step")
max_iterations = checkpoint.get("args", {}).get("max_iterations")
complete = step is not None and max_iterations is not None and step >= max_iterations
raise SystemExit(0 if complete else 1)
PYEOF
}

# args: dataset rho B ckpt_dir results_dir
# Skips training entirely when ckpt_dir already holds a completed
# checkpoint (safe to Ctrl-C and rerun this script). Otherwise removes any
# partial/stale ckpt_dir/results_dir first -- guard_fresh_output_dir
# (common_3d.py) refuses a fresh training run into a non-empty
# --output_dir, so a half-finished checkpoint would otherwise block the
# retry rather than get replaced by it.
ensure_trained() {
  local dataset="$1" rho="$2" strata="$3" ckpt_dir="$4" results_dir="$5"
  if is_checkpoint_complete "$ckpt_dir"; then
    echo "=== dataset=$dataset rho=$rho B=$strata: checkpoint at $ckpt_dir already complete -- skipping training ===" >&2
    return
  fi
  if [ -e "$ckpt_dir" ] || [ -e "$results_dir" ]; then
    echo "Incomplete/stale checkpoint at $ckpt_dir -- removing checkpoint+results and retraining from scratch" >&2
    rm -rf "$ckpt_dir" "$results_dir"
  fi
  echo "=== dataset=$dataset rho=$rho B=$strata eta=$holdout_fraction seed=$seed" \
       "max_iterations=$max_iterations: train ===" >&2
  python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" --ta_ema 0 \
    --ablation full --holdout_fraction "$holdout_fraction" --max_iterations "$max_iterations" \
    --target_precision "$rho" --distance_strata "$strata" --wilson_delta "$wilson_delta" \
    --calibration_min_samples "$calibration_min_samples" --calibration_buffer_size "$calibration_buffer_size" \
    --ema_decay "$ema_decay" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args
}

# args: results_dir -> prints "dice pl_acc pl_cov" and succeeds iff a
# metrics.json is already there (a completed prior evaluation to reuse).
read_cached_metrics() {
  local results_dir="$1"
  [ -f "$results_dir/metrics.json" ] || return 1
  python - "$results_dir/metrics.json" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print(payload["mean_dice"], payload["pl_acc"], payload["pl_cov"])
PYEOF
}

evaluate() {
  # args: ckpt_dir results_dir -> prints "dice pl_acc pl_cov" (pl_* may be "null")
  # Reuses a cached metrics.json when present instead of re-running the
  # (slow) evaluator, so a rerun of this script after a fully completed
  # prior run is a fast no-op.
  local ckpt_dir="$1" results_dir="$2"
  local cached
  if cached="$(read_cached_metrics "$results_dir")"; then
    echo "$cached"
    return
  fi
  python "$test_dir/test_voxtrust3d_ablation_2d.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" \
    $device_flag $root_path_flag $extra_eval_args >&2
  read_cached_metrics "$results_dir"
}

append_row() {
  # args: dataset rho strata mean_dice_pct pl_acc_pct pl_cov_pct
  python - "$csv_path" "$1" "$2" "$3" "$4" "$5" "$6" <<'PYEOF'
import csv, datetime, sys
from pathlib import Path

csv_path, dataset, rho, strata, mean_dice_pct, pl_acc, pl_cov = sys.argv[1:8]
fieldnames = [
    "timestamp", "dataset", "target_precision", "distance_strata",
    "mean_dice_pct", "pl_acc_pct", "pl_cov_pct",
]
row = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "dataset": dataset,
    "target_precision": rho,
    "distance_strata": strata,
    "mean_dice_pct": mean_dice_pct,
    "pl_acc_pct": pl_acc,
    "pl_cov_pct": pl_cov,
}
path = Path(csv_path)
path.parent.mkdir(parents=True, exist_ok=True)
write_header = not path.is_file() or path.stat().st_size == 0
with path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print("Appended {}/rho={}/B={} -> {}".format(dataset, rho, strata, csv_path))
PYEOF
}

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: this hyperparameter-sensitivity figure is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  for rho in "${target_precisions[@]}"; do
    for strata in "${distance_strata_values[@]}"; do
      echo "=== dataset=$dataset rho=$rho B=$strata ==="

      ckpt_dir="$checkpoint_root/$namespace/$dataset/rho${rho}_B${strata}"
      results_dir="$results_root/$namespace/$dataset/rho${rho}_B${strata}"

      ensure_trained "$dataset" "$rho" "$strata" "$ckpt_dir" "$results_dir"

      echo "=== dataset=$dataset rho=$rho B=$strata: evaluate ===" >&2
      eval_out="$(evaluate "$ckpt_dir" "$results_dir")"
      read -r dice pl_acc pl_cov <<< "$eval_out"

      mean_dice_pct=$(python -c "print(round(float(\"$dice\") * 100, 4))")
      pl_acc_pct=$(python -c "v=\"$pl_acc\"; print('null' if v=='None' else round(float(v), 4))")
      pl_cov_pct=$(python -c "v=\"$pl_cov\"; print('null' if v=='None' else round(float(v), 4))")

      append_row "$dataset" "$rho" "$strata" "$mean_dice_pct" "$pl_acc_pct" "$pl_cov_pct"
    done
  done
done

echo "Done. Hyperparameter-sensitivity summary CSV: $csv_path"
echo "Plot directly: x=target_precision, one line per distance_strata, y=mean_dice_pct" \
     "(pl_acc_pct/pl_cov_pct kept for reference only -- Table 2's story, not this figure's)."
