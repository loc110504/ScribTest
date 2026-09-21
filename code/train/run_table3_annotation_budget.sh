#!/usr/bin/env bash
# Table 3 ("Annotation-allocation study", paper_icassp2027/main.tex): sweeps
# the held-out fraction eta over {5%, 15%, 30%} on ACDC (Table 3's own
# scope; pass SCRIBBLE_DATASETS="ACDC MSCMR" to also reproduce it on
# MSCMR), producing every column the table needs:
#   Held-out strokes (%)  the nominal eta itself (the input control knob)
#   Held-out pixels (%)   REALIZED fraction of annotated pixels withheld,
#                         via compute_holdout_pixel_fraction.py -- a pure
#                         data statistic, no training involved, computed at
#                         the SAME --seed the arms below train with
#   Delta-Dice (points)   ScribCal (full) minus a Mean Teacher control
#                         trained on the IDENTICAL Omega_sup at that eta
#                         (--ablation all_pseudo_labels), both Dice numbers
#                         from test_voxtrust3d_ablation_2d.py's held-out
#                         VALIDATION-split replay -- the same evaluator and
#                         split run_table2_component_ablation.sh uses, so
#                         eta=0.15's pair here is the direct, low-noise
#                         paired comparison Table 3 asks for
#   PL-Acc/PL-Cov (%)     ScribCal (full)'s own accept-rule accuracy/coverage
#                         at that eta, also from test_voxtrust3d_ablation_2d.py
#
# Self-contained: writes to its OWN checkpoint/results namespace
# (ScribbleBench_VoxTrust3D_table3_budget), independent of
# run_table2_component_ablation.sh. Note this means eta=0.15's pair here
# (scribcal_full / mt_matched_all_pl) duplicates two of that script's runs
# under a different --output_dir; if you already ran that script with the
# SAME --seed/--holdout_fraction and want to skip retraining them, set
# SCRIBBLE_SKIP_ETA=0.15 to reuse checkpoints from
# SCRIBBLE_TABLE2_CHECKPOINT_ROOT's scribcal_full/mt_matched_all_pl dirs
# instead (see the "reuse" branch below) -- off by default for robustness
# (this script never assumes another script's output exists).
#
# Per dataset x eta, this produces:
#   <checkpoint_root>/ScribbleBench_VoxTrust3D_table3_budget/<dataset>/eta<PP>/{scribcal_full,mt_matched}/
# and appends one row per (dataset, eta) to one summary CSV with exactly
# Table 3's columns.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC" (Table 3's own scope)
#   SCRIBBLE_HOLDOUT_FRACTIONS       space-separated eta values, default "0.05 0.15 0.30"
#   SCRIBBLE_SEED                    default 2026; SAME seed used for every arm and for the pixel-fraction statistic
#   SCRIBBLE_MAX_ITERATIONS          default 30000; forwarded as --max_iterations to every arm
#   SCRIBBLE_SKIP_ETA                one eta value (e.g. "0.15") whose scribcal_full/mt_matched_all_pl
#                                     checkpoints should be copied from SCRIBBLE_TABLE2_CHECKPOINT_ROOT's
#                                     scribcal_full/mt_matched_all_pl dirs instead of retraining; default ""
#                                     (always retrain everything)
#   SCRIBBLE_TABLE2_CHECKPOINT_ROOT  only read when SCRIBBLE_SKIP_ETA is set; default "<repo>/checkpoints"
#   SCRIBBLE_BATCH_SIZE              default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_TABLE3_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_TABLE3_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_TABLE3_CSV              default "<results_root>/table3_annotation_budget_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS        extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_EVAL_ARGS         extra args appended to every test_voxtrust3d_ablation_2d.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only, one eta:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_HOLDOUT_FRACTIONS=0.15 SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" \
#   SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 SCRIBBLE_MAX_ITERATIONS=8 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--warmup_frac 0.25 --rampup_frac 0.25 --early_interval 4 --late_interval 4 --late_phase_start 4" \
#   SCRIBBLE_EXTRA_EVAL_ARGS="--case_limit 2" \
#   bash code/train/run_table3_annotation_budget.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC})
etas=(${SCRIBBLE_HOLDOUT_FRACTIONS:-0.05 0.15 0.30})
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-30000}"
skip_eta="${SCRIBBLE_SKIP_ETA:-}"
table2_checkpoint_root="${SCRIBBLE_TABLE2_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_eval_args="${SCRIBBLE_EXTRA_EVAL_ARGS:-}"

checkpoint_root="${SCRIBBLE_TABLE3_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_TABLE3_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_TABLE3_CSV:-$results_root/table3_annotation_budget_summary.csv}"
namespace="ScribbleBench_VoxTrust3D_table3_budget"

evaluate() {
  # args: ckpt_dir results_dir -> prints "dice pl_acc pl_cov" (pl_* may be "null")
  local ckpt_dir="$1" results_dir="$2"
  python "$test_dir/test_voxtrust3d_ablation_2d.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" \
    $device_flag $root_path_flag $extra_eval_args >&2
  python - "$results_dir/metrics.json" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print(payload["mean_dice"], payload["pl_acc"], payload["pl_cov"])
PYEOF
}

train_scribcal_full() {
  # args: dataset eta ckpt_dir
  local dataset="$1" eta="$2" ckpt_dir="$3"
  echo "=== [scribcal_full] dataset=$dataset eta=$eta seed=$seed max_iterations=$max_iterations: train ===" >&2
  python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" --ta_ema 0 \
    --ablation full --holdout_fraction "$eta" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args
}

train_mt_matched() {
  # args: dataset eta ckpt_dir
  local dataset="$1" eta="$2" ckpt_dir="$3"
  echo "=== [mt_matched] dataset=$dataset eta=$eta seed=$seed max_iterations=$max_iterations: train ===" >&2
  python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" --ta_ema 0 \
    --ablation all_pseudo_labels --holdout_fraction "$eta" --max_iterations "$max_iterations" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args
}

append_row() {
  # args: dataset eta_pct realized_pixel_pct delta_dice pl_acc pl_cov
  python - "$csv_path" "$1" "$2" "$3" "$4" "$5" "$6" <<'PYEOF'
import csv, datetime, sys
from pathlib import Path

csv_path, dataset, eta_pct, pixel_pct, delta_dice, pl_acc, pl_cov = sys.argv[1:8]
fieldnames = [
    "timestamp", "dataset", "held_out_strokes_pct", "held_out_pixels_pct",
    "delta_dice_points", "pl_acc_pct", "pl_cov_pct",
]
row = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "dataset": dataset,
    "held_out_strokes_pct": eta_pct,
    "held_out_pixels_pct": pixel_pct,
    "delta_dice_points": delta_dice,
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
print("Appended {}/eta={}% -> {}".format(dataset, eta_pct, csv_path))
PYEOF
}

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: Table 3's annotation-allocation study is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  for eta in "${etas[@]}"; do
    eta_pct=$(python -c "print(round(float(\"$eta\") * 100, 4))")
    echo "=== dataset=$dataset eta=$eta (${eta_pct}%) ==="

    full_ckpt_dir="$checkpoint_root/$namespace/$dataset/eta${eta_pct}/scribcal_full"
    full_results_dir="$results_root/$namespace/$dataset/eta${eta_pct}/scribcal_full"
    matched_ckpt_dir="$checkpoint_root/$namespace/$dataset/eta${eta_pct}/mt_matched"
    matched_results_dir="$results_root/$namespace/$dataset/eta${eta_pct}/mt_matched"

    if [ "$skip_eta" = "$eta" ]; then
      echo "SCRIBBLE_SKIP_ETA=$eta: reusing $table2_checkpoint_root's Table 2 checkpoints instead of retraining" >&2
      full_ckpt_dir="$table2_checkpoint_root/ScribbleBench_VoxTrust3D_table2_ablation/$dataset/scribcal_full"
      matched_ckpt_dir="$table2_checkpoint_root/ScribbleBench_VoxTrust3D_table2_ablation/$dataset/mt_matched_all_pl"
    else
      train_scribcal_full "$dataset" "$eta" "$full_ckpt_dir"
      train_mt_matched "$dataset" "$eta" "$matched_ckpt_dir"
    fi

    echo "=== [scribcal_full] dataset=$dataset eta=$eta: evaluate ===" >&2
    read -r full_dice full_pl_acc full_pl_cov <<< "$(evaluate "$full_ckpt_dir" "$full_results_dir")"
    echo "=== [mt_matched] dataset=$dataset eta=$eta: evaluate ===" >&2
    read -r matched_dice matched_pl_acc matched_pl_cov <<< "$(evaluate "$matched_ckpt_dir" "$matched_results_dir")"

    delta_dice=$(python -c "print(round((float(\"$full_dice\") - float(\"$matched_dice\")) * 100, 4))")

    echo "=== dataset=$dataset eta=$eta: realized held-out pixel fraction (data statistic, no training) ===" >&2
    pixel_json="$results_root/$namespace/$dataset/eta${eta_pct}/pixel_fraction.json"
    python "$test_dir/compute_holdout_pixel_fraction.py" \
      --dataset "$dataset" --holdout_fraction "$eta" --seed "$seed" \
      $root_path_flag --output_json "$pixel_json"
    pixel_pct=$(python -c "import json; print(round(json.load(open(\"$pixel_json\"))[0][\"realized_pixel_fraction_pct\"], 4))")

    full_pl_acc_pct=$(python -c "v=\"$full_pl_acc\"; print('null' if v=='None' else round(float(v), 4))")
    full_pl_cov_pct=$(python -c "v=\"$full_pl_cov\"; print('null' if v=='None' else round(float(v), 4))")

    append_row "$dataset" "$eta_pct" "$pixel_pct" "$delta_dice" "$full_pl_acc_pct" "$full_pl_cov_pct"
  done
done

echo "Done. Table 3 summary CSV: $csv_path"
echo "Fill main.tex's Table 3 (tab:budget) directly: held_out_strokes_pct/held_out_pixels_pct/delta_dice_points/pl_acc_pct+pl_cov_pct per row."
