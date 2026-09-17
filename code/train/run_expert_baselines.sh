#!/usr/bin/env bash
# Train + test pCE, CycleMix, SDT-Net, EFFDNet, SC-MT and DMSPS (stage 1 and
# stage 2) on the "expert scribble" ACDC/MSCMR archive
# (<repo_root>/data/{ACDC,MSCMR}, the original WSL4MIS/CycleMix h5 dataset),
# plus ModelMix jointly on ACDC+MSCMR, appending every result to one summary
# CSV -- the *_2d_expert.py counterpart of run_baselines.sh, kept as a
# separate script (not folded into run_baselines.sh) so the two data
# sources' sweeps can be called independently. See
# code/dataloader/expert_scribble_2d.py's module docstring for exactly how
# this archive differs from dataset/ScribbleBench (same 4-class label
# convention and published train/val/test patient split, different literal
# scribble/preprocessing source).
#
# VoxTrust-3D is deliberately not part of this sweep (its
# train_voxtrust3d_2d_expert.py script still exists and can be run
# standalone, see its --help, but is not wired in here).
#
# This archive is 2D-only (WORD has no expert-scribble counterpart here), so
# unlike run_baselines.sh there is no 2D/3D dimension split to route.
#
#   1. train_pce_2d_expert.py       -> test_pce_2d_expert.py
#   2. train_cyclemix_2d_expert.py  -> test_pce_2d_expert.py
#   3. train_sdtnet_2d_expert.py    -> test_pce_2d_expert.py
#   4. train_effdnet_2d_expert.py   -> test_pce_2d_expert.py
#   5. train_scmt_2d_expert.py      -> test_pce_2d_expert.py
#   6. train_dmsps_2d_expert.py     -> test_dmsps_2d_expert.py (--stage 1/2)
#
# pCE/CycleMix/SDT-Net/EFFDNet/SC-MT all checkpoint a plain UNet2D (SC-MT's
# deployed model is its EMA teacher, EFFDNet's is its student) -- so all five
# share test_pce_2d_expert.py; DMSPS's dual-decoder DB-Net has its own
# evaluator, test_dmsps_2d_expert.py.
#
# ModelMix (Zhang & Patel, MICCAI 2024) is run separately, once, after the
# per-dataset loop below: it always jointly trains ACDC+MSCMR in one run (no
# --dataset flag) and is skipped unless both are in `datasets` below.
#
# Every train+test cycle appends one row to the results CSV (default:
# results/expert_baselines_summary.csv) via append_metrics_csv.py. The CSV is
# append-only across script runs, same as run_baselines.sh.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_EXPERT_DATASETS          space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_BATCH_SIZE               default 16; forwarded as --batch_size to every train_*.py call
#   SCRIBBLE_NUM_WORKERS              default 4; forwarded as --num_workers to every train_*.py call
#   SCRIBBLE_AMP_FLAG                 default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                   e.g. "cpu"; forwarded as --device to every command
#   SCRIBBLE_EXPERT_ROOT_PATH         expert-scribble archive root override (--root_path,
#                                     default <repo_root>/data)
#   SCRIBBLE_EXPERT_CHECKPOINT_ROOT   default "<repo>/checkpoints"
#   SCRIBBLE_EXPERT_RESULTS_ROOT      default "<repo>/results"
#   SCRIBBLE_EXPERT_CSV               default "<results_root>/expert_baselines_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_*.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only:
#   SCRIBBLE_EXPERT_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 \
#   SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--max_iterations 4 --early_interval 2 --late_interval 2" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_expert_baselines.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_EXPERT_DATASETS:-ACDC MSCMR})
batch_size="${SCRIBBLE_BATCH_SIZE:-16}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
root_path_flag="${SCRIBBLE_EXPERT_ROOT_PATH:+--root_path $SCRIBBLE_EXPERT_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_EXPERT_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_EXPERT_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_EXPERT_CSV:-$results_root/expert_baselines_summary.csv}"

append_row() {
  # args: method dataset stage checkpoint metrics_json
  python "$test_dir/append_metrics_csv.py" \
    --method "$1" --dataset "$2" --stage "$3" --checkpoint "$4" --metrics_json "$5" --csv "$csv_path"
}

train_and_test() {
  # args: method_label dataset stage train_checkpoint_dir train_results_dir extra_train_flags...
  local method_label="$1" dataset="$2" stage="$3" ckpt_dir="$4" results_dir="$5"
  shift 5

  echo "=== [$method_label] dataset=$dataset (expert) stage=${stage:-none}: train ==="
  case "$method_label" in
    pCE)
      python "$script_dir/train_pce_2d_expert.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    CycleMix)
      python "$script_dir/train_cyclemix_2d_expert.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    SDTNet)
      python "$script_dir/train_sdtnet_2d_expert.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    EFFDNet)
      python "$script_dir/train_effdnet_2d_expert.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    SCMT)
      python "$script_dir/train_scmt_2d_expert.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    DMSPS)
      python "$script_dir/train_dmsps_2d_expert.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    *)
      echo "Unknown method_label: $method_label" >&2
      exit 1
      ;;
  esac

  echo "=== [$method_label] dataset=$dataset (expert) stage=${stage:-none}: test ==="
  case "$method_label" in
    pCE|CycleMix|SDTNet|EFFDNet|SCMT)
      python "$test_dir/test_pce_2d_expert.py" \
        --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" \
        $amp_flag $device_flag $root_path_flag $extra_test_args
      ;;
    DMSPS)
      python "$test_dir/test_dmsps_2d_expert.py" \
        --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" \
        $amp_flag $device_flag $root_path_flag $extra_test_args
      ;;
  esac

  append_row "$method_label" "$dataset" "$stage" "$ckpt_dir/best.pth" "$results_dir/metrics.json"
}

for dataset in "${datasets[@]}"; do
  train_and_test pCE "$dataset" "" \
    "$checkpoint_root/ExpertScribble_pCE/$dataset" \
    "$results_root/ExpertScribble_pCE/$dataset"

  train_and_test CycleMix "$dataset" "" \
    "$checkpoint_root/ExpertScribble_CycleMix/$dataset" \
    "$results_root/ExpertScribble_CycleMix/$dataset"

  train_and_test SDTNet "$dataset" "" \
    "$checkpoint_root/ExpertScribble_SDTNet/$dataset" \
    "$results_root/ExpertScribble_SDTNet/$dataset"

  train_and_test EFFDNet "$dataset" "" \
    "$checkpoint_root/ExpertScribble_EFFDNet/$dataset" \
    "$results_root/ExpertScribble_EFFDNet/$dataset"

  train_and_test SCMT "$dataset" "" \
    "$checkpoint_root/ExpertScribble_SCMT/$dataset" \
    "$results_root/ExpertScribble_SCMT/$dataset"

  dmsps_stage1_ckpt_dir="$checkpoint_root/ExpertScribble_DMSPS/$dataset/stage1"
  train_and_test DMSPS "$dataset" stage1 \
    "$dmsps_stage1_ckpt_dir" \
    "$results_root/ExpertScribble_DMSPS/$dataset/stage1" \
    --stage 1

  train_and_test DMSPS "$dataset" stage2 \
    "$checkpoint_root/ExpertScribble_DMSPS/$dataset/stage2" \
    "$results_root/ExpertScribble_DMSPS/$dataset/stage2" \
    --stage 2 --init_checkpoint "$dmsps_stage1_ckpt_dir/best.pth"
done

has_dataset() {
  local wanted="$1"
  for dataset in "${datasets[@]}"; do
    [ "$dataset" = "$wanted" ] && return 0
  done
  return 1
}

if has_dataset ACDC && has_dataset MSCMR; then
  echo "=== [ModelMix] ACDC+MSCMR (expert, joint): train ==="
  modelmix_ckpt_dir="$checkpoint_root/ExpertScribble_ModelMix"
  python "$script_dir/train_modelmix_2d_expert.py" \
    --output_dir "$modelmix_ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  for dataset in ACDC MSCMR; do
    echo "=== [ModelMix] dataset=$dataset (expert): test ==="
    results_dir="$results_root/ExpertScribble_ModelMix/$dataset"
    python "$test_dir/test_pce_2d_expert.py" \
      --checkpoint "$modelmix_ckpt_dir/$dataset/best.pth" --output_dir "$results_dir" \
      $amp_flag $device_flag $root_path_flag $extra_test_args
    append_row ModelMix "$dataset" "" "$modelmix_ckpt_dir/$dataset/best.pth" "$results_dir/metrics.json"
  done
else
  echo "Skipping ModelMix: requires both ACDC and MSCMR in SCRIBBLE_EXPERT_DATASETS (got: ${datasets[*]})"
fi

echo "Done. Summary CSV: $csv_path"
