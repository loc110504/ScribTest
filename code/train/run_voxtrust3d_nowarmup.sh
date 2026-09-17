#!/usr/bin/env bash
# Train + test VoxTrust-3D on ACDC and MSCMR with NO scribble-only warm-up:
# calibration and the pseudo-label loss are active from iteration 0, and
# lambda(t) ramps from 0 up to --pseudo_loss_weight over the first 20000
# iterations of a 30000-iteration run (--rampup_frac 0.6667), instead of the
# default warm-up-then-ramp schedule in run_voxtrust3d.sh.
#
# Writes to separate checkpoint/results directories and a separate summary
# CSV from run_voxtrust3d.sh's default run, so a default-warmup run and this
# no-warmup run never overwrite each other's outputs.
#
# This just forwards to run_voxtrust3d.sh with the right environment
# variables set (see that script's header for the full list of overrides);
# any extra CLI args given here are appended on top of --warmup_frac/
# --rampup_frac/--pseudo_loss_weight (still overridable by repeating them).
#
# Usage (run on your GPU server, not this session):
#   bash code/train/run_voxtrust3d_nowarmup.sh
# Optional overrides, e.g. a different batch size or device:
#   SCRIBBLE_BATCH_SIZE=4 bash code/train/run_voxtrust3d_nowarmup.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"

export SCRIBBLE_DATASETS="${SCRIBBLE_DATASETS:-ACDC MSCMR}"
export SCRIBBLE_DEVICE="${SCRIBBLE_DEVICE:-cuda}"
export SCRIBBLE_BATCH_SIZE="${SCRIBBLE_BATCH_SIZE:-8}"
export SCRIBBLE_AMP_FLAG="${SCRIBBLE_AMP_FLAG-}"
export SCRIBBLE_EXTRA_TRAIN_ARGS="--warmup_frac 0.0 --rampup_frac 0.6667 --pseudo_loss_weight 8.0 ${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"

export SCRIBBLE_VOXTRUST3D_CHECKPOINT_ROOT="${SCRIBBLE_VOXTRUST3D_CHECKPOINT_ROOT:-$repo_dir/checkpoints_nowarmup}"
export SCRIBBLE_VOXTRUST3D_RESULTS_ROOT="${SCRIBBLE_VOXTRUST3D_RESULTS_ROOT:-$repo_dir/results_nowarmup}"
export SCRIBBLE_VOXTRUST3D_CSV="${SCRIBBLE_VOXTRUST3D_CSV:-$SCRIBBLE_VOXTRUST3D_RESULTS_ROOT/voxtrust3d_nowarmup_summary.csv}"

exec bash "$script_dir/run_voxtrust3d.sh"
