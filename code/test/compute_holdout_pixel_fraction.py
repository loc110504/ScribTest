"""Realized held-out PIXEL fraction for a nominal stroke fraction eta
(Table 3, "Annotation-allocation study", ``paper_icassp2027/main.tex``).

``utils.voxtrust3d.spatially_blocked_partition`` splits each class's
scribble *strokes* (connected components, one per class per slice) into
Omega_sup/Omega_cal, holding out ``round(n_blocks * eta)`` whole strokes --
never a class's last one. Because strokes have unequal pixel counts, the
fraction of annotated *pixels* actually withheld generally differs slightly
from the nominal ``eta`` passed as ``--holdout_fraction``. This is a pure
statistic of the data plus the partition rule -- no model, no training -- so
it is computed once, directly, over every training slice at the SAME
``--seed`` a real training run would use, reusing the exact per-slice
``stable_seed(seed, slice_id)`` construction ``VoxTrustSlice2DDataset``
(``train/train_voxtrust3d_2d.py``) uses, so the numbers this script reports
are bit-identical to what that training run's own partition realized.

Table 3's "Held-out strokes (%)" column is the nominal ``eta`` you pass in
(5/15/30); this script reports the realized "Held-out pixels (%)" column
only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_2d import ScribbleBench2DDataset  # noqa: E402
from train.train_pce_2d import resolve_case_split  # noqa: E402
from utils.voxtrust3d import spatially_blocked_partition, stable_seed  # noqa: E402


def realized_holdout_pixel_fraction(raw_dataset, train_slice_positions, holdout_fraction, seed):
    """Pooled held-out pixel fraction (in ``[0, 1]``) over every given flat
    slice position of ``raw_dataset`` (a :class:`ScribbleBench2DDataset`),
    at one ``holdout_fraction``/``seed`` -- the same two values a real
    training run passes to ``VoxTrustSlice2DDataset``.

    Returns ``(pixel_fraction, total_pixels, held_out_pixels)``; the last
    two let a caller sanity-check against ``total_pixels == 0`` (undefined
    fraction) instead of silently dividing by an epsilon.
    """
    total_pixels = 0
    held_out_pixels = 0
    for position in train_slice_positions:
        volume_index, slice_index = raw_dataset.slice_index[position]
        slice_id = "{}_{}".format(raw_dataset.cases[volume_index], slice_index)
        label = raw_dataset.labels[volume_index][slice_index]
        rng = np.random.default_rng(stable_seed(seed, slice_id))
        sup_coords, cal_coords, _ = spatially_blocked_partition(
            label, raw_dataset.ignore_index, raw_dataset.num_classes, holdout_fraction, rng
        )
        for class_id in range(raw_dataset.num_classes):
            n_sup = len(sup_coords[class_id])
            n_cal = len(cal_coords[class_id])
            total_pixels += n_sup + n_cal
            held_out_pixels += n_cal
    if total_pixels == 0:
        raise RuntimeError("no annotated scribble pixels found across the given slices")
    return held_out_pixels / total_pixels, total_pixels, held_out_pixels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("ACDC", "MSCMR"))
    parser.add_argument("--root_path", default=None)
    parser.add_argument(
        "--holdout_fraction", type=float, nargs="+", required=True,
        help="one or more eta values, e.g. --holdout_fraction 0.05 0.15 0.30",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dataset = ScribbleBench2DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", transform=None
    )
    train_case_indices, _, _, _, _ = resolve_case_split(raw_dataset.cases, args.dataset)
    train_slice_positions = raw_dataset.slice_positions_for_volumes(train_case_indices)

    rows = []
    for holdout_fraction in args.holdout_fraction:
        fraction, total_pixels, held_out_pixels = realized_holdout_pixel_fraction(
            raw_dataset, train_slice_positions, holdout_fraction, args.seed
        )
        row = {
            "dataset": args.dataset,
            "seed": args.seed,
            "nominal_stroke_fraction_pct": 100.0 * holdout_fraction,
            "realized_pixel_fraction_pct": 100.0 * fraction,
            "total_annotated_pixels": total_pixels,
            "held_out_pixels": held_out_pixels,
        }
        rows.append(row)
        print(
            "{} eta={:.2f} -> held-out pixels: {:.4f}% ({}/{})".format(
                args.dataset, holdout_fraction, row["realized_pixel_fraction_pct"], held_out_pixels, total_pixels
            )
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print("Wrote {}".format(output_path))
    return rows


if __name__ == "__main__":
    main()
